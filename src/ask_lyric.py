import os
import json

import logging

from app.models import AskLyricRequest
from extraction_tool.core.vector_db_handler import VectorDBHandler
from extraction_tool.core.llm_handler import LLMHandler
from extraction_tool.config.prompts import get_prompts
from extraction_tool.models import TableSchema
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Load prompts configuration once at module level
PROMPTS_CONFIG = get_prompts()

class AskLyricConversation:
    def __init__(self):
        # Initialize LLM handler with OpenAI for Ask Lyric
        openai_key = os.getenv("OPENAI_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.llm_handler = LLMHandler(api_key=openai_key, provider="openai", model_name=openai_model)
        self.icd_handler=None
        self.cpt_handler = None
        self.hcpcs_handler = None
        self.get_db_handlers()

        # Load formatting prompt from config
        self.system_instruction = PROMPTS_CONFIG.get("ask_lyric_prompt", {}).get("system_instructions", "")

    async def get_context(self, query:str, codes: Optional[List[Dict[str, Any]]] = None):
        try:
            context  = self.fetch_ask_lyric_context(query=query,mentioned_medical_codes=codes)

            # Get enhanced prompt template from config
            enhanced_prompt_template = PROMPTS_CONFIG.get("ask_lyric_prompt", {}).get("enhanced_prompt", "")
            system_instruction=self.system_instruction.replace("{context}", json.dumps(context, indent=2))

            # Replace placeholders with actual values
            enhanced_prompt = enhanced_prompt_template.replace("{question}", query)

            prompt = enhanced_prompt
            return system_instruction, prompt
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            raise

    async def make_conversation_poc(self, payload:AskLyricRequest):
        question=payload.question
        codes, missing_codes = self.icd_handler.fetch_medical_codes_from_codemaster(query=question)

        if len(codes) < 7:
            try:
                system_instruction, prompt = await self.get_context(query=question,codes=codes)
                logger.info(f"Starting LLM stream for question: {question[:50]}...")

                # Use LLM handler for streaming
                async for sse_line in self.llm_handler.stream_completion(system_instructions=system_instruction,prompt=prompt, temperature=0.7):
                    yield sse_line

                logger.info("LLM stream completed successfully")
            except Exception as e:
                logger.error(f"LLM streaming error: {str(e)}")
                raise
        else:
            # intro message
            intro_msg=f"**Listed Medical codes**\n"
            sse_line = f'data: {json.dumps({"content": intro_msg})}\n\n'
            yield sse_line

            # Table header
            table_header = (
                "| Code | Type | Description |\n"
                "|------|------|-------------|\n"
            )
            yield f'data: {json.dumps({"content": table_header})}\n\n'

            for code in codes:
                c, t, d = code['code'], code['type'], code['description']
                # Escape pipes to avoid breaking the table
                d = d.replace("|", "\\|")
                row = f"| **{c}** | {t} | {d} |\n"
                # Proper SSE format: data: <content>\n\n
                yield f'data: {json.dumps({"content": row})}\n\n'

            # Check for missing codes and yield a message if any
            if missing_codes:
                missing_msg = f"\n**Note:** The following codes are not listed: **{', '.join(missing_codes)}**\n\n"
                sse_line = f'data: {json.dumps({"content": missing_msg})}\n\n'
                yield sse_line

    def get_ask_lyric_common_code_columns(self):
        """Get the common code columns schema for Ask Lyric"""
        return {
            "id": "SERIAL PRIMARY KEY",
            "code": "TEXT NOT NULL",
            "description": "TEXT NOT NULL"
        }

    def get_ask_lyric_table_schemas(self) -> List[TableSchema]:
        """Get list of TableSchema objects for ICD, CPT, and HCPCS tables"""
        tables = ["medical_codes_icd", "medical_codes_cpt", "medical_codes_hcpcs"]
        schemas = []
        for table in tables:
            schema = TableSchema(
                table_name=table,
                columns=self.get_ask_lyric_common_code_columns()
            )
            schemas.append(schema)
        return schemas

    def get_db_handlers(self):
        """Initialize VectorDBHandler for each medical code table"""
        schemas = self.get_ask_lyric_table_schemas()

        # Use get_or_create_handler for caching
        self.icd_handler = VectorDBHandler.get_or_create_handler(schemas[0])
        self.cpt_handler = VectorDBHandler.get_or_create_handler(schemas[1])
        self.hcpcs_handler = VectorDBHandler.get_or_create_handler(schemas[2])

        logger.info("Initialized 3 VectorDBHandlers for ICD, CPT, and HCPCS tables")

    def fetch_codes_from_query_rag(self, question:str, top_k: int = 10, max_total_results: int = 15) -> List[Dict[str, Any]]:
        try:
            if not question or len(question.strip()) < 3:
                logger.info("Query too short for RAG search")
                return []

            logger.info(f"Starting RAG-based code search for query: {question[:100]}...")

            all_matches = []

            # Search all 3 handlers
            handlers = [
                (self.icd_handler, 'ICD'),
                (self.cpt_handler, 'CPT'),
                (self.hcpcs_handler, 'HCPCS')
            ]

            for handler, code_type in handlers:
                try:
                    results = handler.search_similar_codes(query_description=question, top_k=top_k)
                    if results and len(results) > 0:
                        matches = results[0].get('matches', [])
                        for match in matches:
                            all_matches.append({
                                'code': match.get('code', ''),
                                'description': match.get('description', ''),
                                'type': code_type,
                                'similarity_score': 1 - match.get('distance', 1.0),
                                'source': 'rag_embedding_search'
                            })
                        logger.info(f"Found {len(matches)} {code_type} matches")
                except Exception as e:
                    logger.warning(f"Error searching {code_type} codes: {str(e)}")

            # Sort by similarity score and return top matches
            all_matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            top_matches = all_matches[:max_total_results]

            logger.info(f"RAG search completed: {len(top_matches)} total matches found")
            return top_matches

        except Exception as e:
            logger.error(f"Error in RAG code search: {str(e)}")
            return []

    def fetch_ask_lyric_context(self, query:str, mentioned_medical_codes: Optional[List[Dict[str, Any]]] = None, top_k: int = 10, max_total_results: int = 15) -> Dict[str, Any]:
        try:
            # 1. Fetch medical codes using RAG
            rag_medical_codes = self.fetch_codes_from_query_rag(
                question=query,
                top_k=top_k,
                max_total_results=max_total_results
            )

            # Build context object
            context = {}
            if mentioned_medical_codes or rag_medical_codes:
                context = {
                    "mentioned_medical_codes": mentioned_medical_codes or [],
                    "rag_fetched_medical_codes": rag_medical_codes or []
                }

            return context

        except Exception as e:
            logger.error(f"Error fetching Ask Lyric context: {str(e)}")
            raise