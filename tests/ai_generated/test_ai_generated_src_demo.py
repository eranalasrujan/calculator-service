from extraction_tool.config.prompts import get_prompts
from extraction_tool.core.vector_db_handler import VectorDBHandler
from extraction_tool.models import TableSchema
from src.demo import AskLyricConversation
from src.demo import calculate_payment
from src.demo import validate_signup
from src.demo import validate_user_api_payload
from typing import List, Dict, Any
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock
import os
import pytest

def test_validate_signup():
    assert validate_signup("john", "john@example.com", 25, "password123") == True
    assert validate_signup("", "john@example.com", 25, "password123") == False
    assert validate_signup("john", "john.com", 25, "password123") == False
    assert validate_signup("john", "john@example", 25, "password123") == False
    assert validate_signup("john", "john@example.com", 17, "password123") == False
    assert validate_signup("john", "john@example.com", 25, "password") == False
    with pytest.raises(TypeError):
        validate_signup(123, "john@example.com", 25, "password123")
    with pytest.raises(TypeError):
        validate_signup("john", 123, 25, "password123")
    with pytest.raises(TypeError):
        validate_signup("john", "john@example.com", "25", "password123")
    with pytest.raises(TypeError):
        validate_signup("john", "john@example.com", 25, 123)

def test_calculate_payment():
    assert calculate_payment(10, "premium") == 8.0
    assert calculate_payment(10, "vip") == 7.0
    assert calculate_payment(10, "basic") == 10.0
    with pytest.raises(ValueError):
        calculate_payment(-10, "premium")
    with pytest.raises(ValueError):
        calculate_payment(10, "invalid")

def test_validate_user_api_payload():
    assert validate_user_api_payload({"name": "John", "email": "john@example.com", "age": 25}) == True
    assert validate_user_api_payload({"name": "Jane", "email": "jane@example.com", "age": 30}) == True
    assert validate_user_api_payload({"name": "Bob", "email": "bob@example.com"}) == False
    assert validate_user_api_payload({"name": "Alice", "email": "alice@example"}) == False
    assert validate_user_api_payload({"name": "Charlie", "email": "charlie@example.com", "age": 17}) == False
    with pytest.raises(KeyError):
        validate_user_api_payload({"name": "David", "email": "david@example.com"})

def test_AskLyricConversation__init__():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = 'mock_openai_key'
        with patch('extraction_tool.core.llm_handler.LLMHandler') as mock_llm_handler:
            mock_llm_handler.return_value = MagicMock()
            conversation = AskLyricConversation()
            assert conversation.llm_handler.api_key == 'mock_openai_key'
            assert conversation.llm_handler.provider == 'openai'
            assert conversation.llm_handler.model_name == 'gpt-4'
            assert conversation.icd_handler is None
            assert conversation.cpt_handler is None
            assert conversation.hcpcs_handler is None

    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = 'mock_openai_key'
        with patch('extraction_tool.core.llm_handler.LLMHandler') as mock_llm_handler:
            mock_llm_handler.return_value = MagicMock()
            conversation = AskLyricConversation()
            assert conversation.system_instruction == PROMPTS_CONFIG.get("ask_lyric_prompt", {}).get("system_instructions", "")

    with pytest.raises(KeyError):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = None
            with patch('extraction_tool.core.llm_handler.LLMHandler') as mock_llm_handler:
                mock_llm_handler.return_value = MagicMock()
                conversation = AskLyricConversation()
                assert conversation.llm_handler is None

def test_get_context():
    # Test with valid query and codes
    conversation = AskLyricConversation()
    system_instruction, prompt = conversation.get_context(query="What is the meaning of life?", codes=[{"code": "123"}])
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)

    # Test with invalid query
    with pytest.raises(Exception):
        conversation.get_context(query="")

    # Test with None query
    with pytest.raises(Exception):
        conversation.get_context(query=None)

    # Test with empty codes
    conversation.get_context(query="What is the meaning of life?", codes=[])

    # Test with None codes
    conversation.get_context(query="What is the meaning of life?", codes=None)

    # Test with invalid codes
    with pytest.raises(Exception):
        conversation.get_context(query="What is the meaning of life?", codes=[{"code": "abc"}])

    # Test with valid query and codes, but error in fetching context
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context = lambda *args, **kwargs: None
        conversation.get_context(query="What is the meaning of life?", codes=[{"code": "123"}])

def test_get_ask_lyric_common_code_columns():
    ask_lyric_conversation = AskLyricConversation()
    assert ask_lyric_conversation.get_ask_lyric_common_code_columns() == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }
    with pytest.raises(AttributeError):
        ask_lyric_conversation.get_ask_lyric_common_code_columns(None)
    with pytest.raises(TypeError):
        ask_lyric_conversation.get_ask_lyric_common_code_columns("invalid_input")

def test_get_ask_lyric_table_schemas():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert isinstance(schemas[0], TableSchema)
    assert isinstance(schemas[1], TableSchema)
    assert isinstance(schemas[2], TableSchema)
    assert schemas[0].table_name == "medical_codes_icd"
    assert schemas[1].table_name == "medical_codes_cpt"
    assert schemas[2].table_name == "medical_codes_hcpcs"
    assert len(schemas[0].columns) > 0
    assert len(schemas[1].columns) > 0
    assert len(schemas[2].columns) > 0
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas().table_name = "test"

def test_get_db_handlers():
    conversation = AskLyricConversation()
    conversation.get_ask_lyric_table_schemas.return_value = [
        TableSchema(name="ICD", columns=["column1", "column2"]),
        TableSchema(name="CPT", columns=["column3", "column4"]),
        TableSchema(name="HCPCS", columns=["column5", "column6"]),
    ]

    conversation.get_db_handlers()

    assert conversation.icd_handler is not None
    assert conversation.cpt_handler is not None
    assert conversation.hcpcs_handler is not None

    assert isinstance(conversation.icd_handler, VectorDBHandler)
    assert isinstance(conversation.cpt_handler, VectorDBHandler)
    assert isinstance(conversation.hcpcs_handler, VectorDBHandler)

    assert len(conversation.icd_handler.schemas) == 1
    assert len(conversation.cpt_handler.schemas) == 1
    assert len(conversation.hcpcs_handler.schemas) == 1

    with pytest.raises(AttributeError):
        del conversation.icd_handler

    with pytest.raises(AttributeError):
        del conversation.cpt_handler

    with pytest.raises(AttributeError):
        del conversation.hcpcs_handler

def test_fetch_codes_from_query_rag():
    conversation = AskLyricConversation()
    assert fetch_codes_from_query_rag(conversation, "", 10, 15) == []
    assert fetch_codes_from_query_rag(conversation, "short", 10, 15) == []
    assert fetch_codes_from_query_rag(conversation, "long query", 10, 15) != []
    assert fetch_codes_from_query_rag(conversation, "long query", 0, 15) == []
    assert fetch_codes_from_query_rag(conversation, "long query", 10, 0) == []
    with pytest.raises(Exception):
        fetch_codes_from_query_rag(conversation, "long query", -10, 15)
    with pytest.raises(Exception):
        fetch_codes_from_query_rag(conversation, "long query", 10, -15)

def fetch_codes_from_query_rag(conversation, question: str, top_k: int = 10, max_total_results: int = 15) -> List[Dict[str, Any]]:
    try:
        if not question or len(question.strip()) < 3:
            logging.info("Query too short for RAG search")
            return []

        logging.info(f"Starting RAG-based code search for query: {question[:100]}...")

        all_matches = []

        # Search all 3 handlers
        handlers = [
            (conversation.icd_handler, 'ICD'),
            (conversation.cpt_handler, 'CPT'),
            (conversation.hcpcs_handler, 'HCPCS')
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
                    logging.info(f"Found {len(matches)} {code_type} matches")
            except Exception as e:
                logging.warning(f"Error searching {code_type} codes: {str(e)}")

        # Sort by similarity score and return top matches
        all_matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        top_matches = all_matches[:max_total_results]

        logging.info(f"RAG search completed: {len(top_matches)} total matches found")
        return top_matches

    except Exception as e:
        logging.error(f"Error in RAG code search: {str(e)}")
        return []

def test_fetch_ask_lyric_context():
    conversation = AskLyricConversation()
    assert conversation.fetch_ask_lyric_context(query="") == {}
    assert conversation.fetch_ask_lyric_context(query="test", top_k=5) == {}
    assert conversation.fetch_ask_lyric_context(query="test", mentioned_medical_codes=[{"code": "123"}]) == {
        "mentioned_medical_codes": [{"code": "123"}],
        "rag_fetched_medical_codes": []
    }
    assert conversation.fetch_ask_lyric_context(query="test", top_k=5, max_total_results=20) == {}
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query="test", mentioned_medical_codes=[{"code": "123"}], top_k=-1)
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query="test", mentioned_medical_codes=[{"code": "123"}], max_total_results=-1)
