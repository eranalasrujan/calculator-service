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

def test_make_conversation_poc():
    conversation = AskLyricConversation()
    payload = AskLyricRequest(question="What is the meaning of life?")
    with patch('extraction_tool.core.vector_db_handler.VectorDBHandler.fetch_medical_codes_from_codemaster') as mock_fetch_codes:
        mock_fetch_codes.return_value = ([{"code": "123", "type": "type1", "description": "desc1"}], [])
        with patch('extraction_tool.core.llm_handler.LLMHandler.stream_completion') as mock_stream_completion:
            mock_stream_completion.return_value = [MagicMock()]
            result = conversation.make_conversation_poc(payload)
            assert len(result) == 1
            assert result[0].startswith('data: ')

    with patch('extraction_tool.core.vector_db_handler.VectorDBHandler.fetch_medical_codes_from_codemaster') as mock_fetch_codes:
        mock_fetch_codes.return_value = ([], ["missing_code"])
        with patch('extraction_tool.core.llm_handler.LLMHandler.stream_completion') as mock_stream_completion:
            mock_stream_completion.return_value = [MagicMock()]
            result = conversation.make_conversation_poc(payload)
            assert len(result) == 3
            assert result[1].startswith('data: **Listed Medical codes**\n')
            assert result[2].startswith('data: **Note:** The following codes are not listed: **missing_code**\n')

    with patch('extraction_tool.core.vector_db_handler.VectorDBHandler.fetch_medical_codes_from_codemaster') as mock_fetch_codes:
        mock_fetch_codes.return_value = ([], [])
        with patch('extraction_tool.core.llm_handler.LLMHandler.stream_completion') as mock_stream_completion:
            mock_stream_completion.return_value = [MagicMock()]
            result = conversation.make_conversation_poc(payload)
            assert len(result) == 2
            assert result[0].startswith('data: **Listed Medical codes**\n')
            assert result[1].startswith('data: | Code | Type | Description |\n')

    with patch('extraction_tool.core.vector_db_handler.VectorDBHandler.fetch_medical_codes_from_codemaster') as mock_fetch_codes:
        mock_fetch_codes.side_effect = Exception("Test exception")
        with pytest.raises(Exception):
            conversation.make_conversation_poc(payload)

    with patch('extraction_tool.core.vector_db_handler.VectorDBHandler.fetch_medical_codes_from_codemaster') as mock_fetch_codes:
        mock_fetch_codes.return_value = ([], [])
        with patch('extraction_tool.core.llm_handler.LLMHandler.stream_completion') as mock_stream_completion:
            mock_stream_completion.side_effect = Exception("Test exception")
            with pytest.raises(Exception):
                conversation.make_conversation_poc(payload)

def test_get_ask_lyric_common_code_columns():
    conversation = AskLyricConversation()
    assert conversation.get_ask_lyric_common_code_columns() == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_common_code_columns = None
        conversation.get_ask_lyric_common_code_columns()

def test_get_ask_lyric_table_schemas():
    conversation = AskLyricConversation()
    assert len(conversation.get_ask_lyric_table_schemas()) == 3
    assert all(isinstance(schema, TableSchema) for schema in conversation.get_ask_lyric_table_schemas())
    assert all(schema.table_name in ["medical_codes_icd", "medical_codes_cpt", "medical_codes_hcpcs"] for schema in conversation.get_ask_lyric_table_schemas())
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas()

def test_get_db_handlers():
    conversation = AskLyricConversation()
    conversation.get_db_handlers()
    assert conversation.icd_handler is not None
    assert conversation.cpt_handler is not None
    assert conversation.hcpcs_handler is not None

    with pytest.raises(AttributeError):
        conversation.icd_handler = None
        conversation.get_db_handlers()

    with pytest.raises(ValueError):
        conversation.icd_handler = VectorDBHandler()
        conversation.get_db_handlers()

    with pytest.raises(TypeError):
        conversation.get_db_handlers()

    with pytest.raises(TypeError):
        conversation.get_db_handlers(schemas=[TableSchema()])

    with pytest.raises(TypeError):
        conversation.get_db_handlers(schemas=[TableSchema(), TableSchema()])

    with pytest.raises(TypeError):
        conversation.get_db_handlers(schemas=[TableSchema(), TableSchema(), TableSchema(), TableSchema()])

def test_fetch_codes_from_query_rag():
    conversation = AskLyricConversation()
    conversation.icd_handler = MagicMock()
    conversation.cpt_handler = MagicMock()
    conversation.hcpcs_handler = MagicMock()

    # Test with valid query and top_k
    assert fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=10) == []

    # Test with invalid query
    assert fetch_codes_from_query_rag(conversation, "", top_k=10) == []

    # Test with valid query and max_total_results
    assert fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", max_total_results=5) == []

    # Test with valid query and top_k, max_total_results
    assert fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=10, max_total_results=5) == []

    # Test with invalid code type
    with pytest.raises(ValueError):
        fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=10, max_total_results=5)

    # Test with invalid handler
    conversation.icd_handler = None
    with pytest.raises(AttributeError):
        fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=10, max_total_results=5)

    # Test with invalid query and top_k
    with pytest.raises(ValueError):
        fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=-10, max_total_results=5)

    # Test with invalid query and max_total_results
    with pytest.raises(ValueError):
        fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=10, max_total_results=-5)

    # Test with invalid query and top_k, max_total_results
    with pytest.raises(ValueError):
        fetch_codes_from_query_rag(conversation, "What is the ICD code for diabetes?", top_k=-10, max_total_results=-5)

def test_fetch_ask_lyric_context():
    conversation = AskLyricConversation()
    assert fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, 15) == {
        "mentioned_medical_codes": [],
        "rag_fetched_medical_codes": []
    }

    assert fetch_ask_lyric_context("What is the definition of diabetes?", [{"code": "123"}], 10, 15) == {
        "mentioned_medical_codes": [{"code": "123"}],
        "rag_fetched_medical_codes": []
    }

    assert fetch_ask_lyric_context("What is the definition of diabetes?", None, 0, 15) == {
        "mentioned_medical_codes": [],
        "rag_fetched_medical_codes": []
    }

    with pytest.raises(ValueError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, -10, 15)

    with pytest.raises(ValueError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, -15)

    with pytest.raises(ValueError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, 15)

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, "10", 15)

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, "15")

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, 15.5)

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, "abc")

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, (15,))

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90, "j": 100})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90, "j": 100, "k": 110})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90, "j": 100, "k": 110, "l": 120})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90, "j": 100, "k": 110, "l": 120, "m": 130})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, {"a": 15, "b": 20, "c": 30, "d": 40, "e": 50, "f": 60, "g": 70, "h": 80, "i": 90, "j": 100, "k": 110, "l": 120, "m": 130, "n": 140})

    with pytest.raises(TypeError):
        fetch_ask_lyric_context("What is the definition of diabetes?", None, 10, [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])

    with pytest.raises(TypeError):
        fetch_ask_lyric
