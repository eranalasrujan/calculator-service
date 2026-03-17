from app.models import AskLyricRequest
from ask_lyric import AskLyricConversation
from extraction_tool.config.prompts import get_prompts
from extraction_tool.core.llm_handler import LLMHandler
from extraction_tool.core.vector_db_handler import VectorDBHandler
from extraction_tool.models import TableSchema
from src.ask_lyric import AskLyricConversation
from typing import Dict
from typing import List
from typing import List, Dict, Any
from typing import List, Dict, Any, Optional
import json
import logging
import os
import pytest

def test_AskLyricConversation__init__valid_input():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    assert instance.llm_handler.api_key == "valid_key"
    assert instance.llm_handler.provider == "openai"
    assert instance.llm_handler.model_name == "gpt-4"
    assert instance.icd_handler is None
    assert instance.cpt_handler is None
    assert instance.hcpcs_handler is None

def test_AskLyricConversation__init__invalid_openai_key():
    os.environ["OPENAI_KEY"] = ""
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__invalid_openai_model():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = ""
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__missing_openai_key():
    del os.environ["OPENAI_KEY"]
    with pytest.raises(KeyError):
        AskLyricConversation()

def test_AskLyricConversation__init__missing_openai_model():
    os.environ["OPENAI_KEY"] = "valid_key"
    del os.environ["OPENAI_MODEL"]
    with pytest.raises(KeyError):
        AskLyricConversation()

def test_AskLyricConversation__init__empty_openai_key():
    os.environ["OPENAI_KEY"] = ""
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__empty_openai_model():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = ""
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__large_openai_key():
    os.environ["OPENAI_KEY"] = "a" * 1000
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__large_openai_model():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "a" * 1000
    with pytest.raises(ValueError):
        AskLyricConversation()

def test_AskLyricConversation__init__invalid_llm_handler():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    instance.llm_handler = None
    with pytest.raises(AttributeError):
        instance.get_db_handlers()

def test_AskLyricConversation__init__invalid_icd_handler():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    instance.icd_handler = None
    instance.get_db_handlers()

def test_AskLyricConversation__init__invalid_cpt_handler():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    instance.cpt_handler = None
    instance.get_db_handlers()

def test_AskLyricConversation__init__invalid_hcpcs_handler():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    instance.hcpcs_handler = None
    instance.get_db_handlers()

def test_AskLyricConversation__init__get_db_handlers():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    instance.get_db_handlers()
    assert instance.icd_handler is not None
    assert instance.cpt_handler is not None
    assert instance.hcpcs_handler is not None

def test_AskLyricConversation__init__load_formatting_prompt():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    instance = AskLyricConversation()
    assert instance.system_instruction == PROMPTS_CONFIG.get("ask_lyric_prompt", {}).get("system_instructions", "")

def test_AskLyricConversation__init__load_formatting_prompt_invalid_config():
    os.environ["OPENAI_KEY"] = "valid_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    PROMPTS_CONFIG = {}
    instance = AskLyricConversation()
    assert instance.system_instruction == ""

@pytest.mark.asyncio
async def test_get_context_valid_input():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = [{"code": "123", "description": "Medical code 123"}]
    system_instruction, prompt = await conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

@pytest.mark.asyncio
async def test_get_context_invalid_input():
    conversation = AskLyricConversation()
    query = None
    codes = [{"code": "123", "description": "Medical code 123"}]
    with pytest.raises(Exception):
        await conversation.get_context(query, codes)

@pytest.mark.asyncio
async def test_get_context_empty_input():
    conversation = AskLyricConversation()
    query = ""
    codes = [{"code": "123", "description": "Medical code 123"}]
    system_instruction, prompt = await conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "" in prompt

@pytest.mark.asyncio
async def test_get_context_large_input():
    conversation = AskLyricConversation()
    query = "a" * 1000
    codes = [{"code": "123", "description": "Medical code 123"}]
    system_instruction, prompt = await conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "a" * 1000 in prompt

@pytest.mark.asyncio
async def test_get_context_none_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = None
    system_instruction, prompt = await conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

@pytest.mark.asyncio
async def test_get_context_empty_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = []
    system_instruction, prompt = await conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

@pytest.mark.asyncio
async def test_get_context_invalid_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = [{"code": "abc", "description": "Invalid medical code"}]
    with pytest.raises(Exception):
        await conversation.get_context(query, codes)

@pytest.mark.asyncio
async def test_get_context_invalid_query_type():
    conversation = AskLyricConversation()
    query = 123
    codes = [{"code": "123", "description": "Medical code 123"}]
    with pytest.raises(Exception):
        await conversation.get_context(query, codes)

@pytest.mark.asyncio
async def test_get_context_invalid_codes_type():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = "invalid"
    with pytest.raises(Exception):
        await conversation.get_context(query, codes)

@pytest.mark.asyncio
async def test_make_conversation_poc_valid_input():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="What is the meaning of life?")
    result = await conversation.make_conversation_poc(payload)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_make_conversation_poc_empty_question():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="")
    with pytest.raises(Exception):
        await conversation.make_conversation_poc(payload)

@pytest.mark.asyncio
async def test_make_conversation_poc_none_question():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question=None)
    with pytest.raises(Exception):
        await conversation.make_conversation_poc(payload)

@pytest.mark.asyncio
async def test_make_conversation_poc_invalid_input():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question=123)
    with pytest.raises(Exception):
        await conversation.make_conversation_poc(payload)

@pytest.mark.asyncio
async def test_make_conversation_poc_large_question():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="a" * 1000)
    result = await conversation.make_conversation_poc(payload)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_make_conversation_poc_missing_codes():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="What is the meaning of life?")
    codes, missing_codes = ["code1", "code2"], ["code3", "code4"]
    conversation.icd_handler.fetch_medical_codes_from_codemaster = lambda x: codes, missing_codes
    result = await conversation.make_conversation_poc(payload)
    assert len(result) > 0
    assert "Note:" in str(result)

@pytest.mark.asyncio
async def test_make_conversation_poc_llm_streaming_error():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="What is the meaning of life?")
    conversation.llm_handler.stream_completion = lambda x, y, z: [None]
    with pytest.raises(Exception):
        await conversation.make_conversation_poc(payload)

@pytest.mark.asyncio
async def test_make_conversation_poc_context_error():
    conversation = AskLyricConversation(VectorDBHandler(), LLMHandler())
    payload = AskLyricRequest(question="What is the meaning of life?")
    conversation.get_context = lambda x, y: None
    with pytest.raises(Exception):
        await conversation.make_conversation_poc(payload)

def test_get_ask_lyric_common_code_columns_valid_input():
    conversation = AskLyricConversation()
    result = conversation.get_ask_lyric_common_code_columns()
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

def test_get_ask_lyric_common_code_columns_empty_input():
    conversation = AskLyricConversation()
    result = conversation.get_ask_lyric_common_code_columns()
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

def test_get_ask_lyric_common_code_columns_none_input():
    conversation = AskLyricConversation()
    result = conversation.get_ask_lyric_common_code_columns(None)
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

def test_get_ask_lyric_common_code_columns_invalid_input():
    conversation = AskLyricConversation()
    with pytest.raises(TypeError):
        conversation.get_ask_lyric_common_code_columns("invalid_input")

def test_get_ask_lyric_common_code_columns_large_input():
    conversation = AskLyricConversation()
    result = conversation.get_ask_lyric_common_code_columns({"id": "SERIAL PRIMARY KEY" * 1000,
                                                            "code": "TEXT NOT NULL" * 1000,
                                                            "description": "TEXT NOT NULL" * 1000})
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

def test_get_ask_lyric_common_code_columns_negative_input():
    conversation = AskLyricConversation()
    result = conversation.get_ask_lyric_common_code_columns({"id": -1,
                                                            "code": -1,
                                                            "description": -1})
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

def test_get_ask_lyric_table_schemas_valid_input():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert all(isinstance(schema, TableSchema) for schema in schemas)

def test_get_ask_lyric_table_schemas_empty_input():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert all(isinstance(schema, TableSchema) for schema in schemas)

def test_get_ask_lyric_table_schemas_invalid_input():
    conversation = AskLyricConversation()
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas("invalid_input")

def test_get_ask_lyric_table_schemas_none_input():
    conversation = AskLyricConversation()
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas(None)

def test_get_ask_lyric_table_schemas_large_input():
    conversation = AskLyricConversation()
    with pytest.raises(MemoryError):
        conversation.get_ask_lyric_table_schemas(["large_input"] * 1000000)

def test_get_ask_lyric_table_schemas_invalid_table_name():
    conversation = AskLyricConversation()
    with pytest.raises(ValueError):
        conversation.get_ask_lyric_table_schemas(["invalid_table_name"])

def test_get_ask_lyric_table_schemas_missing_table():
    conversation = AskLyricConversation()
    with pytest.raises(ValueError):
        conversation.get_ask_lyric_table_schemas(["missing_table"])

def test_get_ask_lyric_table_schemas_invalid_columns():
    conversation = AskLyricConversation()
    with pytest.raises(ValueError):
        conversation.get_ask_lyric_table_schemas(["invalid_columns"])

def test_get_db_handlers_valid_input():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert isinstance(schemas[0], TableSchema)
    assert isinstance(schemas[1], TableSchema)
    assert isinstance(schemas[2], TableSchema)

def test_get_db_handlers_invalid_input():
    conversation = AskLyricConversation()
    with pytest.raises(IndexError):
        conversation.get_ask_lyric_table_schemas()[3]

def test_get_db_handlers_empty_input():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert all(schema is not None for schema in schemas)

def test_get_db_handlers_none_input():
    conversation = AskLyricConversation()
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas(None)

def test_get_db_handlers_logger_info():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert logging.getLogger().level == logging.INFO
    assert "Initialized 3 VectorDBHandlers for ICD, CPT, and HCPCS tables" in logging.getLogger().handlers[0].emit.call_args[0][0]

def test_get_db_handlers_vector_db_handler_creation():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert isinstance(conversation.icd_handler, VectorDBHandler)
    assert isinstance(conversation.cpt_handler, VectorDBHandler)
    assert isinstance(conversation.hcpcs_handler, VectorDBHandler)

def test_get_db_handlers_vector_db_handler_cache():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    conversation.icd_handler = None
    conversation.get_db_handlers()
    assert conversation.icd_handler is not None

def test_get_db_handlers_vector_db_handler_cache_multiple_calls():
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    conversation.icd_handler = None
    conversation.get_db_handlers()
    conversation.get_db_handlers()
    assert conversation.icd_handler is not None

def test_fetch_codes_from_query_rag_valid_input():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert isinstance(result, list)
    assert len(result) <= max_total_results

def test_fetch_codes_from_query_rag_empty_question():
    conversation = AskLyricConversation()
    question = ""
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert result == []

def test_fetch_codes_from_query_rag_too_short_question():
    conversation = AskLyricConversation()
    question = "a"
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert result == []

def test_fetch_codes_from_query_rag_invalid_top_k():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = -1
    max_total_results = 15
    with pytest.raises(ValueError):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)

def test_fetch_codes_from_query_rag_invalid_max_total_results():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = -1
    with pytest.raises(ValueError):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)

def test_fetch_codes_from_query_rag_exception_handling():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    conversation.icd_handler.search_similar_codes = lambda *args, **kwargs: None
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert result == []

def test_fetch_codes_from_query_rag_logging():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    with pytest.raises(ValueError):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert "Error in RAG code search" in conversation.logger.error.call_args[0][0]

def test_fetch_codes_from_query_rag_sorting():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    results = [
        {'similarity_score': 0.9, 'code': 'code1'},
        {'similarity_score': 0.8, 'code': 'code2'},
        {'similarity_score': 0.7, 'code': 'code3'}
    ]
    conversation.icd_handler.search_similar_codes = lambda *args, **kwargs: {'matches': results}
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert result[0]['similarity_score'] == 0.9
    assert result[1]['similarity_score'] == 0.8
    assert result[2]['similarity_score'] == 0.7

def test_fetch_codes_from_query_rag_max_total_results():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 5
    results = [
        {'similarity_score': 0.9, 'code': 'code1'},
        {'similarity_score': 0.8, 'code': 'code2'},
        {'similarity_score': 0.7, 'code': 'code3'},
        {'similarity_score': 0.6, 'code': 'code4'},
        {'similarity_score': 0.5, 'code': 'code5'},
        {'similarity_score': 0.4, 'code': 'code6'}
    ]
    conversation.icd_handler.search_similar_codes = lambda *args, **kwargs: {'matches': results}
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert len(result) == max_total_results

def test_fetch_ask_lyric_context_valid_input():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result

def test_fetch_ask_lyric_context_invalid_query():
    conversation = AskLyricConversation()
    query = ""
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_empty_medical_codes():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = []
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result

def test_fetch_ask_lyric_context_large_top_k():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 1000
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result

def test_fetch_ask_lyric_context_negative_max_total_results():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = -1
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_none_query():
    conversation = AskLyricConversation()
    query = None
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_none_mentioned_medical_codes():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = None
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result

def test_fetch_ask_lyric_context_none_top_k():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = None
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_none_max_total_results():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = None
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
