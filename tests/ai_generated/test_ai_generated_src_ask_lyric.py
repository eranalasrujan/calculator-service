from ask_lyric import AskLyricConversation
from extraction_tool.config.prompts import get_prompts
from extraction_tool.core.llm_handler import LLMHandler
from extraction_tool.core.vector_db_handler import VectorDBHandler
from extraction_tool.models import TableSchema
from src.ask_lyric import AskLyricConversation
from typing import Dict, Any
from typing import List, Dict, Any
from typing import List, Dict, Any, Optional
import json
import logging
import os
import pytest

def test_AskLyricConversation_init():
    # Normal case
    openai_key = "test_openai_key"
    openai_model = "gpt-4"
    with pytest.raises(ValueError):
        AskLyricConversation()

    # Edge case: missing OPENAI_KEY environment variable
    os.environ.pop("OPENAI_KEY", None)
    with pytest.raises(EnvironmentError):
        AskLyricConversation()

    # Edge case: invalid OPENAI_MODEL environment variable
    os.environ["OPENAI_KEY"] = "test_openai_key"
    os.environ["OPENAI_MODEL"] = "invalid_model"
    with pytest.raises(ValueError):
        AskLyricConversation()

    # Edge case: missing PROMPTS_CONFIG
    os.environ["OPENAI_KEY"] = "test_openai_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    del get_prompts
    with pytest.raises(AttributeError):
        AskLyricConversation()

    # Edge case: invalid PROMPTS_CONFIG
    os.environ["OPENAI_KEY"] = "test_openai_key"
    os.environ["OPENAI_MODEL"] = "gpt-4"
    get_prompts = lambda: {"ask_lyric_prompt": {"system_instructions": "invalid"}}
    with pytest.raises(KeyError):
        AskLyricConversation()

    # Clean up environment variables
    os.environ.pop("OPENAI_KEY", None)
    os.environ.pop("OPENAI_MODEL", None)

def test_get_context_normal_case():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = [{"code": "123", "description": "Test code"}]
    system_instruction, prompt = conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

def test_get_context_empty_query():
    conversation = AskLyricConversation()
    query = ""
    codes = [{"code": "123", "description": "Test code"}]
    with pytest.raises(Exception):
        conversation.get_context(query, codes)

def test_get_context_none_query():
    conversation = AskLyricConversation()
    query = None
    codes = [{"code": "123", "description": "Test code"}]
    with pytest.raises(Exception):
        conversation.get_context(query, codes)

def test_get_context_empty_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = []
    system_instruction, prompt = conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

def test_get_context_none_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = None
    system_instruction, prompt = conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

def test_get_context_invalid_query_type():
    conversation = AskLyricConversation()
    query = 123
    codes = [{"code": "123", "description": "Test code"}]
    with pytest.raises(Exception):
        conversation.get_context(query, codes)

def test_get_context_invalid_codes_type():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = "Invalid codes"
    with pytest.raises(Exception):
        conversation.get_context(query, codes)

def test_get_context_large_query():
    conversation = AskLyricConversation()
    query = "a" * 1000
    codes = [{"code": "123", "description": "Test code"}]
    system_instruction, prompt = conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "a" * 1000 in prompt

def test_get_context_large_codes():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = [{"code": "123", "description": "Test code"}] * 100
    system_instruction, prompt = conversation.get_context(query, codes)
    assert isinstance(system_instruction, str)
    assert isinstance(prompt, str)
    assert "What is the meaning of life?" in prompt

def test_get_context_invalid_json():
    conversation = AskLyricConversation()
    query = "What is the meaning of life?"
    codes = [{"code": "123", "description": "Test code"}]
    with pytest.raises(Exception):
        conversation.get_context(query, codes)
        json.dumps({"key": None}, indent=2)

def test_get_ask_lyric_common_code_columns():
    # Normal case
    ask_lyric_conversation = AskLyricConversation()
    result = ask_lyric_conversation.get_ask_lyric_common_code_columns()
    assert isinstance(result, dict)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

    # Edge case: empty result
    ask_lyric_conversation = AskLyricConversation()
    result = ask_lyric_conversation.get_ask_lyric_common_code_columns()
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

    # Edge case: None input
    ask_lyric_conversation = AskLyricConversation()
    result = ask_lyric_conversation.get_ask_lyric_common_code_columns(None)
    assert result == {
        "id": "SERIAL PRIMARY KEY",
        "code": "TEXT NOT NULL",
        "description": "TEXT NOT NULL"
    }

    # Edge case: invalid input type
    ask_lyric_conversation = AskLyricConversation()
    with pytest.raises(TypeError):
        ask_lyric_conversation.get_ask_lyric_common_code_columns("invalid_input")

    # Edge case: missing required key
    ask_lyric_conversation = AskLyricConversation()
    with pytest.raises(KeyError):
        ask_lyric_conversation.get_ask_lyric_common_code_columns({"code": "TEXT NOT NULL"})

    # Edge case: invalid value type
    ask_lyric_conversation = AskLyricConversation()
    with pytest.raises(TypeError):
        ask_lyric_conversation.get_ask_lyric_common_code_columns({"id": 123})

def test_get_ask_lyric_table_schemas():
    # Normal case
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert isinstance(schemas, list)
    assert all(isinstance(schema, TableSchema) for schema in schemas)

    # Edge case: empty list of tables
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas(tables=[])
    assert len(schemas) == 0

    # Edge case: None input
    conversation = AskLyricConversation()
    with pytest.raises(AttributeError):
        conversation.get_ask_lyric_table_schemas(tables=None)

    # Edge case: invalid input type
    conversation = AskLyricConversation()
    with pytest.raises(TypeError):
        conversation.get_ask_lyric_table_schemas(tables="invalid")

    # Edge case: large input
    conversation = AskLyricConversation()
    tables = ["medical_codes_icd"] * 1000
    schemas = conversation.get_ask_lyric_table_schemas(tables=tables)
    assert len(schemas) == 1

    # Edge case: small input
    conversation = AskLyricConversation()
    tables = ["medical_codes_icd"]
    schemas = conversation.get_ask_lyric_table_schemas(tables=tables)
    assert len(schemas) == 1

    # Edge case: empty string input
    conversation = AskLyricConversation()
    tables = ""
    schemas = conversation.get_ask_lyric_table_schemas(tables=tables)
    assert len(schemas) == 0

    # Edge case: invalid table name
    conversation = AskLyricConversation()
    tables = ["invalid_table"]
    schemas = conversation.get_ask_lyric_table_schemas(tables=tables)
    assert len(schemas) == 0

def test_get_db_handlers():
    # Normal case
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert len(schemas) == 3
    assert isinstance(schemas[0], TableSchema)
    assert isinstance(schemas[1], TableSchema)
    assert isinstance(schemas[2], TableSchema)

    # Edge case: empty schemas
    conversation = AskLyricConversation()
    schemas = []
    with pytest.raises(IndexError):
        conversation.icd_handler = VectorDBHandler.get_or_create_handler(schemas[0])

    # Edge case: None schemas
    conversation = AskLyricConversation()
    schemas = None
    with pytest.raises(TypeError):
        conversation.icd_handler = VectorDBHandler.get_or_create_handler(schemas)

    # Edge case: invalid schemas
    conversation = AskLyricConversation()
    schemas = ["invalid"]
    with pytest.raises(TypeError):
        conversation.icd_handler = VectorDBHandler.get_or_create_handler(schemas)

    # Normal case: logging
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger().handlers

    # Edge case: logging level
    logging.getLogger().setLevel(logging.ERROR)
    conversation = AskLyricConversation()
    schemas = conversation.get_ask_lyric_table_schemas()
    assert logging.getLogger().level == logging.ERROR

def test_fetch_codes_from_query_rag_normal_case():
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
    with pytest.raises(Exception):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)

def test_fetch_codes_from_query_rag_invalid_max_total_results():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = -1
    with pytest.raises(Exception):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)

def test_fetch_codes_from_query_rag_none_question():
    conversation = AskLyricConversation()
    question = None
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert result == []

def test_fetch_codes_from_query_rag_large_question():
    conversation = AskLyricConversation()
    question = "a" * 1000
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_codes_from_query_rag(question, top_k, max_total_results)
    assert isinstance(result, list)
    assert len(result) <= max_total_results

def test_fetch_codes_from_query_rag_invalid_code_type():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results, code_type="invalid")

def test_fetch_codes_from_query_rag_invalid_handler():
    conversation = AskLyricConversation()
    question = "What is the meaning of life?"
    top_k = 10
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_codes_from_query_rag(question, top_k, max_total_results, handler="invalid")

def test_fetch_ask_lyric_context_normal_case():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result
    assert len(result["mentioned_medical_codes"]) == 1
    assert len(result["rag_fetched_medical_codes"]) == 10

def test_fetch_ask_lyric_context_empty_query():
    conversation = AskLyricConversation()
    query = ""
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 15
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

def test_fetch_ask_lyric_context_empty_mentioned_medical_codes():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = []
    top_k = 10
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result
    assert len(result["mentioned_medical_codes"]) == 0
    assert len(result["rag_fetched_medical_codes"]) == 10

def test_fetch_ask_lyric_context_large_top_k():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 100
    max_total_results = 15
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result
    assert len(result["mentioned_medical_codes"]) == 1
    assert len(result["rag_fetched_medical_codes"]) == 10

def test_fetch_ask_lyric_context_negative_top_k():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = -10
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_large_max_total_results():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = 100
    result = conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
    assert isinstance(result, dict)
    assert "mentioned_medical_codes" in result
    assert "rag_fetched_medical_codes" in result
    assert len(result["mentioned_medical_codes"]) == 1
    assert len(result["rag_fetched_medical_codes"]) == 10

def test_fetch_ask_lyric_context_negative_max_total_results():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = -15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_invalid_top_k_type():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = "ten"
    max_total_results = 15
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)

def test_fetch_ask_lyric_context_invalid_max_total_results_type():
    conversation = AskLyricConversation()
    query = "What is the definition of a medical code?"
    mentioned_medical_codes = [{"code": "12345", "description": "Test code"}]
    top_k = 10
    max_total_results = "fifteen"
    with pytest.raises(Exception):
        conversation.fetch_ask_lyric_context(query, mentioned_medical_codes, top_k, max_total_results)
