from src.demo_test import calculate_payment
from src.demo_test import validate_signup
from src.demo_test import validate_user_api_payload
import pytest

def test_validate_signup():
    assert validate_signup("test", "test@example.com", 25, "password123") == True
    assert validate_signup("", "test@example.com", 25, "password123") == False
    assert validate_signup("test", "test", 25, "password123") == False
    assert validate_signup("test", "test@example.com", 17, "password123") == False
    assert validate_signup("test", "test@example.com", 25, "password") == False
    with pytest.raises(TypeError):
        validate_signup(123, "test@example.com", 25, "password123")
    with pytest.raises(TypeError):
        validate_signup("test", 123, 25, "password123")
    with pytest.raises(TypeError):
        validate_signup("test", "test@example.com", "25", "password123")
    with pytest.raises(TypeError):
        validate_signup("test", "test@example.com", 25, 123)

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
    assert validate_user_api_payload({"name": "Bob", "email": "bob@example.com", "age": 17}) == False
    assert validate_user_api_payload({"name": "Alice", "email": "alice@example.com"}) == False
    assert validate_user_api_payload({"name": "Charlie", "email": "charlie", "age": 20}) == False
    assert validate_user_api_payload({"name": "David", "email": "david@example.com", "age": 25}) == True
    with pytest.raises(KeyError):
        validate_user_api_payload({"name": "Eve", "email": "eve@example.com"})
    with pytest.raises(KeyError):
        validate_user_api_payload({"age": 25})
    with pytest.raises(KeyError):
        validate_user_api_payload({"name": "Frank"})
