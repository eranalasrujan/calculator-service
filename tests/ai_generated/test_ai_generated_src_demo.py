from src.demo import calculate_payment
from src.demo import validate_signup
from src.demo import validate_user_api_payload
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
