def validate_signup(username: str, email: str, age: int, password: str) -> bool:

    if not username or len(username) < 3:
        return False

    if "@" not in email or "." not in email:
        return False

    if age < 18:
        return False

    if len(password) < 8:
        return False
    
    
    if not any(c.isdigit() for c in password):
        return False

    return True


def calculate_payment(price: float, user_type: str) -> float:

    if price < 0:
        raise ValueError("price cannot be negative")

    if user_type == "premium":
        return price * 0.8

    if user_type == "vip":
        return price * 0.7

    return price




def validate_user_api_payload(payload: dict) -> bool:

    required_fields = ["name", "email", "age"]

    for field in required_fields:
        if field not in payload:
            return False

    if payload["age"] < 18:
        return False

    if "@" not in payload["email"]:
        return False

    return True

