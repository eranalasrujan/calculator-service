def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

def pow(a,b):
    return a**b
#new commenttttttt

def mod_div(a,b):
    if b == 0:
        raise ValueError("Division by zero")
    return a // b

