from typing import Type


x = "Hello"

if not type(x) is int: 
    raise  TypeError("Only integer are allowed")