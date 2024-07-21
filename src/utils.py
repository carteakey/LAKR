import ast
import re

# Define a function to safely parse JSON-like strings
def safe_parse(data):
    try:
        return ast.literal_eval(data)
    except (ValueError, SyntaxError):
        return data

# Function to safely convert values
def safe_convert(value, type_func):
    if value is None or value == 'None' or (isinstance(value, str) and value.strip() == ''):
        return None
    try:
        return type_func(value)
    except (ValueError, TypeError):
        return None

# Function to clean strings
def clean_string(s):
    if not isinstance(s, str):
        return s
    # Remove or replace problematic characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    # Replace other non-UTF8 characters
    return s.encode('utf-8', errors='ignore').decode('utf-8')
