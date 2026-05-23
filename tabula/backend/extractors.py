import re
from datetime import datetime
#extracting conversion type
def extract_conversion_type(response: str) -> str:
    known_types = {
        "gregorian_to_jalali",
        "gregorian_to_hijri",
        "date_format_change",
        "date_to_weekday",
        "dob_to_age",
        "extract_year_from_date",
        "decimal_to_binary",
        "binary_to_decimal",
        "hex_to_decimal",
        "decimal_to_roman",
        "roman_to_decimal",
        "char_to_ascii",
        "ascii_to_char",
        "base64_encode",
        "base64_decode",
        "unicode_to_char",
        "convert_12h_to_24h",
        "minutes_to_hhmm",
        "time_to_seconds",
        "extract_domain_from_email",
        "extract_path_from_url",
        "validate_ip_format",
    }

    response = response.strip().lower()
    for t in known_types:
        if t in response:
            return t
    return ""

#extracting actual reponse from the model
def extract_response_only(prompt: str, full_response: str) -> str:
    if full_response.startswith(prompt):
        return full_response[len(prompt):].strip()
    return full_response.strip()  # Fallback if prompt isn't echoed


# Extract transformation type
def extract_transformation_type(response):
    from langchain.schema.messages import AIMessage
    # Extract content if it's an AIMessage
    response_text = response.content if isinstance(response, AIMessage) else str(response)
    
    categories = ["String-based", "Numerical", "Algorithmic", "General"]
    for category in categories:
        if category.lower() in response_text.lower():
            return category
    return "Unknown"
    
# Code extraction from response
def extract_code_from_response(response):
    
    # Match triple-backtick code blocks
    match = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if match:
        return match[-1].strip()
    return ""  # Return empty string if no code found


# Helper function to check if a string is a date
def is_date(string):
    formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            datetime.strptime(string, fmt)
            return True
        except ValueError:
            pass
    return False


 