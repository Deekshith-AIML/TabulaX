"""Algorithmic transformation templates.

Each template is a self-contained Python ``transform`` function returned as a
source string.  They are **deterministic** and require no LLM.  The sandbox
validator accepts all of them (they use only allowlisted constructs).
"""

from __future__ import annotations

from typing import Callable, Dict

# --------------------------------------------------------------------------- #
# Template source strings
# --------------------------------------------------------------------------- #
_GREGORIAN_TO_JALALI = '''
def transform(x):
    try:
        gy, gm, gd = map(int, str(x).strip().replace('/', '-').split('-'))
        g_days_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        gy2 = gy + 1 if gm > 2 else gy
        total_days = (
            355666 + 365 * gy + ((gy2 + 3) // 4)
            - ((gy2 + 99) // 100) + ((gy2 + 399) // 400)
            + gd + g_days_in_month[gm - 1]
        )
        jy = -1595 + (33 * (total_days // 12053))
        total_days %= 12053
        jy += 4 * (total_days // 1461)
        total_days %= 1461
        if total_days > 365:
            jy += (total_days - 1) // 365
            total_days = (total_days - 1) % 365
        if total_days < 186:
            jm = 1 + (total_days // 31)
            jd = 1 + (total_days % 31)
        else:
            total_days -= 186
            jm = 7 + (total_days // 30)
            jd = 1 + (total_days % 30)
        return str(jy) + "/" + str(jm) + "/" + str(jd)
    except Exception:
        return "error"
'''

_GREGORIAN_TO_HIJRI = '''
def transform(x):
    try:
        from hijri_converter import convert
        gy, gm, gd = map(int, str(x).strip().replace('/', '-').split('-'))
        h = convert.Gregorian(gy, gm, gd).to_hijri()
        return str(h.year) + "-" + str(h.month) + "-" + str(h.day)
    except Exception:
        return "error"
'''

_DATE_FORMAT_CHANGE = '''
def transform(x):
    from datetime import datetime
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(str(x).strip(), fmt)
            return dt.strftime("%d-%m-%Y")
        except Exception:
            continue
    return "error"
'''

_DATE_TO_WEEKDAY = '''
def transform(x):
    from datetime import datetime
    try:
        s = str(x).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%A")
            except Exception:
                continue
        return "error"
    except Exception:
        return "error"
'''

_DOB_TO_AGE = '''
def transform(x):
    from datetime import datetime, date
    try:
        s = str(x).strip()
        birth = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                birth = datetime.strptime(s, fmt).date()
                break
            except Exception:
                continue
        if birth is None:
            return "error"
        today = date.today()
        return str(today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day)))
    except Exception:
        return "error"
'''

_EXTRACT_YEAR = '''
def transform(x):
    from datetime import datetime
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return str(datetime.strptime(s, fmt).year)
        except Exception:
            continue
    return "error"
'''

_DECIMAL_TO_BINARY = '''
def transform(x):
    try:
        return bin(int(str(x).strip()))[2:]
    except Exception:
        return "error"
'''

_BINARY_TO_DECIMAL = '''
def transform(x):
    try:
        return str(int(str(x).strip(), 2))
    except Exception:
        return "error"
'''

_HEX_TO_DECIMAL = '''
def transform(x):
    try:
        return str(int(str(x).strip().replace("0x", "").replace("0X", ""), 16))
    except Exception:
        return "error"
'''

_DECIMAL_TO_ROMAN = '''
def transform(x):
    try:
        val = int(str(x).strip())
        if val <= 0 or val > 3999:
            return "error"
        val_map = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
        ]
        result = ""
        for num, roman in val_map:
            while val >= num:
                result += roman
                val -= num
        return result
    except Exception:
        return "error"
'''

_ROMAN_TO_DECIMAL = '''
def transform(x):
    try:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        prev = 0
        total = 0
        for c in reversed(str(x).strip().upper()):
            value = roman_map.get(c, 0)
            if value == 0:
                return "error"
            if value < prev:
                total -= value
            else:
                total += value
                prev = value
        return str(total)
    except Exception:
        return "error"
'''

_CHAR_TO_ASCII = '''
def transform(x):
    s = str(x).strip()
    return str(ord(s)) if s else "error"
'''

_ASCII_TO_CHAR = '''
def transform(x):
    try:
        return chr(int(str(x).strip()))
    except Exception:
        return "error"
'''

_BASE64_ENCODE = '''
import base64

def transform(x):
    try:
        return base64.b64encode(str(x).strip().encode("utf-8")).decode("ascii")
    except Exception:
        return "error"
'''

_BASE64_DECODE = '''
import base64

def transform(x):
    try:
        return base64.b64decode(str(x).strip().encode("ascii"), validate=True).decode("utf-8")
    except Exception:
        return "error"
'''

_UNICODE_TO_CHAR = '''
def transform(x):
    try:
        s = str(x).strip().replace("U+", "").replace("u+", "")
        return chr(int(s, 16))
    except Exception:
        return "error"
'''

_CONVERT_12H_TO_24H = '''
def transform(x):
    from datetime import datetime
    s = str(x).strip()
    for fmt in ("%I:%M %p", "%I:%M%p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M")
        except Exception:
            continue
    return "error"
'''

_MINUTES_TO_HHMM = '''
def transform(x):
    try:
        minutes = int(str(x).strip())
        sign = "-" if minutes < 0 else ""
        minutes = abs(minutes)
        return sign + str(minutes // 60) + ":" + str(minutes % 60).zfill(2)
    except Exception:
        return "error"
'''

_TIME_TO_SECONDS = '''
def transform(x):
    from datetime import datetime
    s = str(x).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return str(dt.hour * 3600 + dt.minute * 60 + dt.second)
        except Exception:
            continue
    return "error"
'''

_EXTRACT_DOMAIN = '''
def transform(x):
    try:
        s = str(x).strip()
        if "@" not in s:
            return "error"
        return s.split("@")[1]
    except Exception:
        return "error"
'''

_EXTRACT_URL_PATH = '''
def transform(x):
    from urllib.parse import urlparse
    try:
        return urlparse(str(x).strip()).path
    except Exception:
        return "error"
'''

_VALIDATE_IP = '''
import re

def transform(x):
    pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\\.)){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    return "valid" if re.match(pattern, str(x).strip()) else "invalid"
'''

_EXTRACT_DOMAIN_FROM_URL = '''
from urllib.parse import urlparse

def transform(x):
    try:
        s = str(x).strip()
        if "://" not in s:
            s = "http://" + s
        netloc = urlparse(s).netloc
        return netloc.replace("www.", "") if netloc else "error"
    except Exception:
        return "error"
'''

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TEMPLATES: Dict[str, str] = {
    "gregorian_to_jalali": _GREGORIAN_TO_JALALI,
    "gregorian_to_hijri": _GREGORIAN_TO_HIJRI,
    "date_format_change": _DATE_FORMAT_CHANGE,
    "date_to_weekday": _DATE_TO_WEEKDAY,
    "dob_to_age": _DOB_TO_AGE,
    "extract_year_from_date": _EXTRACT_YEAR,
    "decimal_to_binary": _DECIMAL_TO_BINARY,
    "binary_to_decimal": _BINARY_TO_DECIMAL,
    "hex_to_decimal": _HEX_TO_DECIMAL,
    "decimal_to_roman": _DECIMAL_TO_ROMAN,
    "roman_to_decimal": _ROMAN_TO_DECIMAL,
    "char_to_ascii": _CHAR_TO_ASCII,
    "ascii_to_char": _ASCII_TO_CHAR,
    "base64_encode": _BASE64_ENCODE,
    "base64_decode": _BASE64_DECODE,
    "unicode_to_char": _UNICODE_TO_CHAR,
    "convert_12h_to_24h": _CONVERT_12H_TO_24H,
    "minutes_to_hhmm": _MINUTES_TO_HHMM,
    "time_to_seconds": _TIME_TO_SECONDS,
    "extract_domain_from_email": _EXTRACT_DOMAIN,
    "extract_path_from_url": _EXTRACT_URL_PATH,
    "validate_ip_format": _VALIDATE_IP,
    "extract_domain_from_url": _EXTRACT_DOMAIN_FROM_URL,
}

TEMPLATE_DESCRIPTIONS: Dict[str, str] = {
    "gregorian_to_jalali": "Convert Gregorian dates (YYYY-MM-DD) to Jalali calendar dates.",
    "gregorian_to_hijri": "Convert Gregorian dates to Islamic Hijri calendar dates.",
    "date_format_change": "Reformat a date string without changing the calendar system.",
    "date_to_weekday": "Extract the weekday name from a date.",
    "dob_to_age": "Compute an age in years from a date of birth.",
    "extract_year_from_date": "Extract the year component from a date.",
    "decimal_to_binary": "Convert a decimal integer to its binary representation.",
    "binary_to_decimal": "Convert a binary string to a decimal integer.",
    "hex_to_decimal": "Convert a hexadecimal value to a decimal integer.",
    "decimal_to_roman": "Convert a decimal integer (1-3999) to a Roman numeral.",
    "roman_to_decimal": "Convert a Roman numeral to a decimal integer.",
    "char_to_ascii": "Return the ASCII/Unicode code point of a character.",
    "ascii_to_char": "Convert an integer code point to its character.",
    "base64_encode": "Base64-encode a UTF-8 string.",
    "base64_decode": "Decode a Base64 string back to text.",
    "unicode_to_char": "Convert a Unicode code point (e.g. U+0627) to a character.",
    "convert_12h_to_24h": "Convert a 12-hour clock time to 24-hour format.",
    "minutes_to_hhmm": "Convert a total-minutes integer to HH:MM.",
    "time_to_seconds": "Convert an HH:MM:SS time to total seconds.",
    "extract_domain_from_email": "Extract the domain from an email address.",
    "extract_path_from_url": "Extract the path component from a URL.",
    "validate_ip_format": "Validate whether a string is a well-formed IPv4 address.",
    "extract_domain_from_url": "Extract the hostname (minus www) from a URL.",
}


def get_template(conversion_type: str) -> str:
    """Return template source for ``conversion_type`` or raise ``KeyError``."""
    return TEMPLATES[conversion_type]


def list_template_types() -> list[str]:
    """Return all known conversion-type labels."""
    return sorted(TEMPLATES.keys())
