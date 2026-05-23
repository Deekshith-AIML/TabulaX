def gregorian_to_jalali():
    return '''
from datetime import datetime

def transform(x):
    try:
        gy, gm, gd = map(int, x.strip().split('-'))
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
        return f"{jy}/{jm:02}/{jd:02}"
    except:
        return "error"
'''
def gregorian_to_hijri():
    return '''
from hijri_converter import convert

def transform(x):
    gy, gm, gd = map(int, x.split('-'))
    hijri = convert.Gregorian(gy, gm, gd).to_hijri()
    return (hijri.year, hijri.month, hijri.day)
'''

def date_format_change():
    return '''
from datetime import datetime

def transform(x):
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(x.strip(), fmt)
            return dt.strftime("%d-%m-%Y")
        except:
            continue
    return "error"
'''

def date_to_weekday():
    return '''
from datetime import datetime

def transform(x):
    try:
        return datetime.strptime(x.strip(), "%Y-%m-%d").strftime("%A")
    except:
        return "error"
'''

def dob_to_age():
    return '''
from datetime import datetime, date

def transform(x):
    try:
        birth_date = datetime.strptime(x.strip(), "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return str(age)
    except:
        return ""
'''

def extract_year_from_date():
    return '''
from datetime import datetime

def transform(x):
    try:
        return str(datetime.strptime(x.strip(), "%Y-%m-%d").year)
    except:
        return "error"
'''

def decimal_to_binary():
    return '''
def transform(x):
    try:
        return bin(int(x.strip()))[2:]
    except:
        return ""
'''

def binary_to_decimal():
    return '''
def transform(x):
    try:
        return str(int(x.strip(), 2))
    except:
        return "error"
'''

def hex_to_decimal():
    return '''
def transform(x):
    try:
        return str(int(x.strip(), 16))
    except:
        return "error"
'''

def decimal_to_roman():
    return '''
def transform(x):
    try:
        val = int(x.strip())
        val_map = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
        ]
        result = ""
        for (num, roman) in val_map:
            while val >= num:
                result += roman
                val -= num
        return result
    except:
        return "error"
'''

def roman_to_decimal():
    return '''
def transform(x):
    try:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        prev = 0
        total = 0
        for c in reversed(x.strip().upper()):
            value = roman_map.get(c, 0)
            if value < prev:
                total -= value
            else:
                total += value
                prev = value
        return str(total)
    except:
        return "error"
'''

def char_to_ascii():
    return '''
def transform(x):
    return str(ord(x.strip())) if x else ""
'''

def ascii_to_char():
    return '''
def transform(x):
    try:
        return chr(int(x.strip()))
    except:
        return "error"
'''

def base64_encode():
    return '''
import base64

def transform(x):
    try:
        return base64.b64encode(x.strip().encode()).decode()
    except:
        return "error"
'''

def base64_decode():
    return '''
import base64

def transform(x):
    try:
        return base64.b64decode(x.strip().encode()).decode()
    except:
        return "error"
'''

def unicode_to_char():
    return '''
def transform(x):
    try:
        return chr(int(x.strip().replace("U+", ""), 16))
    except:
        return ""
'''

def convert_12h_to_24h():
    return '''
from datetime import datetime

def transform(x):
    try:
        return datetime.strptime(x.strip(), "%I:%M %p").strftime("%H:%M")
    except:
        return "error"
'''

def minutes_to_hhmm():
    return '''
def transform(x):
    try:
        minutes = int(x.strip())
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
    except:
        return "error"
'''

def time_to_seconds():
    return '''
from datetime import datetime

def transform(x):
    try:
        dt = datetime.strptime(x.strip(), "%H:%M:%S")
        return str(dt.hour * 3600 + dt.minute * 60 + dt.second)
    except:
        return "error"
'''

def extract_domain_from_email():
    return '''
def transform(x):
    try:
        return x.strip().split('@')[1]
    except:
        return "error"
'''

def extract_path_from_url():
    return '''
from urllib.parse import urlparse

def transform(x):
    try:
        return urlparse(x.strip()).path
    except:
        return "error"
'''

def validate_ip_format():
    return '''
import re

def transform(x):
    pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.)){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    return "valid" if re.match(pattern, x.strip()) else "invalid"
'''
