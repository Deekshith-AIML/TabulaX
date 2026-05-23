#importing libraries
import numpy as np
import pandas as pd
import re
import ast
from scipy.optimize import curve_fit
from extractors import is_date,extract_code_from_response,extract_response_only,extract_transformation_type,extract_conversion_type
from model_interface import langchain_model,get_cached_model
import algorithmic_templates as at



llama_pipeline = get_cached_model()
 
# Fallback generation logic
def generate_response(prompt, tokens=100,gpt=False):
    try:
        if llama_pipeline and not gpt:
            response = llama_pipeline(
                prompt,
                max_new_tokens=tokens,
                num_return_sequences=1,
                truncation=True,
                eos_token_id=llama_pipeline.tokenizer.eos_token_id,
            )
            generated_text = response[0]['generated_text'].strip()
            
            return generated_text
        else:
            response = langchain_model.invoke(prompt)
            return response.content  # Extract content from AIMessage
    except Exception as e:
        print("Fallback GPT:\n", e)
        response = langchain_model.invoke(prompt)
        return response.content

#serializeing examples      
def serialize_examples(examples):
    return ", ".join([f'(\"{src}\" -> \"{tgt}\")' for src, tgt in examples])

#classification
def classify_transformation(examples_serialized):
    print(examples_serialized)
    def convert_value(s):
        """Convert string to appropriate type (int, float, or keep as string)"""
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s

    def convert_pairs(input_str):
        pattern = r'"([^"]+)"\s*->\s*"([^"]+)"'
        matches = re.findall(pattern, input_str)
        return [(convert_value(src), convert_value(tgt)) for src, tgt in matches]
    
    examples=convert_pairs(examples_serialized)
    print(examples)
    # Attempt numerical classification
    numeric_examples = []
    for src, tgt in examples:
        try:
            src_val = float(src)
            tgt_val = float(tgt)
            numeric_examples.append((src_val, tgt_val))
        except ValueError:
            continue

    if len(numeric_examples) >= 2:
        x_data = np.array([x for x, _ in numeric_examples])
        y_data = np.array([y for _, y in numeric_examples])
        slope, intercept = np.polyfit(x_data, y_data, 1)
        residuals = y_data - (slope * x_data + intercept)
        mse = np.mean(residuals ** 2)
        if mse < 1e-2:
            return "Numerical"

    # Check if all examples are date transformations
    if all(is_date(src) and is_date(tgt) for src, tgt in examples):
        return "Algorithmic"

    # Fall back to model classification
    examples_serialized = serialize_examples(examples)
    prompt = f"""
    Analyze the relationship between source and target data pairs and classify the transformation type into one of these categories:
    1. String-based: Transformations involving text operations like case changes, formatting, substring extraction, etc.
    2. Numerical: Mathematical operations, scaling, rounding, statistical transforms, numeric conversions, etc.
    3. Algorithmic: Complex logical operations, conditional mappings, encoding/decoding, date conversions, calendar system transformations, lookup-based transformations with rules, etc.
    4. General: Simple mappings, lookups, or other transformations that don't clearly fit the above categories.

    Source → Target pairs:
    {examples_serialized}

    Respond ONLY with the class name (String-based, Numerical, Algorithmic, or General).
    """
    response_text = generate_response(prompt, tokens=100)
    classification = extract_transformation_type(response_text)
    return classification


# ===== TRANSFORMATION GENERATORS =====
def generate_string_function(examples):
    # Format examples for the prompt
    example_lines = [f"Input: {src} -> Output: {tgt}" for src, tgt in examples[:2]]
    example_text = "\n".join(example_lines)

    # Modified prompt to explicitly request code in triple backticks
    prompt = f"""
You are a Python string manipulation expert.

Write a function named `transform` that performs the string transformation based on the following examples:
{example_text}

Return only the function code, enclosed in triple backticks, like this:

def transform(x):
    # your code here


Use only standard Python string methods like split(), lower(), upper(), strip(), join(), etc.
Write exactly one function. Do not include any other text, explanations, or markdown, or test cases.
"""

    # Generate response using LLaMA pipeline
    full_response = generate_response(prompt, tokens=400)
    response=extract_response_only(prompt,full_response)
    # Extract the code block from the response
    code = extract_code_from_response(response)
    return code

def fit_numeric_function(pairs):
    x_vals = []
    y_vals = []
    for src, tgt in pairs:
        try:
            source, target = float(src), float(tgt)
            x_vals.append(source)
            y_vals.append(target)
        except ValueError:
            raise ValueError("All source and target values must be numeric for numerical transformation")

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    num_points = len(x_vals)
    if num_points < 2:
        raise ValueError("At least 2 data points are required for numerical transformation")

    def linear_model(x, a, b):
        return a * x + b

    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    def rational_model(x, a, b, c):
        return (a * x + b) / (x + c)

    def polynomial_model(x, a, b, c):
        return a * x**2 + b * x + c

    models = {}
    mse_values = {}

    try:
        params_linear, _ = curve_fit(linear_model, x_vals, y_vals, method='lm')
        a, b = params_linear
        y_pred_linear = linear_model(x_vals, a, b)
        mse_linear = np.mean((y_vals - y_pred_linear) ** 2)
        models['linear'] = (lambda x: a * x + b, a, b)
        mse_values['linear'] = mse_linear
    except Exception as e:
        print(f"Linear model fitting failed: {e}")
        mse_values['linear'] = float('inf')

    try:
        params_exp, _ = curve_fit(exponential_model, x_vals, y_vals, method='lm')
        a, b = params_exp
        y_pred_exp = exponential_model(x_vals, a, b)
        mse_exp = np.mean((y_vals - y_pred_exp) ** 2)
        models['exponential'] = (lambda x: a * np.exp(b * x), a, b)
        mse_values['exponential'] = mse_exp
    except Exception as e:
        print(f"Exponential model fitting failed: {e}")
        mse_values['exponential'] = float('inf')

    if num_points >= 3:
        try:
            params_rational, _ = curve_fit(rational_model, x_vals, y_vals, method='lm')
            a, b, c = params_rational
            y_pred_rational = rational_model(x_vals, a, b, c)
            mse_rational = np.mean((y_vals - y_pred_rational) ** 2)
            models['rational'] = (lambda x: (a * x + b) / (x + c), a, b, c)
            mse_values['rational'] = mse_rational
        except Exception as e:
            print(f"Rational model fitting failed: {e}")
            mse_values['rational'] = float('inf')
    else:
        print(f"Skipping rational model: insufficient data points (need at least 3, got {num_points})")
        mse_values['rational'] = float('inf')

    if num_points >= 3:
        try:
            params_poly, _ = curve_fit(polynomial_model, x_vals, y_vals, method='lm')
            a, b, c = params_poly
            y_pred_poly = polynomial_model(x_vals, a, b, c)
            mse_poly = np.mean((y_vals - y_pred_poly) ** 2)
            models['polynomial'] = (lambda x: a * x**2 + b * x + c, a, b, c)
            mse_values['polynomial'] = mse_poly
        except Exception as e:
            print(f"Polynomial model fitting failed: {e}")
            mse_values['polynomial'] = float('inf')
    else:
        print(f"Skipping polynomial model: insufficient data points (need at least 3, got {num_points})")
        mse_values['polynomial'] = float('inf')

    best_model_name = min(mse_values, key=mse_values.get)
    if mse_values[best_model_name] == float('inf'):
        raise ValueError("No models could be successfully fitted to the data")

    best_function, *params = models[best_model_name]

    if best_model_name == 'linear':
        a, b = params
        function_code = f"""
import pandas as pd
import numpy as np
import re
def transform(x):
    if pd.isna(x) or str(x).strip().lower() in ['n/a', 'na', 'null', 'nan', '']:
        return "MISSING"
    try:
        x_str = str(x)
        x_str = re.sub(r'[^\d.-]', '', x_str).strip()
        x_num = float(x_str)
        return {a:.6f} * x_num + {b:.6f}
    except (ValueError, TypeError):
        return np.nan
"""
    elif best_model_name == 'exponential':
        a, b = params
        function_code = f"""
import pandas as pd
import numpy as np
import re
def transform(x):
    if pd.isna(x) or str(x).strip().lower() in ['n/a', 'na', 'null', 'nan', '']:
        return "MISSING"
    try:
        x_str = str(x)
        x_str = re.sub(r'[^\d.-]', '', x_str).strip()
        x_num = float(x_str)
        return {a:.6f} * np.exp({b:.6f} * x_num)
    except (ValueError, TypeError):
        return np.nan
"""
    elif best_model_name == 'rational':
        a, b, c = params
        function_code = f"""
import pandas as pd
import numpy as np
import re
def transform(x):
    if pd.isna(x) or str(x).strip().lower() in ['n/a', 'na', 'null', 'nan', '']:
        return "MISSING"
    try:
        x_str = str(x)
        x_str = re.sub(r'[^\d.-]', '', x_str).strip()
        x_num = float(x_str)
        return ({a:.6f} * x_num + {b:.6f}) / (x_num + {c:.6f})
    except (ValueError, TypeError):
        return np.nan
"""
    elif best_model_name == 'polynomial':
        a, b, c = params
        function_code = f"""
import pandas as pd
import numpy as np
import re
def transform(x):
    if pd.isna(x) or str(x).strip().lower() in ['n/a', 'na', 'null', 'nan', '']:
        return "MISSING"
    try:
        x_str = str(x)
        x_str = re.sub(r'[^\d.-]', '', x_str).strip()
        x_num = float(x_str)
        return {a:.6f} * x_num**2 + {b:.6f} * x_num + {c:.6f}
    except (ValueError, TypeError):
        return np.nan
"""
    return function_code


def generate_algorithmic_function(examples):
    # Format up to 2 examples for the prompt
    print(examples)
    example_lines = [f"Input: {src} -> Output: {tgt}" for src, tgt in examples[:2]]
    example_text = "\n".join(example_lines) if example_lines else "No examples provided"

    # Predict specific conversion label under algorithmic transformations
    conversion_label_prompt = f"""
You are an expert in identifying algorithmic transformations based on input-output examples. Your task is to analyze the provided examples and predict the most likely transformation type from a predefined list. The transformation should logically map the input to the output based on common algorithmic patterns.

### Instructions:
1. Carefully examine the input and output formats in the examples.
2. Match the transformation to one of the listed conversion types based on the pattern observed.
3. If the input and output involve dates, prioritize calendar conversions (e.g., Gregorian to Jalali, Gregorian to Hijri) or date-related transformations (e.g., date format change, extract year).
4. For date conversions, note the following:
   - **Gregorian calendar**: Uses YYYY-MM-DD format (e.g., 2023-03-21).
   - **Jalali calendar**: Uses YYYY/M/D format (e.g., 1402/1/1), common in Persian contexts.
   - **Hijri calendar**: Uses YYYY-MM-DD format but with different year/month/day values (e.g., 1444-08-15).
5. Avoid misclassifying non-date transformations (e.g., Roman numerals, binary conversions) when date patterns are evident.
6. If the transformation is unclear, prioritize the most specific match based on the input-output structure.

### Possible Conversion Types (with examples and explanations):
- **gregorian_to_jalali**: Converts Gregorian dates (YYYY-MM-DD) to Jalali dates (YYYY/M/D).
  - Example: Input: 2023-03-21 -> Output: 1402/1/1
  - Reason: Maps Gregorian calendar dates to Persian Jalali calendar.
- **gregorian_to_hijri**: Converts Gregorian dates (YYYY-MM-DD) to Hijri dates (YYYY-MM-DD).
  - Example: Input: 2024-04-10 -> Output: 1445-09-30
  - Reason: Maps Gregorian calendar to Islamic Hijri calendar.
- **date_format_change**: Reformats date structure without changing the calendar system.
  - Example: Input: 2023-01-05 -> Output: 05/01/2023
  - Reason: Changes date format (e.g., YYYY-MM-DD to DD/MM/YYYY).
- **date_to_weekday**: Extracts the day of the week from a date.
  - Example: Input: 2023-04-12 -> Output: Wednesday
  - Reason: Computes the weekday for a given date.
- **dob_to_age**: Calculates age from a date of birth to the current year.
  - Example: Input: 2000-01-01 -> Output: 24
  - Reason: Computes age based on birth date and current year.
- **extract_year_from_date**: Extracts only the year from a date.
  - Example: Input: 2023-12-25 -> Output: 2023
  - Reason: Returns the year component of a date.
- **decimal_to_binary**: Converts a decimal number to binary.
  - Example: Input: 10 -> Output: 1010
  - Reason: Converts base-10 to base-2.
- **binary_to_decimal**: Converts a binary number to decimal.
  - Example: Input: 1010 -> Output: 10
  - Reason: Converts base-2 to base-10.
- **hex_to_decimal**: Converts a hexadecimal number to decimal.
  - Example: Input: 1F -> Output: 31
  - Reason: Converts base-16 to base-10.
- **decimal_to_roman**: Converts a decimal number to Roman numerals.
  - Example: Input: 4 -> Output: IV
  - Reason: Maps numbers to Roman numeral representation.
- **roman_to_decimal**: Converts Roman numerals to decimal numbers.
  - Example: Input: IX -> Output: 9
  - Reason: Maps Roman numerals to base-10 numbers.
- **char_to_ascii**: Converts a character to its ASCII code.
  - Example: Input: A -> Output: 65
  - Reason: Returns the ASCII value of a character.
- **ascii_to_char**: Converts an ASCII code to a character.
  - Example: Input: 66 -> Output: B
  - Reason: Maps ASCII value to corresponding character.
- **base64_encode**: Encodes a string to Base64.
  - Example: Input: hello -> Output: aGVsbG8=
  - Reason: Applies Base64 encoding to text.
- **base64_decode**: Decodes a Base64 string to text.
  - Example: Input: aGVsbG8= -> Output: hello
  - Reason: Decodes Base64 to original text.
- **unicode_to_char**: Converts a Unicode code point to a character.
  - Example: Input: U+0627 -> Output: ا
  - Reason: Maps Unicode code to corresponding character.
- **convert_12h_to_24h**: Converts 12-hour time to 24-hour time.
  - Example: Input: 03:45 PM -> Output: 15:45
  - Reason: Converts AM/PM time to 24-hour format.
- **minutes_to_hhmm**: Converts total minutes to HH:MM format.
  - Example: Input: 135 -> Output: 02:15
  - Reason: Converts minutes to hours and minutes.
- **time_to_seconds**: Converts time (HH:MM:SS) to total seconds.
  - Example: Input: 01:30:00 -> Output: 5400
  - Reason: Converts time to seconds.
- **extract_domain_from_email**: Extracts the domain from an email address.
  - Example: Input: test@example.com -> Output: example.com
  - Reason: Returns the domain part of an email.
- **extract_path_from_url**: Extracts the path from a URL.
  - Example: Input: https://site.com/path/info -> Output: /path/info
  - Reason: Returns the path component of a URL.
- **validate_ip_format**: Checks if a string is a valid IP address.
  - Example: Input: 192.168.1.1 -> Output: valid
  - Reason: Validates IPv4 address format.

### Actual Examples:
{example_text}

### Response:
Respond **only** with the name of the best matching conversion type (e.g., gregorian_to_jalali). Do not include explanations or additional text.
"""

    # Simulate the response generation (replace with actual model call)
    label_response = generate_response(conversion_label_prompt, tokens=100,gpt=True)
    clean_label_response = extract_response_only(conversion_label_prompt, label_response)
    conversion_type = extract_conversion_type(clean_label_response)
    print("Predicted Conversion Type:", conversion_type)
    

    # Map conversion labels to base functions
    conversion_map = {
        "gregorian_to_jalali": at.gregorian_to_jalali,
        "gregorian_to_hijri": at.gregorian_to_hijri,
        "date_format_change": at.date_format_change,
        "date_to_weekday": at.date_to_weekday,
        "dob_to_age": at.dob_to_age,
        "extract_year_from_date": at.extract_year_from_date,
        "decimal_to_binary": at.decimal_to_binary,
        "binary_to_decimal": at.binary_to_decimal,
        "hex_to_decimal": at.hex_to_decimal,
        "decimal_to_roman": at.decimal_to_roman,
        "roman_to_decimal": at.roman_to_decimal,
        "char_to_ascii": at.char_to_ascii,
        "ascii_to_char": at.ascii_to_char,
        "base64_encode": at.base64_encode,
        "base64_decode": at.base64_decode,
        "unicode_to_char": at.unicode_to_char,
        "convert_12h_to_24h": at.convert_12h_to_24h,
        "minutes_to_hhmm": at.minutes_to_hhmm,
        "time_to_seconds": at.time_to_seconds,
        "extract_domain_from_email": at.extract_domain_from_email,
        "extract_path_from_url": at.extract_path_from_url,
        "validate_ip_format": at.validate_ip_format
    }

    base_code = conversion_map.get(conversion_type, lambda: "def transform(x): return ''")()

    # Prompt LLaMA to refine the code using the examples
    refinement_prompt = f"""
You are given a base Python function and examples of input/output transformations.

Function:

{base_code}


Examples:
{example_text}

Modify the function so it exactly follows the transformation pattern shown in the examples.
Return only the final Python code inside triple backticks.
"""

    refined_response = generate_response(refinement_prompt, tokens=600)
    #print(refined_response)
    final_code = extract_code_from_response(refined_response)

    return final_code
   

def generate_general_lookup_function(examples):
    # Convert examples to a prompt-friendly format, using up to 2 examples for brevity
    example_lines = [f"Input: {src} -> Output: {tgt}" for src, tgt in examples[:2]]
    example_text = "\n".join(example_lines) if example_lines else "No examples provided"

    prompt = f"""
Write a Python function to perform a lookup-based transformation for the following examples:
{example_text}

The function should:
- Create a dictionary `lookup_table` mapping each input to its corresponding output.
- Include all provided input-output pairs in `lookup_table`.
- Augment `lookup_table` with additional relevant mappings based on your knowledge, inferred from the pattern in the examples (e.g., for company-to-CEO mappings, add other well-known company-CEO pairs).
- If you cannot generate additional mappings due to limited knowledge or ambiguous patterns, use only the provided pairs.
- Define a function named `transform` that takes a single string input `x` and returns the output from `lookup_table`, or "UNKNOWN" if the input is not found.
- Use only standard Python constructs (e.g., dictionaries, get method).
- Match this exact structure:
```python
lookup_table = {{input_output_dict}}

def transform(x):
    return lookup_table.get(x, "UNKNOWN")
```
where `{{input_output_dict}}` is a dictionary containing all provided pairs and any additional mappings, e.g., {{"Microsoft": "Satya Nadella", "Pepsico": "Ramon Laguarta", "Apple": "Tim Cook"}}.

**Instructions**:
1. **Analyze the Examples**:
   - Identify the transformation pattern from the provided input-output pairs (e.g., company to CEO, country to capital).
   - Ensure all provided pairs are included in `lookup_table` with exact matches.

2. **Augment with Knowledge**:
   - Based on the pattern, add up to 3 additional relevant mappings using your knowledge (e.g., for company-to-CEO, add well-known pairs like "Apple": "Tim Cook").
   - Ensure additional mappings are accurate, relevant, and consistent with the pattern.
   - If the pattern is unclear or you lack sufficient knowledge, include only the provided pairs.

3. **Build the Dictionary**:
   - Create `lookup_table` as a dictionary with string keys and values, preserving the exact values from the examples.
   - If an input appears multiple times, use the last output value provided.
   - Format the dictionary as a valid Python dictionary with proper string quotes.

4. **Generate the Function**:
   - Define the `transform` function to use `lookup_table.get(x, "UNKNOWN")`.
   - Use 4-space indentation for the function body.
   - Ensure the output is syntactically correct and executable.

5. **Handle Edge Cases**:
   - If no examples are provided, use an empty dictionary (`lookup_table = {{}}`) and do not add mappings.
   - If the pattern is ambiguous, prioritize the provided pairs and do not add speculative mappings.
   - Avoid hallucinating incorrect or irrelevant mappings; if uncertain, use only the provided pairs.

**Constraints**:
- Do not use external libraries, APIs, or network calls; rely on your pre-trained knowledge.
- Ensure the output is syntactically correct Python code.
- Do not include backticks, code fences, comments, or additional text in the response.
- Limit additional mappings to 3 to keep the dictionary concise and relevant.

**Examples**:
1. For examples:
   Input: Microsoft -> Output: Satya Nadella
   Input: Pepsico -> Output: Ramon Laguarta
   Output (augmented with knowledge):
   lookup_table = {{"Microsoft": "Satya Nadella", "Pepsico": "Ramon Laguarta", "Apple": "Tim Cook", "Google": "Sundar Pichai"}}
   
   def transform(x):
       return lookup_table.get(x, "UNKNOWN")

2. For examples:
   Input: France -> Output: Paris
   Input: Japan -> Output: Tokyo
   Output (augmented with knowledge):
   lookup_table = {{"France": "Paris", "Japan": "Tokyo", "Brazil": "Brasília", "Canada": "Ottawa"}}
   
   def transform(x):
       return lookup_table.get(x, "UNKNOWN")

3. For examples:
   Input: The Great Gatsby -> Output: F. Scott Fitzgerald
   Input: 1984 -> Output: George Orwell
   Output (augmented with knowledge):
   lookup_table = {{"The Great Gatsby": "F. Scott Fitzgerald", "1984": "George Orwell", "To Kill a Mockingbird": "Harper Lee"}}
   
   def transform(x):
       return lookup_table.get(x, "UNKNOWN")

4. For no examples:
   Output:
   lookup_table = {{}}
   
   def transform(x):
       return lookup_table.get(x, "UNKNOWN")

5. For examples with unclear pattern:
   Input: ABC123 -> Output: XYZ789
   Input: DEF456 -> Output: UVW012
   Output (no additional mappings due to ambiguity):
   lookup_table = {{"ABC123": "XYZ789", "DEF456": "UVW012"}}
   
   def transform(x):
       return lookup_table.get(x, "UNKNOWN")
"""

    full_response = generate_response(prompt,tokens=500,gpt=True)
    response=extract_response_only(prompt,full_response)
    return extract_code_from_response(response)

# ===== APPLY FUNCTION =====
def apply_function_on_column(function_code, column):
    local_scope = {}
    exec(function_code, {"re":re,"np":np,"pd":pd}, local_scope)
    transform_func = local_scope.get('transform', lambda x: x)

    def safe_transform(x):
        try:
            return transform_func(x)
        except Exception as e:
            print(f"Transform error for input {x}: {e}")
            return x


    return column.apply(safe_transform)
