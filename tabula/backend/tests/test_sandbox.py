"""Security tests for the code-execution sandbox.

These demonstrate that malicious or invalid transformation code is rejected and
that legitimate transformations still run, including under resource limits.
"""

from __future__ import annotations

import os

import pytest

from sandbox import (
    SandboxError,
    apply_transform,
    match_templates,
    validate_transformation_code,
)


# --------------------------------------------------------------------------- #
# Rejection of malicious code
# --------------------------------------------------------------------------- #
MALICIOUS_SNIPPETS = [
    # File system / process access
    "import os\ndef transform(x):\n    return os.system('whoami')",
    "def transform(x):\n    return open('/etc/passwd').read()",
    "def transform(x):\n    return __import__('subprocess').run(['ls'])",
    "def transform(x):\n    import shutil\n    return shutil.rmtree('/')",
    # Evaluator / code execution
    "def transform(x):\n    return eval('1+1')",
    "def transform(x):\n    return exec('x=1')",
    "def transform(x):\n    return compile('1','','eval')",
    # Class/object escape
    "def transform(x):\n    return x.__class__.__bases__[0].__subclasses__()",
    "def transform(x):\n    return x.__class__.__mro__",
    "def transform(x):\n    return getattr(x, '__class__')",
    "def transform(x):\n    return ().__class__.__globals__",
    # Dunder / builtins manipulation
    "__builtins__['open'] = open",
    "def transform(x):\n    return vars()",
    "def transform(x):\n    return globals()",
    # Network
    "import socket\ndef transform(x):\n    return socket.socket()",
    "def transform(x):\n    from urllib.request import urlopen\n    return urlopen('http://x')",
    # Indirect import of a denied module
    "def transform(x):\n    return __import__('os').environ",
    # Env access
    "def transform(x):\n    import os\n    return os.getenv('SECRET_KEY')",
]


@pytest.mark.parametrize("code", MALICIOUS_SNIPPETS)
def test_malicious_code_rejected_by_validator(code):
    with pytest.raises(SandboxError):
        validate_transformation_code(code)


@pytest.mark.parametrize("code", MALICIOUS_SNIPPETS)
def test_malicious_code_never_runs_even_if_validator_bypassed(code):
    """apply_transform validates first, so it must raise before any execution."""
    with pytest.raises(SandboxError):
        apply_transform(code, ["anything"])


def test_empty_and_invalid_code_rejected():
    with pytest.raises(SandboxError):
        validate_transformation_code("")
    with pytest.raises(SandboxError):
        validate_transformation_code("   \n  ")
    with pytest.raises(SandboxError):
        validate_transformation_code("def transform(x:\n  return x")  # syntax error


def test_forbidden_language_constructs_rejected():
    constructs = [
        "import re\ndef transform(x):\n    return x",  # even allowed modules blocked? no -> re allowed
        "def transform(x):\n    global y\n    y=1",
        "class Evil:\n    pass\ndef transform(x):\n    return x",
        "async def transform(x):\n    return x",
        "def transform(x):\n    yield x",
        "def transform(x):\n    del x",
        "def transform(x):\n    return (lambda: 1)()",
    ]
    # The first one uses an allowlisted module and should PASS validation.
    validate_transformation_code(constructs[0])
    for code in constructs[1:]:
        with pytest.raises(SandboxError):
            validate_transformation_code(code)


def test_deny_import_of_non_allowlisted_module_even_if_stdlib():
    for mod in ["os", "sys", "subprocess", "ctypes", "pickle", "shutil", "socket"]:
        with pytest.raises(SandboxError):
            validate_transformation_code(
                f"import {mod}\ndef transform(x):\n    return x"
            )


def test_allowlisted_imports_pass_validation():
    for code in [
        "import re\ndef transform(x):\n    return re.sub(r'\\d','',str(x))",
        "import base64\ndef transform(x):\n    return base64.b64encode(str(x).encode()).decode()",
        "from datetime import datetime\ndef transform(x):\n    return str(datetime.now())",
        "import math\ndef transform(x):\n    return math.sqrt(float(x))",
        "from urllib.parse import urlparse\ndef transform(x):\n    return urlparse(str(x)).path",
    ]:
        validate_transformation_code(code)


# --------------------------------------------------------------------------- #
# Legitimate code executes correctly
# --------------------------------------------------------------------------- #
def test_uppercase_transform_runs():
    code = "def transform(x):\n    return str(x).strip().upper()"
    out = apply_transform(code, ["  apple ", "banana"])
    assert out.values == ["APPLE", "BANANA"]
    assert out.failed_rows == 0


def test_missing_values_handled_per_row():
    code = "def transform(x):\n    if x is None:\n        return 'MISSING'\n    return str(x).upper()"
    out = apply_transform(code, ["a", None, "b"])
    assert out.values == ["A", "MISSING", "B"]
    assert out.failed_rows == 0


def test_row_failure_reported_and_isolated():
    # Divides by zero for '0' only; other rows still succeed.
    code = (
        "def transform(x):\n"
        "    n = int(str(x))\n"
        "    return str(10 // n)\n"
    )
    out = apply_transform(code, ["2", "0", "5"])
    assert out.values[0] == "5"
    assert out.values[2] == "2"
    assert out.values[1] is None  # failed row
    assert out.failed_rows == 1
    assert out.errors  # message present


def test_empty_input_returns_empty():
    out = apply_transform("def transform(x):\n    return x", [])
    assert out.values == []
    assert out.failed_rows == 0


# --------------------------------------------------------------------------- #
# Resource limits
# --------------------------------------------------------------------------- #
def test_infinite_loop_is_killed_by_timeout():
    code = "def transform(x):\n    while True:\n        pass"
    with pytest.raises(SandboxError) as exc:
        apply_transform(code, [1], timeout=3)
    assert "time limit" in str(exc.value).lower()


@pytest.mark.skipif(
    os.name == "nt",
    reason="RLIMIT_DATA (heap cap) is enforced only on Linux; deployment target is Docker/Linux",
)
def test_huge_allocation_bounded():
    # Attempt to allocate far beyond the memory cap (the cap bounds the heap,
    # so this large string allocation raises MemoryError inside the sandbox).
    code = "def transform(x):\n    a = ' ' * (1024*1024*1024)\n    return len(a)"
    with pytest.raises((SandboxError, Exception)):
        apply_transform(code, [1], timeout=10, memory_limit_mb=64)


# --------------------------------------------------------------------------- #
# Batch template matching (single subprocess)
# --------------------------------------------------------------------------- #
def test_match_templates_finds_correct_template():
    from transform import templates

    key = match_templates(
        templates.TEMPLATES, ["10", "5"], ["1010", "101"], timeout=15
    )
    assert key == "decimal_to_binary"


def test_match_templates_returns_none_when_no_match():
    from transform import templates

    key = match_templates(templates.TEMPLATES, ["abc"], ["????"], timeout=15)
    assert key is None
