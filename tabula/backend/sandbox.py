"""Safe execution sandbox for transformation code.

Design
------
The application **never executes Python source that a client supplies
directly**.  Transformation code is produced server-side (from our own
deterministic templates or from an AST-validated LLM response) and is only ever
run after passing :func:`validate_transformation_code`.

Two layers of defence are applied:

1. **Static AST validation** -- the code is parsed and walked.  Any construct
   that could be used for escape or side effects (imports, ``exec``/``eval``,
   dunder attribute access, ``global``, class/async definitions, ...) causes a
   :class:`SandboxError`.  Only a conservative subset of Python is accepted.

2. **Process isolation** -- the validated code is executed in a separate
   short-lived process with a wall-clock timeout, an optional address-space
   limit (Linux only), an empty environment and *no* application secrets.  A
   hang or crash cannot take down the API process, and the child cannot read
   ``os.environ`` secrets because we pass a cleared environment.

This is defence-in-depth on top of the architectural rule that clients never
submit raw Python.  It is *not* marketed as a perfect sandbox: a determined
attacker with a valid Python interpreter and no OS-level seccomp/namespace
isolation could still attempt resource-exhaustion tricks.  Running the API
inside a locked-down container (no privileges, read-only rootfs, resource
quotas) is recommended and is what the provided Docker setup does.
"""

from __future__ import annotations

import ast
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "SandboxError",
    "validate_transformation_code",
    "apply_transform",
    "TransformOutcome",
]


class SandboxError(Exception):
    """Raised when code is rejected by the validator or fails at runtime."""


# --------------------------------------------------------------------------- #
# Static AST validation
# --------------------------------------------------------------------------- #

# Node types that are always forbidden regardless of context.
_FORBIDDEN_NODES = tuple(
    node
    for node in (
        ast.Global,
        ast.Nonlocal,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Delete,
        ast.Lambda,  # keep behaviour explicit and auditable
        getattr(ast, "Match", None),  # structural pattern matching (py3.10+)
    )
    if node is not None
)

# Modules that transformation code is permitted to import.  Kept deliberately
# small; everything else is rejected at validation *and* at runtime by the
# wrapped ``__import__`` below.
_ALLOWED_IMPORTS = frozenset(
    {
        "re",
        "math",
        "json",
        "string",
        "base64",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "urllib",
        "urllib.parse",
        "hijri_converter",
        "numpy",
        "pandas",
        "calendar",
        "decimal",
        "fractions",
        "statistics",
        "unicodedata",
    }
)

# Builtins that must never be reachable even if a future refactor widens scope.
_FORBIDDEN_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "object",
        "super",
        "memoryview",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "license",
        "copyright",
        "credit",
        "print",
        "buffer",
        "file",
        "reload",
        "id",
        "hash",
        "callable",
        "classmethod",
        "staticmethod",
        "property",
    }
)

# Module names that must never be referenced.
_FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "ssl",
        "ctypes",
        "pickle",
        "marshal",
        "importlib",
        "builtins",
        "inspect",
        "threading",
        "multiprocessing",
        "signal",
        "pathlib",
        "tempfile",
        "webbrowser",
        "requests",
        "urllib.request",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "code",
        "codeop",
        "pty",
        "fcntl",
        "resource",
    }
)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def validate_transformation_code(code: str) -> ast.Module:
    """Parse and validate transformation code, returning the AST on success.

    Raises :class:`SandboxError` with an actionable message on rejection.
    """
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("Transformation code is empty.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:  # pragma: no cover - trivial
        raise SandboxError(f"Transformation code has a syntax error: {exc.msg}") from exc

    for node in ast.walk(tree):
        # 1. Forbidden constructs.
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxError(
                f"Forbidden construct '{type(node).__name__}' in transformation code."
            )

        # 1b. Imports are allowed only for a small allowlist of modules.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    raise SandboxError(
                        f"Import of module '{alias.name}' is not allowed."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in _ALLOWED_IMPORTS:
                raise SandboxError(f"Import from module '{module}' is not allowed.")

        # 2. Names / attributes.
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise SandboxError(f"Forbidden name '{node.id}' in transformation code.")
            if node.id in _FORBIDDEN_MODULES:
                raise SandboxError(
                    f"Forbidden module '{node.id}' in transformation code."
                )
            if _is_dunder(node.id):
                raise SandboxError(
                    f"Dunder name '{node.id}' is not allowed in transformation code."
                )

        if isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                raise SandboxError(
                    f"Dunder attribute '{node.attr}' is not allowed in transformation code."
                )
            if node.attr in _FORBIDDEN_NAMES:
                raise SandboxError(
                    f"Forbidden attribute '{node.attr}' in transformation code."
                )

        # 3. Reject access to dangerous builtins passed as bare names is handled
        #    above; also reject ``__builtins__`` style assignment targets.
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name) and _is_dunder(sub.id):
                        raise SandboxError(
                            f"Cannot assign to dunder name '{sub.id}'."
                        )

    return tree


# --------------------------------------------------------------------------- #
# Restricted execution environment
# --------------------------------------------------------------------------- #
def _build_safe_globals() -> Dict[str, Any]:
    """Construct the restricted global namespace available to transforms."""
    import base64
    import datetime as _datetime
    import json as _json
    import math as _math
    import re as _re
    import string as _string
    from urllib.parse import urlparse as _urlparse

    import numpy as _np
    import pandas as _pd

    # A restricted ``__import__``.  The *static* AST check is the primary gate
    # (it only permits `_ALLOWED_IMPORTS`).  At runtime we enforce a *blocklist*
    # so that allowlisted stdlib modules can still perform their own internal
    # (often private, e.g. ``_strptime``) imports.  Anything on the blocklist is
    # refused even if reached indirectly.
    _real_import = __import__
    _RUNTIME_DENIED = frozenset(
        {
            "os", "sys", "subprocess", "shutil", "socket", "ssl", "ctypes",
            "pickle", "marshal", "importlib", "builtins", "inspect",
            "threading", "multiprocessing", "signal", "pathlib", "tempfile",
            "webbrowser", "requests", "http", "ftplib", "smtplib", "telnetlib",
            "xmlrpc", "code", "codeop", "pty", "fcntl", "resource", "sysconfig",
            "pty", "posix", "nt", "winreg", "msvcrt", "grp", "pwd", "termios",
            "tty", "readline", "rlcompleter", "venv", "zipimport", "runpy",
            "pkgutil", "modulefinder", "parser", "symbol", "token", "tokenize",
            "py_compile", "compileall", "importlib.util", "importlib.machinery",
        }
    )

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Relative imports are not allowed.")
        root = name.split(".")[0]
        if root in _RUNTIME_DENIED or name in _RUNTIME_DENIED:
            raise ImportError(f"Import of module '{name}' is not allowed.")
        return _real_import(name, globals, locals, fromlist, level)

    # A deliberately small builtin surface.  Anything not listed is unavailable.
    safe_builtins: Dict[str, Any] = {
        "__import__": _restricted_import,
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "bool": bool,
        "bytearray": bytearray,
        "bytes": bytes,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        # Exceptions that transforms legitimately catch.
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "ZeroDivisionError": ZeroDivisionError,
        "OverflowError": OverflowError,
        "LookupError": LookupError,
        "ArithmeticError": ArithmeticError,
        "UnicodeError": UnicodeError,
        "StopIteration": StopIteration,
    }

    return {
        "__builtins__": safe_builtins,
        "re": _re,
        "np": _np,
        "pd": _pd,
        "math": _math,
        "datetime": _datetime,
        "date": _datetime.date,
        "time": _datetime.time,
        "timedelta": _datetime.timedelta,
        "base64": base64,
        "json": _json,
        "string": _string,
        "urlparse": _urlparse,
        "MISSING": "MISSING",
    }


# The module-level namespace used by the child process.
_SAFE_GLOBALS: Optional[Dict[str, Any]] = None


def _get_safe_globals() -> Dict[str, Any]:
    global _SAFE_GLOBALS
    if _SAFE_GLOBALS is None:
        _SAFE_GLOBALS = _build_safe_globals()
    return _SAFE_GLOBALS


# --------------------------------------------------------------------------- #
# Child-process worker
# --------------------------------------------------------------------------- #
def _child_main(code: str, values: List[Any], conn) -> None:  # pragma: no cover
    """Runs in a forked/spawned child.  Sends back (ok, payload)."""
    try:
        # Read the configured limit BEFORE clearing the environment below, so
        # the parent-provided value is not wiped along with the secrets.
        mem_limit_mb = int(os.environ.get("TABULAX_SANDBOX_MEM_MB", "512"))

        # Drop inherited environment so application secrets are not visible.
        for key in list(os.environ):
            os.environ.pop(key, None)

        # Apply resource limits where the platform supports them.
        try:  # Linux only
            import resource

            limit_bytes = mem_limit_mb * 1024 * 1024
            # RLIMIT_DATA bounds the heap (where runaway string/list/array
            # allocations live) without counting memory-mapped shared objects.
            # RLIMIT_AS would be wrong here: importing pandas reserves ~560 MB
            # of *virtual* address space for its .so mappings, which would be
            # rejected as "failed to map segment" even though real usage is tiny.
            resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        except Exception:
            pass

        tree = ast.parse(code, mode="exec")
        namespace = dict(_get_safe_globals())
        exec(compile(tree, "<transform>", "exec"), namespace)  # noqa: S102 - validated above
        fn = namespace.get("transform")
        if not callable(fn):
            conn.send((False, "Code must define a callable named 'transform'."))
            return

        results = []
        for v in values:
            try:
                results.append(fn(v))
            except Exception as row_exc:  # per-row resilience
                results.append({"__error__": f"{type(row_exc).__name__}: {row_exc}"})
        conn.send((True, results))
    except MemoryError:
        conn.send((False, "Transformation exceeded the memory limit."))
    except Exception as exc:  # noqa: BLE001
        conn.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_in_subprocess(
    code: str, values: List[Any], timeout: float, memory_limit_mb: int
) -> List[Any]:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    env_backup = os.environ.get("TABULAX_SANDBOX_MEM_MB")
    os.environ["TABULAX_SANDBOX_MEM_MB"] = str(memory_limit_mb)
    proc = ctx.Process(target=_child_main, args=(code, values, child_conn))
    proc.daemon = True
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout):
            ok, payload = parent_conn.recv()
            if ok:
                return payload
            raise SandboxError(str(payload))
        # Timed out: kill the child.
        proc.kill()
        raise SandboxError(
            f"Transformation exceeded the time limit of {timeout:.0f}s and was stopped."
        )
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=1)
        if env_backup is None:
            os.environ.pop("TABULAX_SANDBOX_MEM_MB", None)
        else:
            os.environ["TABULAX_SANDBOX_MEM_MB"] = env_backup


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class TransformOutcome:
    """Result of applying a validated transform to a column of values."""

    values: List[Any]
    failed_rows: int
    errors: List[str]  # distinct failure messages (capped)

    @property
    def ok(self) -> bool:
        return self.failed_rows == 0


def apply_transform(
    code: str,
    values: Sequence[Any],
    timeout: float = 15.0,
    memory_limit_mb: int = 512,
) -> TransformOutcome:
    """Validate ``code`` then apply ``transform`` to every value in ``values``.

    Returns a :class:`TransformOutcome`.  Raises :class:`SandboxError` if the
    code is rejected, fails to define ``transform``, or exceeds limits.
    """
    validate_transformation_code(code)
    values_list = list(values)
    if not values_list:
        return TransformOutcome(values=[], failed_rows=0, errors=[])

    raw_results = _run_in_subprocess(code, values_list, timeout, memory_limit_mb)

    clean: List[Any] = []
    errors: List[str] = []
    failed_rows = 0
    seen_msgs: set[str] = set()
    for r in raw_results:
        if isinstance(r, dict) and "__error__" in r:
            msg = r["__error__"]
            clean.append(None)
            failed_rows += 1
            if msg not in seen_msgs and len(errors) < 20:
                seen_msgs.add(msg)
                errors.append(msg)
        else:
            clean.append(r)

    return TransformOutcome(values=clean, failed_rows=failed_rows, errors=errors)


# --------------------------------------------------------------------------- #
# Batch template matching (single subprocess for speed)
# --------------------------------------------------------------------------- #
def _match_worker(codes: List[str], values: List[Any], expected: List[str], conn) -> None:
    """Child: run every candidate transform and report which match exactly."""
    try:
        for key in list(os.environ):
            os.environ.pop(key, None)
        results = {}
        for code in codes:
            try:
                tree = ast.parse(code, mode="exec")
                namespace = dict(_get_safe_globals())
                exec(compile(tree, "<transform>", "exec"), namespace)  # noqa: S102
                fn = namespace.get("transform")
                if not callable(fn):
                    continue
                got = []
                for v in values:
                    try:
                        got.append(fn(v))
                    except Exception:
                        got.append("__raised__")
                got = ["" if g is None else str(g) for g in got]
                results[code] = got == expected
            except Exception:
                results[code] = False
        conn.send(results)
    except Exception as exc:  # pragma: no cover
        conn.send({"__error__": str(exc)})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def match_templates(
    template_sources: Dict[str, str],
    values: Sequence[Any],
    expected: Sequence[str],
    timeout: float = 10.0,
) -> Optional[str]:
    """Return the key of the template whose output equals ``expected`` exactly.

    All candidates are evaluated inside a **single** subprocess for speed.
    Returns ``None`` when no template matches.  Values are compared as strings.
    """
    if not template_sources:
        return None
    values_list = list(values)
    expected_list = ["" if e is None else str(e) for e in expected]
    if not values_list:
        return None

    # Validate every candidate up front (cheap, local).
    valid: Dict[str, str] = {}
    for key, src in template_sources.items():
        try:
            validate_transformation_code(src)
            valid[key] = src
        except SandboxError:
            continue
    if not valid:
        return None

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_match_worker, args=(list(valid.values()), values_list, expected_list, child_conn)
    )
    proc.daemon = True
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout):
            proc.kill()
            return None
        payload = parent_conn.recv()
        if not isinstance(payload, dict) or "__error__" in payload:
            return None
        # Map code -> key
        code_to_key = {src: key for key, src in valid.items()}
        for code, matched in payload.items():
            if matched:
                return code_to_key.get(code)
        return None
    except Exception:  # pragma: no cover
        return None
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=1)
