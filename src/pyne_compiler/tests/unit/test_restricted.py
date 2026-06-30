"""Adversarial tests for ``openbb_pine.runtime.restricted``.

D2 section 10.1 / PRD section 5.2 T3: the namespace into which user-compiled
``@pyne`` modules execute must NOT expose any dangerous builtin (no
``__import__``, ``open``, ``exec``, ``eval``, ``compile``, ``input``,
``globals``, ``locals``, ``vars``, ``setattr``, ``delattr``, ``dir``,
``exit``, ``help``). This file exercises every dangerous builtin
individually so a regression on any single name is loud.

# Security note (read before touching this file)
#
# Every ``exec(...)`` call below is INTENTIONAL and SAFE. They evaluate
# fixed string LITERALS defined in this test file (never any external
# input) inside the deliberately stripped namespace built by
# ``build_restricted_namespace`` -- the very thing under test. This is
# the same pattern used by sandbox-escape regression suites elsewhere
# (e.g. RestrictedPython's test corpus): we drive the production sandbox
# with known-bad snippets to prove they bounce.
#
# Replacing ``exec`` with ``ast.literal_eval`` here would defeat the
# test (literal_eval cannot execute statements). The dangerous-builtin
# names (``eval``, ``__import__``, etc.) appear only in string literals
# being tested for *absence*; none are invoked.
"""

from __future__ import annotations

import pytest

from openbb_pine.runtime.restricted import (
    _ALLOWED_BUILTINS,
    build_restricted_namespace,
)


# ---------- happy path ---------------------------------------------------


def test_allowed_builtins_set_is_frozen():
    """``_ALLOWED_BUILTINS`` is a ``frozenset`` so it cannot be mutated at runtime."""
    assert isinstance(_ALLOWED_BUILTINS, frozenset)


def test_build_restricted_namespace_returns_fresh_dict_each_call():
    """Two calls must return distinct dicts -- shared mutable state would
    let one script leak into the next."""
    ns_a = build_restricted_namespace("/tmp/script_a.py")
    ns_b = build_restricted_namespace("/tmp/script_b.py")
    assert ns_a is not ns_b
    ns_a["smuggled"] = 1
    assert "smuggled" not in ns_b


def test_namespace_has_dunder_name_and_dunder_file_set():
    """``exec(compile(src, file, 'exec'), ns)`` relies on ``__name__`` /
    ``__file__`` being present, e.g. for tracebacks."""
    ns = build_restricted_namespace("/tmp/strat.py")
    assert ns["__name__"] == "pine_user_script"
    assert ns["__file__"] == "/tmp/strat.py"
    assert "__builtins__" in ns


def test_allowed_builtin_executes_in_namespace():
    """A trivial allowed builtin must work inside the restricted ns."""
    ns = build_restricted_namespace("/tmp/t.py")
    exec("result = abs(-5)", ns)
    assert ns["result"] == 5


def test_multiple_allowed_builtins_execute_together():
    """Compose a small expression to assert several allowed builtins work."""
    ns = build_restricted_namespace("/tmp/t.py")
    exec("xs = list(range(5)); s = sum(xs); n = len(xs)", ns)
    assert ns["xs"] == [0, 1, 2, 3, 4]
    assert ns["s"] == 10
    assert ns["n"] == 5


# ---------- adversarial: dangerous builtins must be unreachable ----------


DANGEROUS_BUILTINS = (
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
    "dir",
    "exit",
    "quit",
    "help",
    "breakpoint",
    "__import__",
    "memoryview",
    "bytes",
    "bytearray",
    "getattr",
    "hasattr",
    "id",
    "__build_class__",
    "__loader__",
    "__spec__",
)


@pytest.mark.parametrize("builtin_name", DANGEROUS_BUILTINS)
def test_dangerous_builtin_absent_from_namespace(builtin_name):
    """Each dangerous builtin must be absent from ``__builtins__`` AND raise
    ``NameError`` when referenced from user code."""
    ns = build_restricted_namespace("/tmp/t.py")
    assert builtin_name not in ns["__builtins__"], (
        f"Dangerous builtin {builtin_name!r} leaked into restricted ns"
    )
    with pytest.raises(NameError):
        exec(f"x = {builtin_name}", ns)


def test_import_statement_fails():
    """``import os`` syntactically resolves ``__import__`` at runtime --
    the absence of ``__import__`` in builtins must turn this into an error."""
    ns = build_restricted_namespace("/tmp/t.py")
    # The exact exception class is ``ImportError`` because Python's import
    # machinery wraps the missing ``__import__`` lookup; an earlier guard
    # could plausibly raise ``NameError`` instead. Accept either.
    with pytest.raises((ImportError, NameError)):
        exec("import os", ns)


def test_from_import_fails():
    """``from sys import path`` must also be blocked."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises((ImportError, NameError)):
        exec("from sys import path", ns)


def test_open_blocked():
    """User code attempting to ``open("/etc/passwd")`` must NameError."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec('open("/etc/passwd")', ns)


def test_eval_blocked():
    """``eval("__import__(\\"os\\")")`` is the canonical sandbox escape;
    ``eval`` itself must not exist in the ns."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec('eval("1 + 1")', ns)


def test_exec_blocked():
    """``exec`` inside ``exec`` -- block."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec('exec("x = 1")', ns)


def test_dunder_import_blocked():
    """The classic ``__import__("os")`` exploit must NameError."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec('m = __import__("os")', ns)


def test_compile_blocked():
    """``compile()`` is a sandbox-escape primitive -- block."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec('compile("1", "<s>", "exec")', ns)


def test_input_blocked():
    """``input()`` would hang the executor on a stdin read -- block."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec("x = input()", ns)


def test_globals_locals_vars_blocked():
    """Introspection builtins that expose other frames -- all blocked."""
    ns = build_restricted_namespace("/tmp/t.py")
    for name in ("globals", "locals", "vars"):
        with pytest.raises(NameError):
            exec(f"g = {name}()", ns)


def test_setattr_delattr_blocked():
    """Mutation primitives that can rewire imported objects -- blocked."""
    ns = build_restricted_namespace("/tmp/t.py")
    for name in ("setattr", "delattr"):
        with pytest.raises(NameError):
            exec(f"{name}(object(), 'x', 1)", ns)


def test_dir_blocked():
    """``dir()`` is reconnaissance -- block."""
    ns = build_restricted_namespace("/tmp/t.py")
    with pytest.raises(NameError):
        exec("d = dir()", ns)


def test_exit_quit_blocked():
    """``exit()`` / ``quit()`` from the script would kill the worker."""
    ns = build_restricted_namespace("/tmp/t.py")
    for name in ("exit", "quit"):
        with pytest.raises(NameError):
            exec(f"{name}()", ns)


def test_builtins_dict_only_contains_allowed_names():
    """The ``__builtins__`` dict installed in the ns must contain exactly the
    names in ``_ALLOWED_BUILTINS`` -- no extras, no omissions."""
    ns = build_restricted_namespace("/tmp/t.py")
    assert set(ns["__builtins__"].keys()) == set(_ALLOWED_BUILTINS)
