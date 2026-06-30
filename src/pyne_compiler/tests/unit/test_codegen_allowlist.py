"""Dedicated allowlist-gate tests for codegen (C5, bead 0e9.5.5).

D1 §3.3 is the language-level T1 mitigation. These tests hand-construct
``ast.Module`` instances that violate each of CG001 (node type), CG002
(module), CG003 (name) and assert :func:`_enforce_allowlist` rejects
them with structured :class:`PineCodegenError` carrying the rule code
the gate uses.

The tests bypass :func:`emit` entirely — they exercise the gate
in isolation so a future emitter refactor doesn't have to invent
malicious Pine source to exercise each rule.

.. security-note::

    Several tests pass strings like ``"x = eval('1+1')\\n"`` or
    ``"x = __import__('os')\\n"`` to :func:`ast.parse`. These strings
    are **parsed** (into an ``ast.Module`` for shape-checking), never
    **executed**. Their entire purpose is to verify that the codegen
    allowlist gate REJECTS exactly these dangerous shapes — this is
    the negative side of the T1 mitigation test. Calling ``eval`` /
    ``exec`` / ``__import__`` is explicitly what the gate must
    prevent; a test that didn't construct these shapes would leave
    the rule unverified.

Reference: ``openbb_pine/compiler/codegen.py::_enforce_allowlist``,
``NODE_TYPE_ALLOWLIST``, ``MODULE_ALLOWLIST``, ``GLOBAL_NAME_ALLOWLIST``.
"""

from __future__ import annotations

import ast

import pytest

from openbb_pine.compiler.codegen import (
    GLOBAL_NAME_ALLOWLIST,
    MODULE_ALLOWLIST,
    NODE_TYPE_ALLOWLIST,
    _collect_user_defined_names,
    _enforce_allowlist,
    _is_module_allowed,
)
from openbb_pine.errors import PineCodegenError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_from_source(source: str) -> ast.Module:
    """Parse Python source into an ast.Module — for tests that want a
    realistic mini-module rather than hand-construction."""
    return ast.parse(source)


def _fix(node: ast.Module) -> ast.Module:
    ast.fix_missing_locations(node)
    return node


# ---------------------------------------------------------------------------
# CG001 — disallowed ast node type
# ---------------------------------------------------------------------------


class TestCG001NodeType:
    """Every ast.* type NOT in NODE_TYPE_ALLOWLIST raises CG001."""

    def test_import_statement_rejected(self) -> None:
        """``import os`` produces ast.Import, which is NOT in the allowlist
        (only ast.ImportFrom is). Forces the explicit ``from X import Y``
        shape, which goes through MODULE_ALLOWLIST."""
        mod = _module_from_source("import os\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"
        assert "Import" in exc.value.node_kind

    def test_lambda_rejected(self) -> None:
        """``lambda x: x + 1`` produces ast.Lambda — anonymous functions
        could smuggle code through builtins like map/filter."""
        mod = _module_from_source("f = lambda x: x + 1\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"
        assert "Lambda" in exc.value.node_kind

    def test_class_def_rejected(self) -> None:
        """ClassDef would let a script smuggle metaclasses; not in
        Phase 1 anyway."""
        mod = _module_from_source("class Foo:\n    pass\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"
        assert "ClassDef" in exc.value.node_kind

    def test_try_rejected(self) -> None:
        """try/except gives a script the ability to suppress security
        errors silently. Not allowed."""
        mod = _module_from_source(
            "try:\n    x = 1\nexcept Exception:\n    pass\n"
        )
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"

    def test_with_rejected(self) -> None:
        """with-blocks would let a script open file context managers."""
        mod = _module_from_source(
            "with open('x') as f:\n    pass\n"
        )
        # The 'open' Name might be checked first via CG003; either way,
        # a CG### error should fire.
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule in {"CG001", "CG003"}

    def test_yield_rejected(self) -> None:
        """Generators / coroutines — not allowed in Phase 1."""
        mod = _module_from_source("def gen():\n    yield 1\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"

    def test_global_statement_rejected(self) -> None:
        mod = _module_from_source("def f():\n    global x\n    x = 1\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"

    def test_walrus_rejected(self) -> None:
        """walrus operator — no analog in Pine."""
        mod = _module_from_source("if (y := 1) > 0:\n    pass\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        # The walrus produces a Constant 0, comparison fine, but NamedExpr
        # is NOT in NODE_TYPE_ALLOWLIST → CG001.
        assert exc.value.rule == "CG001"

    def test_listcomp_rejected(self) -> None:
        """List comprehensions — Phase 2 (need their own scope rules)."""
        mod = _module_from_source("xs = [i for i in range(10)]\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG001"


# ---------------------------------------------------------------------------
# CG002 — disallowed ImportFrom module
# ---------------------------------------------------------------------------


class TestCG002Module:
    """Every ImportFrom whose module is NOT in MODULE_ALLOWLIST raises CG002."""

    def test_from_os_rejected(self) -> None:
        mod = _module_from_source("from os import getcwd\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"
        assert "os" in exc.value.node_kind

    def test_from_sys_rejected(self) -> None:
        mod = _module_from_source("from sys import exit\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_from_subprocess_rejected(self) -> None:
        mod = _module_from_source("from subprocess import Popen\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_from_socket_rejected(self) -> None:
        mod = _module_from_source("from socket import socket\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_from_pathlib_rejected(self) -> None:
        mod = _module_from_source("from pathlib import Path\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_from_builtins_rejected(self) -> None:
        mod = _module_from_source("from builtins import open\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_from_importlib_rejected(self) -> None:
        mod = _module_from_source("from importlib import import_module\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_relative_import_rejected(self) -> None:
        """``from . import x`` has module=None — explicitly rejected by
        :func:`_is_module_allowed`."""
        mod = ast.Module(
            body=[
                ast.ImportFrom(
                    module=None,
                    names=[ast.alias(name="x", asname=None)],
                    level=1,
                )
            ],
            type_ignores=[],
        )
        _fix(mod)
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG002"

    def test_pynecore_lib_allowed(self) -> None:
        """The happy-path import: ``from pynecore.lib import close, plot``."""
        mod = _module_from_source(
            "from pynecore.lib import close, plot\n"
            "x = close\n"
        )
        # Should NOT raise.
        _enforce_allowlist(mod)

    def test_pynecore_lib_ta_submodule_allowed(self) -> None:
        """A future emitter that uses ``from pynecore.lib.ta import sma``
        should pass without amending the allowlist."""
        mod = _module_from_source(
            "from pynecore.lib.ta import sma\n"
            "x = sma\n"
        )
        # Should NOT raise.
        _enforce_allowlist(mod)

    def test_pynecore_types_allowed(self) -> None:
        mod = _module_from_source(
            "from pynecore.types import Series\n"
            "x = Series\n"
        )
        _enforce_allowlist(mod)


# ---------------------------------------------------------------------------
# CG003 — disallowed global Name in Load context
# ---------------------------------------------------------------------------


class TestCG003GlobalName:
    """Every top-level Name(Load) not in GLOBAL_NAME_ALLOWLIST raises CG003,
    unless it was introduced by the user script (def args / Assign targets).
    """

    def test_import_dunder_rejected(self) -> None:
        """``__import__`` is not in GLOBAL_NAME_ALLOWLIST — Call would
        smuggle import-time arbitrary code execution."""
        mod = _module_from_source("x = __import__('os')\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"
        assert "__import__" in exc.value.node_kind

    def test_eval_rejected(self) -> None:
        mod = _module_from_source("x = eval('1+1')\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"
        assert "eval" in exc.value.node_kind

    def test_exec_rejected(self) -> None:
        mod = _module_from_source("exec('print(1)')\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_compile_rejected(self) -> None:
        mod = _module_from_source("x = compile('x', '<>', 'eval')\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_getattr_rejected(self) -> None:
        """getattr could bypass attribute restrictions."""
        mod = _module_from_source("x = getattr(close, 'foo')\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_globals_rejected(self) -> None:
        mod = _module_from_source("g = globals()\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_locals_rejected(self) -> None:
        mod = _module_from_source("l = locals()\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_breakpoint_rejected(self) -> None:
        mod = _module_from_source("breakpoint()\n")
        with pytest.raises(PineCodegenError) as exc:
            _enforce_allowlist(mod)
        assert exc.value.rule == "CG003"

    def test_close_allowed(self) -> None:
        """Pine OHLCV sources are in GLOBAL_NAME_ALLOWLIST."""
        mod = _module_from_source("x = close\n")
        _enforce_allowlist(mod)

    def test_ta_allowed(self) -> None:
        mod = _module_from_source("x = ta\n")
        _enforce_allowlist(mod)

    def test_user_defined_name_in_main_allowed(self) -> None:
        """A name introduced inside def main() is exempt from CG003."""
        src = (
            "def main():\n"
            "    length = 20\n"
            "    x = length\n"
        )
        mod = _module_from_source(src)
        _enforce_allowlist(mod)

    def test_function_param_exempt_from_cg003(self) -> None:
        """``def main(length=20)`` makes ``length`` a user-defined name."""
        src = (
            "def main(length=20):\n"
            "    x = length\n"
        )
        mod = _module_from_source(src)
        _enforce_allowlist(mod)

    def test_for_loop_var_exempt(self) -> None:
        """``for i in range(10): x = i`` — ``i`` is a user-defined name."""
        src = (
            "def main():\n"
            "    for i in range(10):\n"
            "        x = i\n"
        )
        mod = _module_from_source(src)
        _enforce_allowlist(mod)


# ---------------------------------------------------------------------------
# Defense-in-depth: emitter-produced code passes the gate
# ---------------------------------------------------------------------------


class TestEmitterOutputAlwaysPasses:
    """Every emitter output must pass the gate (else emit() would have
    raised). These tests cover a few representative end-to-end shapes
    to lock in the contract."""

    def test_sma_pipeline_passes(self) -> None:
        from openbb_pine.compiler import compile_pine

        compiled = compile_pine(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(ta.sma(close, 20))\n'
        )
        # If emit didn't raise, the gate passed. Re-parse + re-check
        # for belt-and-braces.
        mod = ast.parse(compiled.source)
        _enforce_allowlist(mod)

    def test_bollinger_pipeline_passes(self) -> None:
        from openbb_pine.compiler import compile_pine

        compiled = compile_pine(
            '//@version=6\n'
            'indicator("BB", overlay=true)\n'
            'length = input.int(20, minval=1)\n'
            'mult = input.float(2.0)\n'
            'basis = ta.sma(close, length)\n'
            'dev = mult * ta.stdev(close, length)\n'
            'plot(basis, title="basis")\n'
            'plot(basis + dev, title="upper")\n'
            'plot(basis - dev, title="lower")\n'
        )
        mod = ast.parse(compiled.source)
        _enforce_allowlist(mod)


# ---------------------------------------------------------------------------
# _is_module_allowed helper
# ---------------------------------------------------------------------------


class TestIsModuleAllowed:
    def test_pynecore_lib_top_level(self) -> None:
        assert _is_module_allowed("pynecore.lib") is True

    def test_pynecore_lib_submodule(self) -> None:
        assert _is_module_allowed("pynecore.lib.ta") is True

    def test_pynecore_lib_deeply_nested(self) -> None:
        assert _is_module_allowed("pynecore.lib.ta.helpers") is True

    def test_pynecore_types(self) -> None:
        assert _is_module_allowed("pynecore.types") is True

    def test_os_rejected(self) -> None:
        assert _is_module_allowed("os") is False

    def test_os_path_rejected(self) -> None:
        assert _is_module_allowed("os.path") is False

    def test_none_rejected(self) -> None:
        """Relative imports (level >= 1) have module=None."""
        assert _is_module_allowed(None) is False

    def test_substring_match_not_enough(self) -> None:
        """A module whose name SHARES a prefix with an allowed module but
        is not actually a submodule must be rejected. E.g. ``pynecore.libfoo``
        starts with ``pynecore.lib`` but is not ``pynecore.lib.*``."""
        assert _is_module_allowed("pynecore.libfoo") is False


# ---------------------------------------------------------------------------
# _collect_user_defined_names helper
# ---------------------------------------------------------------------------


class TestCollectUserDefinedNames:
    def test_def_name(self) -> None:
        mod = _module_from_source("def my_fn():\n    pass\n")
        names = _collect_user_defined_names(mod)
        assert "my_fn" in names

    def test_def_args(self) -> None:
        mod = _module_from_source("def f(a, b=1, *, c):\n    pass\n")
        names = _collect_user_defined_names(mod)
        assert {"a", "b", "c"} <= names

    def test_assignment_targets(self) -> None:
        mod = _module_from_source("def main():\n    x = 1\n    y, z = (2, 3)\n")
        names = _collect_user_defined_names(mod)
        assert {"x", "y", "z"} <= names

    def test_for_loop_var(self) -> None:
        mod = _module_from_source("def main():\n    for i in range(10):\n        pass\n")
        names = _collect_user_defined_names(mod)
        assert "i" in names

    def test_attribute_target_not_collected(self) -> None:
        """``obj.field = 1`` doesn't introduce a new top-level name."""
        mod = _module_from_source("def main():\n    close.x = 1\n")
        names = _collect_user_defined_names(mod)
        assert "x" not in names

    def test_subscript_target_not_collected(self) -> None:
        mod = _module_from_source("def main():\n    arr[0] = 1\n")
        names = _collect_user_defined_names(mod)
        # "arr" is a Subscript receiver in Load context inside the target,
        # but it wasn't introduced by the assign; we don't collect it.
        assert "arr" not in names


# ---------------------------------------------------------------------------
# Coverage of NODE_TYPE_ALLOWLIST surface area (selective)
# ---------------------------------------------------------------------------


class TestAllowlistedShapesPass:
    """Hand-built ast.Module instances using every category of allowed node
    must pass the gate. Catches accidental over-restriction (a future commit
    that removes a node type from the allowlist breaks this test loudly)."""

    def test_binop_chain_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = (1 + 2) * 3 - 4 / 5\n"
        )
        _enforce_allowlist(mod)

    def test_boolop_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = True and False or True\n"
        )
        _enforce_allowlist(mod)

    def test_compare_chain_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = 1 < 2\n    y = 1 == 2\n"
        )
        _enforce_allowlist(mod)

    def test_ifexp_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = 1 if True else 2\n"
        )
        _enforce_allowlist(mod)

    def test_subscript_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = close[1]\n"
        )
        _enforce_allowlist(mod)

    def test_attribute_chain_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = ta.sma\n"
        )
        _enforce_allowlist(mod)

    def test_for_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    for i in range(10):\n        x = i\n"
        )
        _enforce_allowlist(mod)

    def test_while_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    while True:\n        break\n"
        )
        _enforce_allowlist(mod)

    def test_if_elif_else_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n"
            "    if True:\n"
            "        x = 1\n"
            "    elif False:\n"
            "        x = 2\n"
            "    else:\n"
            "        x = 3\n"
        )
        _enforce_allowlist(mod)

    def test_unary_passes(self) -> None:
        mod = _module_from_source(
            "def main():\n    x = -close\n    y = not True\n"
        )
        _enforce_allowlist(mod)
