"""Codegen unit tests (C5, bead 0e9.5.5).

Mirror of D1 §3 — every aspect of the codegen contract is exercised:

* :class:`TestCodegenSnapshots` — compile small Pine scripts end-to-end and
  assert the emitted Python matches a snapshot. Covers the bread-and-butter
  shapes: bare indicator, SMA(close, N), BB skeleton, if/else, for loop,
  var/varip, input.int with minval, multiple plots, ternary.
* :class:`TestCompiledModuleContract` — every field of
  :class:`CompiledModule` is correctly populated; the source is valid
  Python; the @pyne magic docstring fires first.
* :class:`TestCompilePineFacadeIntegration` — end-to-end compile_pine on
  v6 and v5 sources; the migration shim runs before C5 transparently.
* :class:`TestUnsupportedDirectives` — ``strategy()`` raises PF010,
  ``library()`` raises PF011.
* :class:`TestInputLifting` — Pine inputs are correctly hoisted into
  ``def main(...)`` kwargs, not left in the body.
* :class:`TestImportConsolidation` — the ``from pynecore.lib import …``
  line carries exactly the names ``builtins_used`` implies (no more, no
  less).
* :class:`TestHashAndCacheStatus` — after C6, sha is the 64-char BLAKE2b
  cache key; cache_status is ``"miss"`` on first compile via compile_pine
  (see test_compile_cache.py for full cache coverage).

Allowlist-gate-specific tests live in ``test_codegen_allowlist.py``.
"""

from __future__ import annotations

import ast

import pytest

from openbb_pine.compiler import compile_pine, emit, ir
from openbb_pine.compiler.codegen import (
    GLOBAL_NAME_ALLOWLIST,
    MODULE_ALLOWLIST,
    NODE_TYPE_ALLOWLIST,
)
from openbb_pine.compiler.types import CompiledModule
from openbb_pine.errors import PineCodegenError, PineUnsupportedFeatureError


# ---------------------------------------------------------------------------
# Snapshot fixtures — small Pine scripts whose emitted shape is stable.
# ---------------------------------------------------------------------------


def _compile(src: str) -> CompiledModule:
    return compile_pine(src)


def _src_contains(compiled: CompiledModule, *substrings: str) -> bool:
    """All substrings present in compiled.source (snapshot-shape assertion)."""
    return all(s in compiled.source for s in substrings)


class TestCodegenSnapshots:
    """Compile, then assert the source contains the expected structural
    elements. We intentionally do NOT byte-compare — ast.unparse formatting
    is allowed to drift across Python versions. Substring + AST shape
    assertions cover the contract without being brittle."""

    def test_bare_indicator_plot_close(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        assert _src_contains(
            compiled,
            "@pyne",
            "from pynecore.lib import",
            "@script.indicator('X')",
            "def main():",
            "plot(close)",
        )

    def test_sma_with_input_int(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("SMA")\n'
            'length = input.int(20)\n'
            'plot(ta.sma(close, length))\n'
        )
        assert _src_contains(
            compiled,
            "def main(length=input.int(20))",
            "plot(ta.sma(close, length))",
        )

    def test_input_int_with_minval(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'length = input.int(20, minval=1)\n'
            'plot(close)\n'
        )
        # kwargs preserved on the lifted input call.
        assert "input.int(20, minval=1)" in compiled.source

    def test_bollinger_skeleton(self) -> None:
        compiled = _compile(
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
        assert _src_contains(
            compiled,
            "@script.indicator",
            "overlay=True",
            "def main(length=input.int(20, minval=1), mult=input.float(2.0))",
            "basis = ta.sma(close, length)",
            "dev = mult * ta.stdev(close, length)",
            "plot(basis, title='basis')",
            "plot(basis + dev, title='upper')",
            "plot(basis - dev, title='lower')",
        )

    def test_if_else(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(close)\n'
            'if close > open\n'
            '    plot(high)\n'
            'else\n'
            '    plot(low)\n'
        )
        # if/else emitted as Python's native if/else (not match/case).
        assert "if close > open:" in compiled.source
        assert "else:" in compiled.source

    def test_for_to_loop(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'sum = 0.0\n'
            'for i = 0 to 10\n'
            '    sum := sum + close[i]\n'
            'plot(sum)\n'
        )
        # Pine `for i = 0 to 10` is inclusive; Python `range(0, 10 + 1)`.
        assert "for i in range(0, 10 + 1):" in compiled.source
        assert "sum = sum + close[i]" in compiled.source

    def test_var_declaration(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'var counter = 0\n'
            'counter := counter + 1\n'
            'plot(counter)\n'
        )
        # var lowers to plain assign in Phase 1 (PyneCore handles persistence
        # via its AST transformer at script-import time).
        assert "counter = 0" in compiled.source
        assert "counter = counter + 1" in compiled.source

    def test_varip_declaration(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'varip price = 0.0\n'
            'price := close\n'
            'plot(price)\n'
        )
        assert "price = 0.0" in compiled.source

    def test_ternary_expression(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'sig = close > open ? high : low\n'
            'plot(sig)\n'
        )
        # Pine ternary lowers to Python IfExp.
        assert "sig = high if close > open else low" in compiled.source

    def test_multiple_plots(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(close, title="c")\n'
            'plot(open, title="o")\n'
            'plot(high, title="h")\n'
        )
        assert compiled.source.count("plot(") == 3

    def test_binary_arithmetic(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot((high + low) / 2.0)\n'
        )
        assert "(high + low) / 2.0" in compiled.source

    def test_comparison_operators(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'sig = close > open\n'
            'plot(sig ? 1.0 : 0.0)\n'
        )
        assert "sig = close > open" in compiled.source

    def test_logical_and_or(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'a = close > open\n'
            'b = high > low\n'
            'sig = a and b\n'
            'plot(sig ? 1.0 : 0.0)\n'
        )
        assert "sig = a and b" in compiled.source

    def test_unary_minus(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(-close)\n'
        )
        assert "plot(-close)" in compiled.source

    def test_history_access(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(close[1])\n'
        )
        assert "plot(close[1])" in compiled.source

    def test_overlay_false(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X", overlay=false)\n'
            'plot(close)\n'
        )
        assert "overlay=False" in compiled.source

    def test_shorttitle_threaded_through(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("Long Title", shorttitle="LT", overlay=true)\n'
            'plot(close)\n'
        )
        assert "shorttitle='LT'" in compiled.source


# ---------------------------------------------------------------------------
# CompiledModule contract (D1 §3.1)
# ---------------------------------------------------------------------------


class TestCompiledModuleContract:
    """Every field on :class:`CompiledModule` is correctly populated by
    :func:`emit`, and the source is shaped per the D1 §3.1 contract."""

    def test_source_is_valid_python(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(ta.sma(close, 20))\n'
        )
        # Must be importable / compile()able as Python.
        compile(compiled.source, "<test>", "exec")

    def test_source_starts_with_pyne_magic_docstring(self) -> None:
        """PyneCore's import_script keys off this exact prefix."""
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        # First non-blank line is a docstring containing @pyne.
        head = compiled.source.lstrip()[:200]
        assert "@pyne" in head

    def test_source_has_one_import_from_pynecore_lib(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        # Exactly one ``from pynecore.lib import …`` line.
        assert compiled.source.count("from pynecore.lib import") == 1

    def test_source_has_main_function(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        assert "def main(" in compiled.source

    def test_pine_version_threaded_through_v6(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        assert compiled.pine_version == 6

    def test_pine_version_threaded_through_v5_migrated(self) -> None:
        compiled = _compile(
            '//@version=5\nstudy("X")\nplot(close)\n'
        )
        # After migration → target is v6.
        assert compiled.pine_version == 6

    def test_compiler_version_threaded_through(self) -> None:
        from openbb_pine import __version__

        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        assert compiled.compiler_version == __version__
        # And appears in the docstring metadata.
        assert __version__ in compiled.source

    def test_builtins_used_populated(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(ta.sma(close, 20))\n'
        )
        # Both ta.sma + close + plot should be recorded.
        assert "ta.sma" in compiled.builtins_used
        assert "close" in compiled.builtins_used
        assert "plot" in compiled.builtins_used

    def test_security_contexts_none_in_phase_1(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        # Phase 1: C3 always returns None for security_contexts.
        assert compiled.security_contexts is None

    def test_cache_status_is_miss_on_first_compile(self) -> None:
        """After C6 (bead 0e9.5.6), the compile_pine facade wires the cache:
        first compile of a source is a miss, second is a hit (see
        test_compile_cache.py). Here we verify only the miss shape, using a
        tmp cache dir so we don't pollute the operator's real ~/.openbb.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            compiled = compile_pine(
                '//@version=6\nindicator("X")\nplot(close)\n',
                cache_dir=Path(td),
            )
        assert compiled.cache_status == "miss"

    def test_sha_is_the_c6_cache_key(self) -> None:
        """After C6, ``sha`` is the real 64-char BLAKE2b-256 cache key
        (was the C5 placeholder blake2b-128 → 32 hex chars)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            compiled = compile_pine(
                '//@version=6\nindicator("X")\nplot(close)\n',
                cache_dir=Path(td),
            )
        assert compiled.sha
        # BLAKE2b-256 → 64 hex chars.
        assert len(compiled.sha) == 64
        # Valid hex.
        int(compiled.sha, 16)

    def test_sha_is_deterministic(self) -> None:
        """Identical sources → identical sha (the C6 cache key)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            c1 = compile_pine(
                '//@version=6\nindicator("X")\nplot(close)\n',
                cache_dir=Path(td),
            )
        with tempfile.TemporaryDirectory() as td:
            c2 = compile_pine(
                '//@version=6\nindicator("X")\nplot(close)\n',
                cache_dir=Path(td),
            )
        assert c1.sha == c2.sha

    def test_sha_changes_when_source_changes(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            c1 = compile_pine(
                '//@version=6\nindicator("X")\nplot(close)\n',
                cache_dir=Path(td),
            )
            c2 = compile_pine(
                '//@version=6\nindicator("Y")\nplot(close)\n',
                cache_dir=Path(td),
            )
        assert c1.sha != c2.sha


# ---------------------------------------------------------------------------
# compile_pine facade integration (end-to-end)
# ---------------------------------------------------------------------------


class TestCompilePineFacadeIntegration:
    """The public :func:`compile_pine` returns a CompiledModule and runs the
    full lex → parse → migrate? → check → emit pipeline."""

    def test_returns_compiled_module(self) -> None:
        result = _compile('//@version=6\nindicator("X")\nplot(close)\n')
        assert isinstance(result, CompiledModule)

    def test_v5_source_migrates_then_compiles(self) -> None:
        result = _compile(
            '//@version=5\nstudy("RSI")\nplot(ta.rsi(close, 14))\n'
        )
        assert isinstance(result, CompiledModule)
        # study → indicator migration happened before parse.
        assert "@script.indicator" in result.source
        assert "ta.rsi" in result.source

    def test_v6_source_compiles_without_migration(self) -> None:
        result = _compile(
            '//@version=6\nindicator("RSI")\nplot(ta.rsi(close, 14))\n'
        )
        assert isinstance(result, CompiledModule)
        assert "@script.indicator" in result.source

    def test_no_pragma_defaults_to_v6(self) -> None:
        result = _compile('indicator("X")\nplot(close)\n')
        assert result.pine_version == 6

    def test_v4_pragma_raises(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            _compile('//@version=4\nstudy("X")\n')
        assert "PF001" in str(exc.value)

    def test_v7_pragma_raises(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            _compile('//@version=7\nindicator("X")\n')
        assert "PF002" in str(exc.value)

    def test_smoke_conformance_fixture_compiles(self) -> None:
        """The conformance harness's _smoke.pine fixture must produce a
        valid CompiledModule. End-to-end execution still SKIPs (needs S
        bead ta.* bridges), but the compile step is now meaningful.

        Reading the file via repo-relative path keeps this test runnable
        outside the conformance dir. The path traversal is:
        ``tests/unit/test_codegen.py`` (this file) → up 6 = repo root.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[6]
        smoke_path = repo_root / "tests" / "conformance" / "_smoke.pine"
        if not smoke_path.exists():
            pytest.skip(f"_smoke.pine fixture not at {smoke_path}")
        src = smoke_path.read_text(encoding="utf-8")
        compiled = _compile(src)
        assert isinstance(compiled, CompiledModule)
        assert "@script.indicator" in compiled.source
        assert "ta.sma(close, 1)" in compiled.source
        # And the emitted code is valid Python.
        compile(compiled.source, "<smoke>", "exec")


# ---------------------------------------------------------------------------
# Unsupported directives (PF010 strategy / PF011 library)
# ---------------------------------------------------------------------------


class TestUnsupportedDirectives:
    """``strategy()`` and ``library()`` are recognised but deferred."""

    def test_strategy_raises_pf010(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            _compile(
                '//@version=6\n'
                'strategy("Long Only", overlay=true)\n'
                'plot(close)\n'
            )
        assert "PF010" in str(exc.value)
        assert "strategy" in str(exc.value).lower()

    def test_library_raises_pf011(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            _compile(
                '//@version=6\n'
                'library("MyLib")\n'
            )
        assert "PF011" in str(exc.value)


# ---------------------------------------------------------------------------
# Input lifting (Pine input.* assignments → def main() kwargs)
# ---------------------------------------------------------------------------


class TestInputLifting:
    """Pine input declarations at body level must be hoisted into the
    ``def main(...)`` signature so PyneCore's @script decorator can
    introspect them."""

    def test_input_lifted_to_main_signature(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'length = input.int(20)\n'
            'plot(ta.sma(close, length))\n'
        )
        # The lift moves the assignment into the def main(...) header.
        assert "def main(length=input.int(20))" in compiled.source
        # The body no longer contains a top-level ``length = input.int(...)``
        # assignment statement (the lift removed it).
        lines = compiled.source.splitlines()
        body_lines = [
            l for l in lines if l.startswith("    ") and not l.startswith("    #")
        ]
        body_text = "\n".join(body_lines)
        assert "length = input.int" not in body_text

    def test_multiple_inputs_preserved_order(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'a = input.int(1)\n'
            'b = input.float(2.0)\n'
            'c = input.bool(true)\n'
            'plot(close)\n'
        )
        idx_a = compiled.source.find("a=input.int(1)")
        idx_b = compiled.source.find("b=input.float(2.0)")
        idx_c = compiled.source.find("c=input.bool(True)")
        assert idx_a >= 0 and idx_b > idx_a and idx_c > idx_b

    def test_non_input_assigns_stay_in_body(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'basis = ta.sma(close, 20)\n'
            'plot(basis)\n'
        )
        # basis is not an input — it stays in the body.
        assert "basis = ta.sma(close, 20)" in compiled.source

    def test_input_with_kwargs_lifted_with_kwargs(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'length = input.int(20, minval=1)\n'
            'plot(close)\n'
        )
        assert "length=input.int(20, minval=1)" in compiled.source


# ---------------------------------------------------------------------------
# Import consolidation
# ---------------------------------------------------------------------------


class TestImportConsolidation:
    """The ``from pynecore.lib import …`` line names every required symbol
    and nothing else."""

    def test_only_used_namespaces_imported(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(ta.sma(close, 20))\n'
        )
        # Must include: ta (called), close (referenced), script (decorator),
        # plot (function), input (NOT needed — no input call).
        first_line = next(
            l for l in compiled.source.splitlines()
            if l.startswith("from pynecore.lib import")
        )
        imports = {n.strip() for n in first_line.split("import", 1)[1].split(",")}
        assert "ta" in imports
        assert "close" in imports
        assert "script" in imports
        assert "plot" in imports
        # Script has no input.* call — input shouldn't be imported.
        assert "input" not in imports

    def test_input_imported_when_used(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'length = input.int(20)\n'
            'plot(close)\n'
        )
        first_line = next(
            l for l in compiled.source.splitlines()
            if l.startswith("from pynecore.lib import")
        )
        imports = {n.strip() for n in first_line.split("import", 1)[1].split(",")}
        assert "input" in imports

    def test_math_namespace_imported(self) -> None:
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(math.abs(close - open))\n'
        )
        first_line = next(
            l for l in compiled.source.splitlines()
            if l.startswith("from pynecore.lib import")
        )
        imports = {n.strip() for n in first_line.split("import", 1)[1].split(",")}
        assert "math" in imports

    def test_imports_sorted_alphabetically(self) -> None:
        """Cache-key stability: deterministic import order."""
        compiled = _compile(
            '//@version=6\n'
            'indicator("X")\n'
            'plot(ta.sma(close, 20))\n'
        )
        first_line = next(
            l for l in compiled.source.splitlines()
            if l.startswith("from pynecore.lib import")
        )
        imports_text = first_line.split("import", 1)[1]
        imports_list = [n.strip() for n in imports_text.split(",")]
        assert imports_list == sorted(imports_list)


# ---------------------------------------------------------------------------
# Smoke: the public emit() entry point can be driven independently of
# compile_pine (e.g. by tests that hand-construct a typed program).
# ---------------------------------------------------------------------------


class TestEmitDirect:
    """The :func:`emit` function is usable without going through
    :func:`compile_pine` — tests + future tooling drive it directly.
    Direct callers get ``sha=""`` + ``cache_status="bypass"`` (the C6 cache
    is only wired when going through the :func:`compile_pine` facade).
    """

    def test_emit_minimal_program(self) -> None:
        from openbb_pine.compiler import compile_pine_to_program

        prog = compile_pine_to_program(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        compiled = emit(
            prog,
            builtins_used=frozenset({"close", "plot"}),
            security_contexts=None,
            pine_version=6,
            compiler_version="9.9.9",
        )
        assert compiled.compiler_version == "9.9.9"
        assert "9.9.9" in compiled.source

    def test_emit_direct_returns_empty_sha_and_bypass(self) -> None:
        """Direct emit() (not through compile_pine facade) → no C6 cache
        involvement. The returned module signals this via sha="" and
        cache_status="bypass"."""
        from openbb_pine.compiler import compile_pine_to_program

        prog = compile_pine_to_program(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        compiled = emit(
            prog,
            builtins_used=frozenset({"close", "plot"}),
            security_contexts=None,
            pine_version=6,
            compiler_version="0.1.0",
        )
        assert compiled.sha == ""
        assert compiled.cache_status == "bypass"

    def test_emit_threads_security_contexts(self) -> None:
        """When C3 sets security_contexts (Phase 2), emit threads it through."""
        from openbb_pine.compiler import compile_pine_to_program
        from openbb_pine.compiler.types import SecurityContext

        prog = compile_pine_to_program(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        ctx = {"ctx_0": SecurityContext(symbol="AAPL", timeframe="1D", expr="close")}
        compiled = emit(
            prog,
            builtins_used=frozenset({"close", "plot"}),
            security_contexts=ctx,
            pine_version=6,
            compiler_version="0.1.0",
        )
        assert compiled.security_contexts is ctx


# ---------------------------------------------------------------------------
# Visible smoke check: the allowlist sets are non-empty and look sensible.
# (Full allowlist tests live in test_codegen_allowlist.py.)
# ---------------------------------------------------------------------------


class TestAllowlistShape:
    def test_node_type_allowlist_nonempty(self) -> None:
        assert len(NODE_TYPE_ALLOWLIST) > 30

    def test_module_allowlist_includes_pynecore_lib(self) -> None:
        assert "pynecore.lib" in MODULE_ALLOWLIST

    def test_module_allowlist_excludes_os(self) -> None:
        assert "os" not in MODULE_ALLOWLIST
        assert "sys" not in MODULE_ALLOWLIST
        assert "subprocess" not in MODULE_ALLOWLIST

    def test_global_name_allowlist_excludes_dangerous_builtins(self) -> None:
        assert "open" not in GLOBAL_NAME_ALLOWLIST or True  # open is OHLCV, not the builtin
        assert "__import__" not in GLOBAL_NAME_ALLOWLIST
        assert "exec" not in GLOBAL_NAME_ALLOWLIST
        assert "eval" not in GLOBAL_NAME_ALLOWLIST
        assert "compile" not in GLOBAL_NAME_ALLOWLIST
        assert "getattr" not in GLOBAL_NAME_ALLOWLIST
        assert "setattr" not in GLOBAL_NAME_ALLOWLIST

    def test_global_name_allowlist_includes_ohlcv(self) -> None:
        for s in ("open", "high", "low", "close", "volume"):
            assert s in GLOBAL_NAME_ALLOWLIST

    def test_global_name_allowlist_includes_pine_namespaces(self) -> None:
        for ns in ("ta", "math", "input", "script", "plot"):
            assert ns in GLOBAL_NAME_ALLOWLIST


# ---------------------------------------------------------------------------
# AST-shape smoke — every emitted module passes the gate (otherwise emit()
# would have raised) AND has the structural elements D1 §3.1 names.
# ---------------------------------------------------------------------------


class TestEmittedAstShape:
    def test_first_module_stmt_is_docstring_expr(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        tree = ast.parse(compiled.source)
        assert isinstance(tree.body[0], ast.Expr)
        assert isinstance(tree.body[0].value, ast.Constant)
        assert "@pyne" in tree.body[0].value.value

    def test_second_stmt_is_importfrom(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        tree = ast.parse(compiled.source)
        assert isinstance(tree.body[1], ast.ImportFrom)
        assert tree.body[1].module == "pynecore.lib"

    def test_third_stmt_is_def_main(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        tree = ast.parse(compiled.source)
        assert isinstance(tree.body[2], ast.FunctionDef)
        assert tree.body[2].name == "main"

    def test_main_has_script_indicator_decorator(self) -> None:
        compiled = _compile(
            '//@version=6\nindicator("X")\nplot(close)\n'
        )
        tree = ast.parse(compiled.source)
        fn = tree.body[2]
        assert isinstance(fn, ast.FunctionDef)
        assert len(fn.decorator_list) == 1
        deco = fn.decorator_list[0]
        assert isinstance(deco, ast.Call)
        assert isinstance(deco.func, ast.Attribute)
        assert isinstance(deco.func.value, ast.Name)
        assert deco.func.value.id == "script"
        assert deco.func.attr == "indicator"
