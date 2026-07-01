"""Tests for the bead 0e9.5.8 C8 error-model wiring.

Coverage groups:

1. **Structured inits** for the classes not covered by ``test_error_classes.py``:
   PineSyntaxError, PineInternalCompilerError, PineProviderError,
   PineFMPRequiredError, PineCacheError, PineExecTimeoutError,
   PineSecurityError, PineStrategyNotYetImplementedError.
2. **Diagnostic dataclass** — D1 §5.1 first-class per-diagnostic carrier.
3. **error_codes registry** — lookup + registration + AST-walking enforcement.
4. **telemetry counters** — ``pine_unsupported_builtin_total{name}`` +
   ``pine_unsupported_feature_total{name}`` stubs per D1 §3.5 / PRD §9.4.

Design sources of truth: D1 §5 (error model) + §3.5 (failure-mode telemetry).
Follows the post-R2/R6/Wave-3A/Wave-4 kw-only-with-backwards-compat pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# PineSyntaxError
# ---------------------------------------------------------------------------


class TestPineSyntaxErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineSyntaxError

        exc = PineSyntaxError(
            rule="PS001",
            source_line="length = input.int(=20)",
            line=7,
            col=18,
            hint="input.int's first arg is the default value.",
        )
        assert exc.rule == "PS001"
        assert exc.source_line == "length = input.int(=20)"
        assert exc.line == 7
        assert exc.col == 18
        assert exc.hint == "input.int's first arg is the default value."

    def test_default_str_renders_rule_line_col(self) -> None:
        from openbb_pine.errors import PineSyntaxError

        exc = PineSyntaxError(
            rule="PS002",
            source_line="a := ",
            line=3,
            col=5,
            hint="expression expected",
        )
        s = str(exc)
        assert "PS002" in s
        assert "line 3" in s
        assert "col 5" in s
        assert "expression expected" in s

    def test_backward_compat_positional_string(self) -> None:
        """The lexer + parser raise PineSyntaxError('msg') today; keep that
        working so this bead doesn't require lock-step edits across the
        compiler front-end."""
        from openbb_pine.errors import PineSyntaxError

        exc = PineSyntaxError("unexpected character '@' at line 4 col 2")
        assert "unexpected character" in str(exc)
        assert exc.rule is None

    def test_message_kwarg_overrides_rendering(self) -> None:
        from openbb_pine.errors import PineSyntaxError

        exc = PineSyntaxError(message="raw pre-stitched")
        assert str(exc) == "raw pre-stitched"
        assert exc.rule is None

    def test_no_args_yields_generic_text(self) -> None:
        from openbb_pine.errors import PineSyntaxError

        exc = PineSyntaxError()
        assert isinstance(str(exc), str) and len(str(exc)) > 0

    def test_class_code_attr_unchanged(self) -> None:
        from openbb_pine.errors import PineSyntaxError

        assert PineSyntaxError.code == "PineSyntaxError"


# ---------------------------------------------------------------------------
# PineInternalCompilerError
# ---------------------------------------------------------------------------


class TestPineInternalCompilerErrorStructuredInit:
    def test_class_exists_and_is_pinecompileerror(self) -> None:
        from openbb_pine.errors import PineCompileError, PineInternalCompilerError

        assert issubclass(PineInternalCompilerError, PineCompileError)

    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineInternalCompilerError

        exc = PineInternalCompilerError(
            rule="IC001",
            invariant="error code raised but not registered",
            node_kind="ir.VarDecl",
            hint="add the code to ERROR_CODES",
        )
        assert exc.rule == "IC001"
        assert exc.invariant == "error code raised but not registered"
        assert exc.node_kind == "ir.VarDecl"
        assert exc.hint == "add the code to ERROR_CODES"

    def test_default_str_renders_rule_and_invariant(self) -> None:
        from openbb_pine.errors import PineInternalCompilerError

        exc = PineInternalCompilerError(
            rule="IC002",
            invariant="unreachable branch in codegen",
        )
        s = str(exc)
        assert "IC002" in s
        assert "unreachable" in s

    def test_backward_compat_positional_string(self) -> None:
        from openbb_pine.errors import PineInternalCompilerError

        exc = PineInternalCompilerError("something impossible happened")
        assert "impossible" in str(exc)
        assert exc.rule is None


# ---------------------------------------------------------------------------
# PineProviderError
# ---------------------------------------------------------------------------


class TestPineProviderErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineProviderError

        exc = PineProviderError(
            requested="yfinance",
            supported=("fmp", "fmp_cached"),
            tracking_url="https://example/tracking",
        )
        assert exc.requested == "yfinance"
        assert exc.supported == ("fmp", "fmp_cached")
        assert exc.tracking_url == "https://example/tracking"

    def test_default_str_mentions_requested_and_supported(self) -> None:
        from openbb_pine.errors import PineProviderError

        exc = PineProviderError(
            requested="yfinance",
            supported=("fmp", "fmp_cached"),
        )
        s = str(exc)
        assert "yfinance" in s
        assert "fmp" in s

    def test_backward_compat_positional_string(self) -> None:
        """provider_selection.py raises ``PineProviderError('...')`` with a
        positional string today; keep it working."""
        from openbb_pine.errors import PineProviderError

        exc = PineProviderError("Only fmp and fmp_cached are supported.")
        assert "fmp" in str(exc)
        assert exc.requested is None


# ---------------------------------------------------------------------------
# PineFMPRequiredError
# ---------------------------------------------------------------------------


class TestPineFMPRequiredErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineFMPRequiredError

        exc = PineFMPRequiredError(
            builtin="request.security",
            mode="byo_only",
        )
        assert exc.builtin == "request.security"
        assert exc.mode == "byo_only"

    def test_default_str_mentions_builtin_and_mode(self) -> None:
        from openbb_pine.errors import PineFMPRequiredError

        exc = PineFMPRequiredError(
            builtin="request.dividends",
            mode="byo_only",
        )
        s = str(exc)
        assert "request.dividends" in s
        assert "byo_only" in s

    def test_backward_compat_positional_string(self) -> None:
        from openbb_pine.errors import PineFMPRequiredError

        exc = PineFMPRequiredError("FMP required for request.dividends")
        assert "FMP" in str(exc)
        assert exc.builtin is None


# ---------------------------------------------------------------------------
# PineCacheError — C6 sibling landed this API (sha/defect/path signature per
# commit 8ccbe1490). Verify it matches the shape we depend on.
# ---------------------------------------------------------------------------


class TestPineCacheErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from pathlib import Path

        from openbb_pine.errors import PineCacheError

        exc = PineCacheError(
            sha="abc123def456",
            defect="os.replace failed after tempfile write",
            path=Path("/tmp/pine_cache/xyz.pkl"),
        )
        assert exc.sha == "abc123def456"
        assert exc.defect == "os.replace failed after tempfile write"
        assert exc.path == Path("/tmp/pine_cache/xyz.pkl")

    def test_default_str_mentions_defect_and_path(self) -> None:
        from pathlib import Path

        from openbb_pine.errors import PineCacheError

        exc = PineCacheError(
            sha="abcd" * 16,
            defect="meta.json parse failed",
            path=Path("~/.openbb/pine_cache/foo.pkl"),
        )
        s = str(exc)
        assert "meta.json parse failed" in s
        assert "foo.pkl" in s

    def test_backward_compat_positional_string(self) -> None:
        from openbb_pine.errors import PineCacheError

        exc = PineCacheError("cache miss (defensive)")
        assert "cache miss" in str(exc)
        assert exc.sha is None


# ---------------------------------------------------------------------------
# PineExecTimeoutError
# ---------------------------------------------------------------------------


class TestPineExecTimeoutErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineExecTimeoutError

        exc = PineExecTimeoutError(
            timeout_s=5,
            elapsed_s=5.4,
            script_hash="abc123",
        )
        assert exc.timeout_s == 5
        assert exc.elapsed_s == 5.4
        assert exc.script_hash == "abc123"

    def test_default_str_mentions_timeout(self) -> None:
        from openbb_pine.errors import PineExecTimeoutError

        exc = PineExecTimeoutError(timeout_s=3, elapsed_s=3.1)
        s = str(exc)
        assert "3" in s
        assert "budget" in s or "timeout" in s or "exceeded" in s

    def test_backward_compat_positional_string(self) -> None:
        """runtime.limits raises ``PineExecTimeoutError('...')`` from inside
        a signal handler; keep it working."""
        from openbb_pine.errors import PineExecTimeoutError

        exc = PineExecTimeoutError("Pine script exceeded 5s wall-clock budget")
        assert "exceeded" in str(exc)
        assert exc.timeout_s is None


# ---------------------------------------------------------------------------
# PineSecurityError
# ---------------------------------------------------------------------------


class TestPineSecurityErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineSecurityError

        exc = PineSecurityError(
            rule="SEC001",
            node_kind="ImportFrom('os')",
        )
        assert exc.rule == "SEC001"
        assert exc.node_kind == "ImportFrom('os')"

    def test_default_str_mentions_rule_and_node(self) -> None:
        from openbb_pine.errors import PineSecurityError

        exc = PineSecurityError(
            rule="SEC001",
            node_kind="ImportFrom('subprocess')",
        )
        s = str(exc)
        assert "SEC001" in s
        assert "subprocess" in s

    def test_backward_compat_positional_string(self) -> None:
        from openbb_pine.errors import PineSecurityError

        exc = PineSecurityError(
            "Compiled module references forbidden modules: ['os']"
        )
        assert "forbidden" in str(exc)
        assert exc.rule is None


# ---------------------------------------------------------------------------
# PineStrategyNotYetImplementedError
# ---------------------------------------------------------------------------


class TestPineStrategyNotYetImplementedError:
    def test_backward_compat_positional_string(self) -> None:
        from openbb_pine.errors import PineStrategyNotYetImplementedError

        exc = PineStrategyNotYetImplementedError("Strategies land at M2 …")
        assert "M2" in str(exc)


# ---------------------------------------------------------------------------
# Diagnostic dataclass (D1 §5.1)
# ---------------------------------------------------------------------------


class TestDiagnostic:
    def test_construction_populates_fields(self) -> None:
        from openbb_pine.errors import Diagnostic

        d = Diagnostic(
            severity="error",
            code="PT001",
            message="expected simple<int>, got series<int>",
            location=("<inline>", 5, 12),
            hint="cast with math.round?",
            tracking_url="https://example/tracking",
        )
        assert d.severity == "error"
        assert d.code == "PT001"
        assert d.message == "expected simple<int>, got series<int>"
        assert d.location == ("<inline>", 5, 12)
        assert d.hint == "cast with math.round?"
        assert d.tracking_url == "https://example/tracking"

    def test_defaults(self) -> None:
        from openbb_pine.errors import Diagnostic

        d = Diagnostic(severity="warning", code="PU042", message="stubbed")
        assert d.location is None
        assert d.span is None
        assert d.hint is None
        assert d.tracking_url is None
        assert d.related == ()

    def test_frozen(self) -> None:
        """@dataclass(frozen=True) — reassignment must raise."""
        from openbb_pine.errors import Diagnostic

        d = Diagnostic(severity="error", code="PT001", message="…")
        with pytest.raises((AttributeError, TypeError)):
            d.code = "PT002"  # type: ignore[misc]

    def test_equality(self) -> None:
        from openbb_pine.errors import Diagnostic

        a = Diagnostic(severity="error", code="PT001", message="x")
        b = Diagnostic(severity="error", code="PT001", message="x")
        assert a == b

    def test_render_includes_severity_code_message(self) -> None:
        from openbb_pine.errors import Diagnostic

        d = Diagnostic(
            severity="error",
            code="PS012",
            message="unexpected token `=`",
            location=("<inline>", 7, 18),
            hint="try `input.int(20)`",
        )
        out = d.render()
        assert "error" in out.lower()
        assert "PS012" in out
        assert "unexpected token" in out
        assert "7" in out
        assert "18" in out
        assert "input.int(20)" in out

    def test_render_with_source_line_shows_caret(self) -> None:
        """When source is provided, render() should emit a Rust/Pine-style
        column pointer under the offending column."""
        from openbb_pine.errors import Diagnostic

        source = "length = input.int(=20)\n"
        d = Diagnostic(
            severity="error",
            code="PS012",
            message="unexpected token `=`",
            location=("<inline>", 1, 20),
        )
        out = d.render(source=source)
        # The rendered output should contain the source line …
        assert "input.int" in out
        # … and a caret line beneath it.
        assert "^" in out

    def test_render_without_location(self) -> None:
        """Diagnostics without a location still render (e.g. IC001)."""
        from openbb_pine.errors import Diagnostic

        d = Diagnostic(severity="error", code="IC001", message="no location")
        out = d.render()
        assert "IC001" in out
        assert "no location" in out


# ---------------------------------------------------------------------------
# error_codes registry (D1 §5.2)
# ---------------------------------------------------------------------------


class TestErrorCodesRegistry:
    def test_lookup_returns_spec_for_known_code(self) -> None:
        from openbb_pine.error_codes import lookup

        spec = lookup("PT001")
        assert spec is not None
        assert spec.code == "PT001"
        assert spec.class_name == "PineTypeError"
        assert spec.short_description  # non-empty
        assert spec.since_version  # non-empty

    def test_lookup_returns_none_for_unknown_code(self) -> None:
        from openbb_pine.error_codes import lookup

        assert lookup("ZZZ999") is None

    def test_all_type_checker_codes_registered(self) -> None:
        """PT001-PT008 per D1 §4.4 reserved range. PT007 may not fire yet
        (v6 type-arg checking is scheduled) but the code MUST be reserved."""
        from openbb_pine.error_codes import lookup

        for i in range(1, 9):
            code = f"PT{i:03d}"
            assert lookup(code) is not None, f"{code} not registered"

    def test_all_codegen_codes_registered(self) -> None:
        """CG001-CG006 — the three allowlist rules per D1 §3.2 + the three
        defensive raises the C5 codegen adds (CG004-CG006)."""
        from openbb_pine.error_codes import lookup

        for i in range(1, 7):
            code = f"CG{i:03d}"
            assert lookup(code) is not None, f"{code} not registered"

    def test_pf_feature_codes_registered(self) -> None:
        from openbb_pine.error_codes import lookup

        for code in ("PF001", "PF002", "PF003", "PF010", "PF011"):
            assert lookup(code) is not None, f"{code} not registered"

    def test_assert_code_registered_raises_ic001_on_miss(self) -> None:
        from openbb_pine.errors import PineInternalCompilerError
        from openbb_pine.error_codes import assert_code_registered

        with pytest.raises(PineInternalCompilerError) as excinfo:
            assert_code_registered("ZZZ999")
        assert "IC001" in str(excinfo.value)
        assert "ZZZ999" in str(excinfo.value)

    def test_assert_code_registered_passes_for_known(self) -> None:
        from openbb_pine.error_codes import assert_code_registered

        # Should not raise.
        assert_code_registered("PT001")
        assert_code_registered("CG001")

    def test_error_code_spec_is_frozen(self) -> None:
        from openbb_pine.error_codes import ErrorCodeSpec

        spec = ErrorCodeSpec(
            code="XX999",
            class_name="X",
            short_description="s",
            detailed_description="d",
            since_version="0.1.0",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.code = "YY000"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AST-walking enforcement test — the raison-d'être for the registry.
# ---------------------------------------------------------------------------


def _iter_pine_py_files() -> list[Path]:
    """Yield every .py under openbb_pine/ EXCEPT the tests dir + error_codes.py
    itself + errors.py (docstring examples aren't real raises)."""
    from openbb_pine import __file__ as _init

    pkg = Path(_init).parent
    files: list[Path] = []
    for p in pkg.rglob("*.py"):
        parts = set(p.parts)
        if "tests" in parts or "__pycache__" in parts:
            continue
        if p.name in ("error_codes.py", "errors.py"):
            continue
        files.append(p)
    return files


def _extract_code_kwargs_from_source(src: str) -> list[tuple[str, int]]:
    """Return every ``rule=<literal-str>`` / ``code=<literal-str>`` kwarg
    passed to a ``raise Pine*Error(...)`` call, as (code, lineno) pairs."""
    try:
        tree = ast.parse(src)
    except SyntaxError:  # pragma: no cover
        return []
    finds: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        call = node.exc
        # Unwrap ``raise Foo(...) from err`` — exc is the Call node.
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        # We only care about calls whose Name / Attribute ends in ``Error``.
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if not name or not name.startswith("Pine") or not name.endswith("Error"):
            continue
        for kw in call.keywords:
            if kw.arg not in ("rule", "code"):
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                finds.append((kw.value.value, node.lineno))
    return finds


class TestErrorCodeEnforcement:
    """Walks every ``raise Pine*Error(rule=...)`` in the codebase and verifies
    the literal code is registered. Catches typos permanently.

    Rationale: without this, someone can write ``rule="CG002"`` (typo for
    ``CG003``) and it looks perfectly fine to lint. This test converts that
    class of bug into an immediate CI failure.
    """

    def test_every_raised_code_is_registered(self) -> None:
        from openbb_pine.error_codes import ERROR_CODES

        missing: list[tuple[Path, str, int]] = []
        for path in _iter_pine_py_files():
            for code, lineno in _extract_code_kwargs_from_source(
                path.read_text(encoding="utf-8")
            ):
                if code not in ERROR_CODES:
                    missing.append((path, code, lineno))
        assert not missing, (
            "unregistered error codes raised in the codebase (register them in "
            "openbb_pine/error_codes.py):\n"
            + "\n".join(f"  {p}:{ln}: {code!r}" for p, code, ln in missing)
        )


# ---------------------------------------------------------------------------
# Telemetry counters (D1 §3.5 + PRD §9.4)
# ---------------------------------------------------------------------------


class TestTelemetryCounters:
    def setup_method(self) -> None:
        """Isolate every test case from residual counter state."""
        from openbb_pine.telemetry import reset_metrics

        reset_metrics()

    def test_record_unsupported_builtin_increments(self) -> None:
        from openbb_pine.telemetry import (
            get_unsupported_builtin_counts,
            record_unsupported_builtin,
        )

        record_unsupported_builtin("ta.ichimoku")
        record_unsupported_builtin("ta.ichimoku")
        record_unsupported_builtin("ta.tsi")
        counts = get_unsupported_builtin_counts()
        assert counts["ta.ichimoku"] == 2
        assert counts["ta.tsi"] == 1

    def test_record_unsupported_feature_increments(self) -> None:
        from openbb_pine.telemetry import (
            get_unsupported_feature_counts,
            record_unsupported_feature,
        )

        record_unsupported_feature("PF010")
        counts = get_unsupported_feature_counts()
        assert counts["PF010"] == 1

    def test_reset_metrics_clears_both(self) -> None:
        from openbb_pine.telemetry import (
            get_unsupported_builtin_counts,
            get_unsupported_feature_counts,
            record_unsupported_builtin,
            record_unsupported_feature,
            reset_metrics,
        )

        record_unsupported_builtin("ta.foo")
        record_unsupported_feature("PF999")
        reset_metrics()
        assert get_unsupported_builtin_counts() == {}
        assert get_unsupported_feature_counts() == {}

    def test_get_returns_copy_not_live_reference(self) -> None:
        """Mutating the returned dict must NOT bleed into the counter state."""
        from openbb_pine.telemetry import (
            get_unsupported_builtin_counts,
            record_unsupported_builtin,
        )

        record_unsupported_builtin("ta.x")
        snapshot = get_unsupported_builtin_counts()
        snapshot["ta.injected"] = 999
        assert "ta.injected" not in get_unsupported_builtin_counts()


# ---------------------------------------------------------------------------
# Integration: unsupported-builtin raise sites increment the counter.
# ---------------------------------------------------------------------------


class TestTelemetryIntegration:
    def setup_method(self) -> None:
        from openbb_pine.telemetry import reset_metrics

        reset_metrics()

    def test_type_checker_raise_records_builtin(self) -> None:
        """The C3 type checker's ``_raise_unsupported_builtin`` MUST call
        ``record_unsupported_builtin(name)`` before raising, so PRD §3.4 L0.5
        wild-corpus coverage attribution stays accurate."""
        from openbb_pine.compiler.type_checker import _TypeChecker
        from openbb_pine.errors import PineUnsupportedBuiltinError
        from openbb_pine.telemetry import get_unsupported_builtin_counts

        tc = _TypeChecker(pine_version=6)
        with pytest.raises(PineUnsupportedBuiltinError):
            tc._raise_unsupported_builtin(
                "ta.ichimoku",
                node=_make_dummy_ir_node(),
            )
        counts = get_unsupported_builtin_counts()
        assert counts.get("ta.ichimoku", 0) >= 1

    def test_codegen_pf010_records_feature(self) -> None:
        """visit_Program's PF010 strategy-deferral raise MUST call
        ``record_unsupported_feature('PF010')`` before raising."""
        from openbb_pine.compiler import ir
        from openbb_pine.compiler.codegen import _CodegenVisitor
        from openbb_pine.errors import PineUnsupportedFeatureError
        from openbb_pine.telemetry import get_unsupported_feature_counts

        span = _make_dummy_span()
        directive = ir.ScriptDirective(
            loc=span,
            kind="strategy",
            title="x",
            shorttitle=None,
            overlay=None,
            arguments=(),
        )
        prog = ir.Program(
            loc=span,
            version=6,
            directive=directive,
            declarations=(),
            body=(),
        )
        with pytest.raises(PineUnsupportedFeatureError):
            _CodegenVisitor(builtins_used=frozenset(), pine_version=6).visit_Program(prog)
        counts = get_unsupported_feature_counts()
        assert counts.get("PF010", 0) >= 1


def _make_dummy_span():
    from openbb_pine.compiler import ir

    return ir.Span(
        file="<inline>",
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=1,
        start_byte=0,
        end_byte=0,
    )


def _make_dummy_ir_node():
    from openbb_pine.compiler import ir

    return ir.Name(id="dummy", loc=_make_dummy_span())
