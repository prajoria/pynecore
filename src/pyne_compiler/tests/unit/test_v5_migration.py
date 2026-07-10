"""Tests for ``openbb_pine.compiler.v5_migration`` — v5→v6 auto-migration shim.

Source of truth: D1 §1.5 (per-version grammar dispatch), PRD §3.2 Phase 1
+ PRD §8.1 M1 gate (h) ("Pine v5 script runs unedited via the v5→v6
migration shim").

The shim's contract:

* :func:`detect_pine_version` reads ``//@version=N`` and returns 5/6 or
  raises ``PineUnsupportedFeatureError`` for out-of-range versions.
* :func:`migrate_v5_to_v6` applies every :data:`V5_REWRITES` entry in
  source order, rewrites the pragma, and returns the new source plus an
  audit log. Every rewrite is ``bidi_safe=True`` — applying twice
  produces the same output.
* :func:`compile_pine` (in ``compiler/__init__.py``) is the high-level
  facade: detects version, migrates if v5→v6, then tokenises and parses.

Per the bead brief: ~25-40 tests covering each rewrite, the idempotence
invariant, end-to-end migration, the compile_pine integration, and the
PF001/PF002/PF003 error envelope.
"""

from __future__ import annotations

import re

import pytest

from openbb_pine.compiler import compile_pine, compile_pine_to_program, ir
from openbb_pine.compiler.v5_migration import (
    V5Rewrite,
    V5_REWRITES,
    detect_pine_version,
    migrate_v5_to_v6,
)
from openbb_pine.errors import PineUnsupportedFeatureError


# ---------------------------------------------------------------------------
# detect_pine_version
# ---------------------------------------------------------------------------


class TestDetectPineVersion:
    """Pragma routing — PF001 / PF002 / default-to-6 (D1 §1.2, PRD §3.3)."""

    def test_detects_v5(self) -> None:
        assert detect_pine_version("//@version=5\nindicator(\"X\")\n") == 5

    def test_detects_v6(self) -> None:
        assert detect_pine_version("//@version=6\nindicator(\"X\")\n") == 6

    def test_no_pragma_defaults_to_6(self) -> None:
        """TradingView's editor auto-inserts the pragma; unprefixed source
        is more likely a paste-of-a-snippet than legacy v4."""
        assert detect_pine_version("indicator(\"X\")\nx = 1\n") == 6

    def test_empty_source_defaults_to_6(self) -> None:
        assert detect_pine_version("") == 6

    def test_v4_raises_pf001(self) -> None:
        """v1-v4 is an explicit PRD §3.3 non-goal."""
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            detect_pine_version("//@version=4\nstudy(\"X\")\n")
        assert "PF001" in str(exc.value)
        assert "v4" in str(exc.value)
        # The tracking URL should be surfaced so the user can file
        # against the right label.
        assert PineUnsupportedFeatureError.tracking_url in str(exc.value)

    def test_v3_raises_pf001(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            detect_pine_version("//@version=3\nstudy(\"X\")\n")
        assert "PF001" in str(exc.value)

    def test_v1_raises_pf001(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            detect_pine_version("//@version=1\n")
        assert "PF001" in str(exc.value)

    def test_v7_raises_pf002(self) -> None:
        """We don't speculate ahead — v6 is the target."""
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            detect_pine_version("//@version=7\nindicator(\"X\")\n")
        assert "PF002" in str(exc.value)

    def test_v10_raises_pf002(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            detect_pine_version("//@version=10\nindicator(\"X\")\n")
        assert "PF002" in str(exc.value)

    def test_pragma_with_trailing_whitespace(self) -> None:
        """Whitespace after the version number is tolerated."""
        assert detect_pine_version("//@version=5   \nindicator(\"X\")\n") == 5

    def test_pragma_not_first_line(self) -> None:
        """A pragma anywhere in source is read; PRD lexer only honors col-0
        but ``detect_pine_version`` is more permissive — it's a fingerprint,
        not authoritative routing. The lexer/parser still owns final routing.
        """
        src = "// some comment\n//@version=5\nindicator(\"X\")\n"
        assert detect_pine_version(src) == 5


# ---------------------------------------------------------------------------
# V5_REWRITES catalog shape — frozen, slotted, well-formed
# ---------------------------------------------------------------------------


class TestRewriteCatalogShape:
    def test_catalog_is_nonempty_tuple(self) -> None:
        assert isinstance(V5_REWRITES, tuple)
        assert len(V5_REWRITES) >= 5  # the bead floor

    def test_every_rewrite_has_compiled_pattern(self) -> None:
        for rw in V5_REWRITES:
            assert isinstance(rw.pattern, re.Pattern), (
                f"{rw.name}: pattern must be a compiled re.Pattern"
            )

    def test_every_rewrite_has_description(self) -> None:
        for rw in V5_REWRITES:
            assert rw.description, f"{rw.name}: description must be non-empty"

    def test_v5_rewrite_is_frozen_and_slotted(self) -> None:
        import dataclasses

        rw = V5_REWRITES[0]
        # frozen
        with pytest.raises(dataclasses.FrozenInstanceError):
            rw.name = "changed"  # type: ignore[misc]
        # slotted: no __dict__
        assert not hasattr(rw, "__dict__")

    def test_every_rewrite_bidi_safe_flag(self) -> None:
        """We currently mark every shipped rewrite bidi_safe; the test
        catches the next contributor who flips one to False without
        weighing the consequences."""
        for rw in V5_REWRITES:
            assert rw.bidi_safe is True, (
                f"{rw.name}: bidi_safe=False — explain why in the V5_REWRITES "
                "table comment and remove this assertion."
            )

    def test_v5rewrite_apply_returns_tuple(self) -> None:
        rw = V5_REWRITES[0]
        result, n = rw.apply("study(\"X\")")
        assert isinstance(result, str)
        assert isinstance(n, int)
        assert n >= 1


# ---------------------------------------------------------------------------
# Per-rewrite assertions
# ---------------------------------------------------------------------------


class TestPerRewrite:
    """One v5 input → expected v6 output per rewrite."""

    def test_study_to_indicator_basic(self) -> None:
        out, log = migrate_v5_to_v6("//@version=5\nstudy(\"My Indicator\")\n")
        assert "indicator(" in out
        assert "study(" not in out
        # Migration log mentions the rewrite by name.
        assert any("study" in entry and "indicator" in entry for entry in log)

    def test_study_to_indicator_with_args(self) -> None:
        out, _ = migrate_v5_to_v6(
            "//@version=5\nstudy(\"X\", overlay=true, shorttitle=\"X\")\n"
        )
        assert 'indicator("X", overlay=true, shorttitle="X")' in out

    def test_study_to_indicator_with_extra_whitespace(self) -> None:
        out, _ = migrate_v5_to_v6('//@version=5\nstudy   ("Spaced")\n')
        assert "indicator(" in out
        assert "study(" not in out

    def test_study_does_not_match_mystudy(self) -> None:
        """`\\b` boundary keeps us from rewriting suffixed identifiers."""
        out, _ = migrate_v5_to_v6('//@version=5\nindicator("X")\nmystudy_var = 1\n')
        assert "mystudy_var" in out

    def test_strip_transp_basic(self) -> None:
        out, _ = migrate_v5_to_v6('//@version=5\nindicator("X")\nplot(close, transp=50)\n')
        assert "transp=" not in out
        assert "plot(close)" in out

    def test_strip_transp_with_other_args(self) -> None:
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\nplot(close, color=color.red, transp=30, linewidth=2)\n'
        )
        assert "transp" not in out
        assert "color=color.red" in out
        assert "linewidth=2" in out

    def test_strip_transp_with_decimal_value(self) -> None:
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\nplot(x, transp=12.5)\n'
        )
        assert "transp" not in out

    def test_iff_to_ternary_simple(self) -> None:
        out, _ = migrate_v5_to_v6('//@version=5\nindicator("X")\ny = iff(c, a, b)\n')
        assert "iff(" not in out
        assert "(c ? a : b)" in out

    def test_iff_to_ternary_with_spaces(self) -> None:
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\ny = iff( cond , left , right )\n'
        )
        assert "iff(" not in out
        assert "(cond ? left : right)" in out

    def test_tickerid_to_ticker_new(self) -> None:
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\nt = tickerid("NYSE", "AAPL")\n'
        )
        assert "tickerid(" not in out
        assert 'ticker.new("NYSE", "AAPL")' in out

    def test_security_to_request_security(self) -> None:
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\ns = security("AAPL", "1D", close)\n'
        )
        assert "request.security(" in out
        # Bare `security(` should be rewritten — but a substring search
        # for `security(` still hits "request.security(", so check by
        # negation against the *bare* form.
        assert re.search(r"(?<![\w.])security\(", out) is None

    def test_security_does_not_rewrite_already_namespaced(self) -> None:
        """`request.security(` already namespaced; leave it alone."""
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\ns = request.security("AAPL", "1D", close)\n'
        )
        # The rewrite should NOT have doubled up to `request.request.security(`.
        assert "request.request.security" not in out
        assert "request.security(" in out

    def test_security_does_not_rewrite_method_access(self) -> None:
        """A method call like ``obj.security(`` should NOT be rewritten."""
        out, _ = migrate_v5_to_v6(
            '//@version=5\nindicator("X")\ns = my_obj.security("X")\n'
        )
        # Should be unchanged; my_obj.security(...) is method access, not
        # the v5 builtin.
        assert "my_obj.security(" in out
        assert "request.security" not in out

    def test_pragma_rewritten_to_v6(self) -> None:
        out, _ = migrate_v5_to_v6("//@version=5\n")
        assert "//@version=6" in out
        assert "//@version=5" not in out

    def test_pragma_log_entry_present(self) -> None:
        _, log = migrate_v5_to_v6("//@version=5\n")
        assert any("//@version=5" in entry and "//@version=6" in entry for entry in log)


# ---------------------------------------------------------------------------
# Idempotence — applying the same rewrites twice == applying them once.
# ---------------------------------------------------------------------------


class TestIdempotence:
    """Every ``bidi_safe`` rewrite must be idempotent.

    This is the invariant that prevents the rewrite-table contributor's
    next-day footgun: a v6-shaped output that itself matches the v5
    pattern, causing cascading rewrites.
    """

    @pytest.mark.parametrize(
        "v5_source",
        [
            '//@version=5\nstudy("X")\n',
            '//@version=5\nstudy("X", overlay=true)\n',
            '//@version=5\nindicator("X")\nplot(close, transp=50)\n',
            '//@version=5\nindicator("X")\ny = iff(c, a, b)\n',
            '//@version=5\nindicator("X")\nt = tickerid("NYSE", "AAPL")\n',
            '//@version=5\nindicator("X")\ns = security("AAPL", "1D", close)\n',
            # Combined: every rewrite firing in one source.
            (
                '//@version=5\nstudy("Combined")\n'
                'plot(close, color=color.blue, transp=20)\n'
                'y = iff(close > open, 1, 0)\n'
                't = tickerid("NYSE", "AAPL")\n'
                's = security("AAPL", "1D", close)\n'
            ),
        ],
    )
    def test_double_apply_is_no_op(self, v5_source: str) -> None:
        once, _ = migrate_v5_to_v6(v5_source)
        twice, log2 = migrate_v5_to_v6(once)
        assert once == twice, (
            f"migrate_v5_to_v6 is not idempotent on:\n{v5_source}\n"
            f"first pass: {once!r}\nsecond pass: {twice!r}\nlog: {log2}"
        )

    def test_already_v6_source_unchanged(self) -> None:
        """Pure v6 source should pass through with no rewrites and empty log."""
        v6_src = '//@version=6\nindicator("X")\nplot(close)\n'
        out, log = migrate_v5_to_v6(v6_src)
        assert out == v6_src
        assert log == []


# ---------------------------------------------------------------------------
# End-to-end migration on a realistic v5 script
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_combined_rewrites_in_one_script(self) -> None:
        """A multi-rewrite v5 script: study + transp + iff in one pass."""
        v5_src = (
            "//@version=5\n"
            'study("Combo", overlay=true)\n'
            'plot(close, color=color.red, transp=40)\n'
            'y = iff(close > open, close, open)\n'
        )
        out, log = migrate_v5_to_v6(v5_src)

        # All three v5 constructs rewritten.
        assert "study(" not in out
        assert "transp=" not in out
        assert "iff(" not in out

        # All three v6 forms present.
        assert "indicator(" in out
        assert "(close > open ? close : open)" in out

        # Pragma rewritten.
        assert "//@version=6" in out
        assert "//@version=5" not in out

        # Log captures every applied transform.
        assert len(log) >= 4  # study, transp, iff, pragma

    def test_returns_str_and_list(self) -> None:
        out, log = migrate_v5_to_v6("//@version=5\n")
        assert isinstance(out, str)
        assert isinstance(log, list)

    def test_non_str_input_raises_typeerror(self) -> None:
        """Type-guard: positional misuse caught at boundary, not at re.subn."""
        with pytest.raises(TypeError):
            migrate_v5_to_v6(b"//@version=5\n")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compile_pine — high-level facade integration
# ---------------------------------------------------------------------------


class TestCompilePineFacade:
    """``compile_pine_to_program`` should hide migration entirely from
    typical callers.

    These tests use :func:`compile_pine_to_program` (the C7-bead-era
    ``compile_pine``-but-returning-IR surface) — the public
    :func:`compile_pine` was extended in Wave 4 (C5 codegen) to return
    a full :class:`CompiledModule` with emitted Python source. The IR-
    inspection surface those C7 tests need lives at the lower-level
    helper. See the docstring on ``compile_pine_to_program`` for the
    rationale split.
    """

    def test_v5_indicator_script_compiles(self) -> None:
        """The canonical M1-gate scenario: a v5 RSI script compiles unedited."""
        src = (
            "//@version=5\n"
            'study("RSI")\n'
            "plot(ta.rsi(close, 14))\n"
        )
        prog = compile_pine_to_program(src)
        assert isinstance(prog, ir.Program)
        assert prog.version == 6  # migrated
        # The directive should be `indicator`, not `study` — Program IR
        # has no `StudyDecl`; the rename happened before the parse.
        assert prog.directive.kind == "indicator"
        assert prog.directive.title == "RSI"

    def test_v6_indicator_script_compiles_without_migration(self) -> None:
        src = (
            "//@version=6\n"
            'indicator("RSI")\n'
            "plot(ta.rsi(close, 14))\n"
        )
        prog = compile_pine_to_program(src)
        assert isinstance(prog, ir.Program)
        assert prog.version == 6
        assert prog.directive.kind == "indicator"

    def test_v5_with_combined_rewrites_compiles(self) -> None:
        src = (
            "//@version=5\n"
            'study("X", overlay=true)\n'
            "plot(close, color=color.red, transp=20)\n"
            "y = iff(close > open, close, open)\n"
        )
        prog = compile_pine_to_program(src)
        assert prog.version == 6
        assert prog.directive.kind == "indicator"

    def test_no_pragma_uses_v6_default(self) -> None:
        src = 'indicator("X")\nplot(close)\n'
        prog = compile_pine_to_program(src)
        assert prog.version == 6

    def test_v4_raises_pf001(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            compile_pine_to_program('//@version=4\nstudy("X")\n')
        assert "PF001" in str(exc.value)

    def test_v7_raises_pf002(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            compile_pine_to_program('//@version=7\nindicator("X")\n')
        assert "PF002" in str(exc.value)

    def test_target_version_must_be_5_or_6(self) -> None:
        with pytest.raises(PineUnsupportedFeatureError):
            compile_pine_to_program('indicator("X")\n', target_version=7)

    def test_returns_ir_program_type(self) -> None:
        prog = compile_pine_to_program('//@version=6\nindicator("X")\n')
        assert type(prog).__name__ == "Program"


# ---------------------------------------------------------------------------
# Unsupported feature path
# ---------------------------------------------------------------------------


class TestUnsupportedFeature:
    """PF003: v5 source still contains a v5-only construct after rewrites."""

    def test_nested_iff_raises_pf003(self) -> None:
        """A nested iff() with a function-call arg isn't matched by the
        simple ``[^,()]+`` regex; the sentinel scan catches it and raises
        PF003 with the tracking URL."""
        v5_src = (
            "//@version=5\n"
            'indicator("X")\n'
            "y = iff(crossover(a, b), c, d)\n"
        )
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            migrate_v5_to_v6(v5_src)
        assert "PF003" in str(exc.value)
        assert "iff" in str(exc.value)
        # The tracking URL is surfaced so the user can file a rewrite
        # request immediately.
        assert PineUnsupportedFeatureError.tracking_url in str(exc.value)

    def test_pf003_error_has_tracking_url_attribute(self) -> None:
        """The tracking_url is an attribute, not just in the message text."""
        with pytest.raises(PineUnsupportedFeatureError) as exc:
            migrate_v5_to_v6(
                "//@version=5\n"
                'indicator("X")\n'
                "y = iff(crossover(a, b), c, d)\n"
            )
        assert exc.value.tracking_url is not None
        assert "pine-v5-migration" in exc.value.tracking_url
