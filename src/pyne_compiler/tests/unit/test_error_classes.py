"""Tests for the structured inits on PineTypeError / PineUnsupportedBuiltinError /
PineUnsupportedFeatureError.

Source of truth: D1 §5.1 (Diagnostic shape), PRD §4.8 (error-body shape). The
post-R2/R6 consolidation pattern (PineDataValidationError + PineFMPUnreachableError)
proved the kw-only structured init shape; these are the C3-bead extension of
that pattern to the type-checker error family.

Backward compatibility: a positional-string raise (``raise PineTypeError("msg")``)
must still work for ad-hoc internal raises such as ``compiler.types.unify``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# PineTypeError
# ---------------------------------------------------------------------------


class TestPineTypeErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError(
            rule="PT001",
            expected="simple<int>",
            got="series<int>",
            expr_text="ta.sma(close, dyn_len)",
            location=("<inline>", 5, 12),
            hint="Pine's ta.sma requires a non-series length.",
        )
        assert exc.rule == "PT001"
        assert exc.expected == "simple<int>"
        assert exc.got == "series<int>"
        assert exc.expr_text == "ta.sma(close, dyn_len)"
        assert exc.location == ("<inline>", 5, 12)
        assert exc.hint == "Pine's ta.sma requires a non-series length."

    def test_default_str_renders_rule_loc_expected_got(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError(
            rule="PT003",
            expected="series<bool>",
            got="series<int>",
            location=("<inline>", 7, 4),
        )
        s = str(exc)
        assert "PT003" in s
        assert "<inline>:7:4" in s
        assert "series<bool>" in s
        assert "series<int>" in s

    def test_default_str_includes_hint_when_given(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError(
            rule="PT008", expected="bar.field", got="(missing)",
            hint="did you mean `close`?",
        )
        assert "did you mean `close`?" in str(exc)

    def test_default_str_includes_expr_text(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError(
            rule="PT004", expected="series<T>", got="simple<int>",
            expr_text="length[1]",
        )
        assert "length[1]" in str(exc)

    def test_backward_compat_message_kwarg(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError(message="raw pre-stitched error text")
        assert str(exc) == "raw pre-stitched error text"
        # Structured attrs default to None.
        assert exc.rule is None
        assert exc.expected is None

    def test_backward_compat_positional_string_still_works(self) -> None:
        """Pre-existing call sites raise PineTypeError("msg") — must keep working
        (e.g. ``compiler.types.unify`` raises this way to keep import-cycle
        free of structured-error churn)."""
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError("cannot unify types")
        assert str(exc) == "cannot unify types"
        assert exc.rule is None

    def test_no_args_yields_generic_text(self) -> None:
        from openbb_pine.errors import PineTypeError

        exc = PineTypeError()
        # Even with no fields, str() doesn't crash and produces some text.
        assert isinstance(str(exc), str)
        assert len(str(exc)) > 0

    def test_subclass_of_pinecompileerror(self) -> None:
        from openbb_pine.errors import PineCompileError, PineTypeError

        assert issubclass(PineTypeError, PineCompileError)

    def test_class_code_attr_unchanged(self) -> None:
        from openbb_pine.errors import PineTypeError

        assert PineTypeError.code == "PineTypeError"


# ---------------------------------------------------------------------------
# PineUnsupportedBuiltinError
# ---------------------------------------------------------------------------


class TestPineUnsupportedBuiltinErrorStructuredInit:
    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        exc = PineUnsupportedBuiltinError(
            "ta.ichimoku",
            suggested_alternative="Compose ta.donchian + ta.sma.",
            tracking_url="https://example/issues/ta.ichimoku",
        )
        assert exc.builtin == "ta.ichimoku"
        assert exc.suggested_alternative == "Compose ta.donchian + ta.sma."
        assert exc.tracking_url == "https://example/issues/ta.ichimoku"

    def test_default_str_includes_builtin_and_extras(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        exc = PineUnsupportedBuiltinError(
            "ta.ichimoku",
            suggested_alternative="Use donchian.",
            tracking_url="https://example/tracking",
        )
        s = str(exc)
        assert "ta.ichimoku" in s
        assert "Use donchian." in s
        assert "https://example/tracking" in s

    def test_message_kwarg_overrides_default_rendering(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        exc = PineUnsupportedBuiltinError(
            "ta.foo",
            message="custom override",
        )
        assert str(exc) == "custom override"
        # Structured attrs still populated.
        assert exc.builtin == "ta.foo"

    def test_no_args_yields_generic_text(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        exc = PineUnsupportedBuiltinError()
        assert isinstance(str(exc), str) and len(str(exc)) > 0
        assert exc.builtin is None

    def test_tracking_url_instance_shadows_class_default(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        exc = PineUnsupportedBuiltinError(
            "ta.x", tracking_url="https://my/track"
        )
        assert exc.tracking_url == "https://my/track"

    def test_class_code_attr_unchanged(self) -> None:
        from openbb_pine.errors import PineUnsupportedBuiltinError

        assert PineUnsupportedBuiltinError.code == "PineUnsupportedBuiltinError"


# ---------------------------------------------------------------------------
# PineUnsupportedFeatureError
# ---------------------------------------------------------------------------


class TestPineUnsupportedFeatureErrorStructuredInit:
    def test_structured_init_populates_attrs(self) -> None:
        from openbb_pine.errors import PineUnsupportedFeatureError

        exc = PineUnsupportedFeatureError(
            "PF002 typed decl in body",
            tracking_url="https://example/issues/pine-feature",
        )
        assert exc.feature == "PF002 typed decl in body"
        assert exc.tracking_url == "https://example/issues/pine-feature"

    def test_class_default_tracking_url_preserved_when_no_override(self) -> None:
        """The v5-migration default URL must remain on the class for callers
        that don't explicitly thread one through."""
        from openbb_pine.errors import PineUnsupportedFeatureError

        # Class-level constant is the v5-migration URL the existing C7 callers
        # rely on. Instance with no override still surfaces that URL.
        exc = PineUnsupportedFeatureError("PF003 v5 construct unmapped")
        assert "pine-v5-migration" in exc.tracking_url

    def test_message_kwarg_overrides_default_rendering(self) -> None:
        from openbb_pine.errors import PineUnsupportedFeatureError

        exc = PineUnsupportedFeatureError(message="raw text")
        assert str(exc) == "raw text"

    def test_class_code_attr_unchanged(self) -> None:
        from openbb_pine.errors import PineUnsupportedFeatureError

        assert PineUnsupportedFeatureError.code == "PineUnsupportedFeatureError"

    def test_no_args_yields_generic_text(self) -> None:
        from openbb_pine.errors import PineUnsupportedFeatureError

        exc = PineUnsupportedFeatureError()
        assert isinstance(str(exc), str) and len(str(exc)) > 0


# ---------------------------------------------------------------------------
# PineCodegenError
# ---------------------------------------------------------------------------


class TestPineCodegenErrorStructuredInit:
    """Mirror of the PineTypeError tests for the C5 codegen-gate error.

    Added as a preempt fix (same shape as the post-R2/R6/Wave-3A pattern)
    so the C5 ``_enforce_allowlist`` raises with structured attrs the REST
    envelope (D3 §4.1) renders as PRD §4.8 JSON.
    """

    def test_structured_init_populates_all_attrs(self) -> None:
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError(
            rule="CG001",
            node_kind="ast.Lambda",
            allowlist_member="any of NODE_TYPE_ALLOWLIST per D1 §3.2",
            tracking_url="https://example/issues/pine-codegen",
        )
        assert exc.rule == "CG001"
        assert exc.node_kind == "ast.Lambda"
        assert exc.allowlist_member == "any of NODE_TYPE_ALLOWLIST per D1 §3.2"
        assert exc.tracking_url == "https://example/issues/pine-codegen"

    def test_default_str_renders_rule_kind_member(self) -> None:
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError(
            rule="CG002",
            node_kind="ImportFrom('os')",
            allowlist_member="any of MODULE_ALLOWLIST",
        )
        s = str(exc)
        assert "CG002" in s
        assert "ImportFrom('os')" in s
        assert "MODULE_ALLOWLIST" in s

    def test_default_str_includes_tracking_url(self) -> None:
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError(
            rule="CG003",
            node_kind="Name('__import__')",
            tracking_url="https://example/tracking",
        )
        assert "https://example/tracking" in str(exc)

    def test_backward_compat_message_kwarg(self) -> None:
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError(message="raw pre-stitched error text")
        assert str(exc) == "raw pre-stitched error text"
        # Structured attrs default to None.
        assert exc.rule is None
        assert exc.node_kind is None
        assert exc.allowlist_member is None

    def test_backward_compat_positional_string_still_works(self) -> None:
        """Pre-existing call sites (defensive raises that don't yet thread
        structured fields) must keep working with a positional string."""
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError("codegen emitted disallowed ast.Lambda")
        assert str(exc) == "codegen emitted disallowed ast.Lambda"
        assert exc.rule is None

    def test_no_args_yields_generic_text(self) -> None:
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError()
        # Even with no fields, str() doesn't crash and produces some text.
        assert isinstance(str(exc), str)
        assert len(str(exc)) > 0

    def test_subclass_of_pinecompileerror(self) -> None:
        """The diagnostic-shape class hierarchy moves PineCodegenError under
        PineCompileError so existing ``except PineCompileError`` handlers
        catch CG### uniformly with PT###."""
        from openbb_pine.errors import PineCompileError, PineCodegenError

        assert issubclass(PineCodegenError, PineCompileError)

    def test_class_code_attr_unchanged(self) -> None:
        from openbb_pine.errors import PineCodegenError

        assert PineCodegenError.code == "PineCodegenError"

    def test_tracking_url_instance_shadows_class_default(self) -> None:
        """The PineError class-level ``tracking_url`` is None. Per-instance
        override must be visible on the exception instance (so D3 §4.1
        renders it into the REST envelope) without mutating the class."""
        from openbb_pine.errors import PineCodegenError

        exc = PineCodegenError(
            rule="CG001",
            node_kind="ast.Exec",
            tracking_url="https://my/tracking",
        )
        assert exc.tracking_url == "https://my/tracking"
        # Class default should still be None.
        assert PineCodegenError.tracking_url is None


# ---------------------------------------------------------------------------
# Existing raise sites — backward compatibility smoke test
# ---------------------------------------------------------------------------


def test_compiler_types_unify_still_raises_pinetypeerror_with_msg() -> None:
    """``compiler.types.unify`` raises ``PineTypeError(f"cannot unify ...")``
    with a positional string. Updating the init must not break it."""
    from openbb_pine.compiler.types import PineType, Scalar, unify
    from openbb_pine.errors import PineTypeError

    a = PineType(qualifier="const", inner=Scalar(kind="float"))
    b = PineType(qualifier="const", inner=Scalar(kind="string"))
    with pytest.raises(PineTypeError) as excinfo:
        unify(a, b)
    # Default str() must render the underlying positional message.
    assert "cannot unify" in str(excinfo.value)
