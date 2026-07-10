"""Type-checker coverage for the ``strategy.*`` signature surface — bead 0e9.6.h14.

Design reference: ``docs/designs/openbb-pine/D5-strategy-engine.md`` §7.2.

This bead adds the C3 signature entries for Pine's strategy engine surface:

* Order-management calls: ``strategy.entry`` / ``strategy.exit`` /
  ``strategy.close`` / ``strategy.close_all`` / ``strategy.cancel`` /
  ``strategy.cancel_all``.
* Direction constants: ``strategy.long`` / ``strategy.short``.
* Qty-type constants: ``strategy.fixed`` / ``strategy.cash`` /
  ``strategy.percent_of_equity``.

Split of concerns (Wave 2 M2, D5 §7 amendment surface):

* **This bead (h14)** — ONLY the C3 signature registry entries. The type
  checker must accept well-typed ``strategy.*`` calls (bindings, arg types,
  attribute-access on the ``strategy`` namespace) and reject ill-typed ones.
* Bead ``pxc`` (grammar) — already landed; makes ``KW_STRATEGY`` a legal
  ``primary`` so ``strategy.foo(...)`` parses.
* Bead ``aeh`` (codegen — Wave 3) — will lift the ``PineUnsupportedFeatureError``
  PF010 stub in ``codegen.py`` (currently at line ~423) and route the
  signature entries through the ``openbb_pine.stdlib.strategy`` bridge.

Contract every test here MUST honour: exercise the FULL pipeline via
``compile_pine(src)`` — lex + parse + type-check + codegen. Because codegen
is not yet wired for ``strategy(...)``, we expect a
``PineUnsupportedFeatureError`` with code PF010 at the very end. That is
deliberate — it proves the type checker's ``strategy.*`` support is
end-to-end and doesn't rely on skipping the walk. When bead ``aeh`` lands,
these tests will start failing on the PF010 expectation and will need to be
relaxed to unconditional-pass; that's the exit signal for the codegen bead.
"""

from __future__ import annotations

import pytest

from openbb_pine.compiler import compile_pine
from openbb_pine.compiler.builtin_signatures import BUILTIN_SIGNATURES, lookup
from openbb_pine.compiler.types import PineType, Scalar
from openbb_pine.errors import (
    PineTypeError,
    PineUnsupportedFeatureError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRATEGY_HEAD = '//@version=6\nstrategy("Test", overlay=true)\n'
"""Minimal v6 strategy scaffold — every fixture prepends this so the body
sits in a strategy-scoped program (mirroring bead pxc's grammar tests)."""

_CONST_STRING = PineType(qualifier="const", inner=Scalar(kind="string"))
"""Expected return type for ``strategy.long`` / ``strategy.short`` /
``strategy.percent_of_equity`` / etc. — matches the module-level constant of
the same name in ``compiler.builtin_signatures``."""


def _compile_strategy_body(body: str) -> None:
    """Compile a Pine v6 strategy with ``body`` in its body position.

    Bead h14 owns type-check; codegen is bead ``aeh``. Compilation MUST reach
    codegen and raise ``PineUnsupportedFeatureError`` PF010 — that proves the
    type checker walked the body cleanly. Any other exception (PineTypeError,
    PineSyntaxError) indicates a real failure that must fail the test.
    """
    src = _STRATEGY_HEAD + body
    with pytest.raises(PineUnsupportedFeatureError) as excinfo:
        compile_pine(src, use_cache=False)
    # PF010 = strategy() codegen deferred to bead aeh; that's the expected
    # failure mode for now. When bead aeh lands, this assertion breaks and
    # tests transition to unconditional-pass. Assert against the structured
    # ``feature`` attribute rather than the stringified error, so a future
    # refactor of the message body doesn't silently break this check.
    feature = excinfo.value.feature or ""
    assert feature.startswith("PF010"), (
        f"expected PF010 codegen deferral (structured .feature); "
        f"got feature={feature!r} err={excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# 1. Registry shape — every landed entry is discoverable and marked IMPLEMENTED
# ---------------------------------------------------------------------------


class TestRegistryShape:
    """Verify the bead h14 additions actually landed in the registry.

    Signature-only entries land here with ``notes="SIGNATURE_ONLY"`` —
    ``_coverage_manifest.py::BUILTINS_IMPLEMENTED`` explicitly excludes them
    (a script using them still crashes at codegen with PF010). Bead ``aeh``
    will flip the notes to ``"IMPLEMENTED"`` when the bridge lands and the
    coverage manifest will pick them up at that point.
    """

    STRATEGY_CALLS = (
        "strategy.entry",
        "strategy.exit",
        "strategy.close",
        "strategy.close_all",
        "strategy.cancel",
        "strategy.cancel_all",
    )
    STRATEGY_CONSTANTS = (
        "strategy.long",
        "strategy.short",
        "strategy.fixed",
        "strategy.cash",
        "strategy.percent_of_equity",
    )

    @pytest.mark.parametrize("name", STRATEGY_CALLS + STRATEGY_CONSTANTS)
    def test_entry_exists_and_marked_signature_only(self, name: str) -> None:
        sig = lookup(name)
        assert sig is not None, f"missing {name!r} in BUILTIN_SIGNATURES"
        # ``SIGNATURE_ONLY`` is the h14 marker — codegen still raises PF010.
        # ``IMPLEMENTED`` is the post-aeh marker — bridge is landed. Accept
        # either so this test survives the bead-aeh flip without a rewrite.
        assert sig.notes in ("SIGNATURE_ONLY", "IMPLEMENTED"), (
            f"{name!r} must be notes='SIGNATURE_ONLY' (h14) or "
            f"'IMPLEMENTED' (post-aeh); got notes={sig.notes!r}"
        )

    def test_constants_return_const_string(self) -> None:
        """Direction and qty-type enum values must resolve as const<string>."""
        for name in self.STRATEGY_CONSTANTS:
            sig = BUILTIN_SIGNATURES[name]
            assert sig.args == (), (
                f"{name!r} is a value, not a call — args must be empty"
            )
            assert sig.returns == _CONST_STRING, (
                f"{name!r} must return const<string> "
                f"(same qualifier as ``\"long\"`` etc.); "
                f"got {sig.returns!r}"
            )


# ---------------------------------------------------------------------------
# 2. Order-management calls type-check cleanly
# ---------------------------------------------------------------------------


class TestStrategyOrderCallsTypeCheck:
    """Well-typed strategy.* calls must survive C3 unaltered and only fail
    later at codegen with PF010."""

    def test_entry_with_string_direction(self) -> None:
        """``strategy.entry("long_1", "long", qty=1.0)`` — the canonical
        entry shape from Pine tutorials. Direction as a string literal
        (not the ``strategy.long`` enum) is legal because both are
        const<string>."""
        _compile_strategy_body('strategy.entry("long_1", "long", qty=1.0)\n')

    def test_entry_with_direction_constant(self) -> None:
        """``strategy.entry("long_1", strategy.long, qty=1)`` — direction
        via the ``strategy.long`` namespace constant. Requires BOTH the
        signature entry for ``strategy.long`` (returns const<string>) AND
        the signature entry for ``strategy.entry`` (accepts const<string>
        for its ``direction`` slot)."""
        _compile_strategy_body(
            'strategy.entry("long_1", strategy.long, qty=1)\n'
        )

    def test_entry_with_short_constant(self) -> None:
        _compile_strategy_body(
            'strategy.entry("short_1", strategy.short, qty=1)\n'
        )

    def test_entry_with_limit_and_stop(self) -> None:
        """``strategy.entry`` supports series<float> for ``limit``/``stop`` —
        prices are typically ``close * 0.99`` or similar computed series."""
        _compile_strategy_body(
            'strategy.entry("entry_1", "long", qty=1, '
            'limit=close * 0.99, stop=close * 0.95)\n'
        )

    def test_exit_with_from_entry_and_stop(self) -> None:
        """``strategy.exit("exit_1", from_entry="long_1", stop=90.0)`` —
        exit tied to a specific entry id."""
        _compile_strategy_body(
            'strategy.exit("exit_1", from_entry="long_1", stop=90.0)\n'
        )

    def test_exit_with_profit_and_loss(self) -> None:
        """``strategy.exit`` with the offset-based bracket kwargs."""
        _compile_strategy_body(
            'strategy.exit("bracket", from_entry="long_1", '
            'profit=100, loss=50)\n'
        )

    def test_exit_with_trailing_stop(self) -> None:
        """Trailing-stop kwargs (``trail_price`` / ``trail_offset``) —
        ``trail_price`` is series<float>."""
        _compile_strategy_body(
            'strategy.exit("trail", from_entry="long_1", '
            'trail_price=close, trail_offset=1.0)\n'
        )

    def test_close_by_id(self) -> None:
        _compile_strategy_body('strategy.close("long_1")\n')

    def test_close_with_comment(self) -> None:
        _compile_strategy_body(
            'strategy.close("long_1", comment="closing long")\n'
        )

    def test_close_with_alert_message(self) -> None:
        _compile_strategy_body(
            'strategy.close("long_1", '
            'alert_message="close signal fired")\n'
        )

    def test_close_all_no_args(self) -> None:
        _compile_strategy_body('strategy.close_all()\n')

    def test_close_all_with_comment(self) -> None:
        _compile_strategy_body(
            'strategy.close_all(comment="EOD flatten")\n'
        )

    def test_cancel_by_id(self) -> None:
        _compile_strategy_body('strategy.cancel("pending_1")\n')

    def test_cancel_all_no_args(self) -> None:
        _compile_strategy_body('strategy.cancel_all()\n')


# ---------------------------------------------------------------------------
# 3. Namespace constants resolve as const<string>
# ---------------------------------------------------------------------------


class TestStrategyDirectionConstants:
    """Direction and qty-type enums must be usable as bare value
    expressions AND as kwarg values to another call."""

    @pytest.mark.parametrize("direction", ["long", "short"])
    def test_direction_bound_to_local(self, direction: str) -> None:
        """``d = strategy.long`` — assign the direction to a local, then
        use it. Because the local resolves to const<string> and
        ``strategy.entry``'s ``direction`` slot is also const<string>, this
        must type-check cleanly."""
        _compile_strategy_body(
            f'd = strategy.{direction}\n'
            f'strategy.entry("id", d)\n'
        )

    def test_direction_as_kwarg_value(self) -> None:
        """``strategy.entry("id", direction=strategy.long)`` — kwarg form
        specifically (distinct from positional in
        ``test_entry_with_direction_constant``)."""
        _compile_strategy_body(
            'strategy.entry("id", direction=strategy.long)\n'
        )

    def test_percent_of_equity_resolves(self) -> None:
        """``qt = strategy.percent_of_equity`` — the qty-type enum resolves
        as const<string> and is bindable to a local."""
        _compile_strategy_body(
            'qt = strategy.percent_of_equity\n'
        )

    def test_all_qty_type_enums_resolve(self) -> None:
        """Bind each of ``strategy.fixed``, ``strategy.cash``,
        ``strategy.percent_of_equity`` to locals — all three must
        type-check as const<string>."""
        _compile_strategy_body(
            'a = strategy.fixed\n'
            'b = strategy.cash\n'
            'c = strategy.percent_of_equity\n'
        )


# ---------------------------------------------------------------------------
# 4. Ill-typed calls are rejected
# ---------------------------------------------------------------------------


class TestStrategyIllTypedRejection:
    """Verify the type checker actually enforces the const<string>/etc.
    contracts we declared — else the signature entries would be no-ops."""

    def test_entry_rejects_series_id(self) -> None:
        """``strategy.entry(close, "long")`` — ``id`` is const<string>;
        ``close`` is series<float>. The qualifier lattice runs
        const→input→simple→series in one direction only, so series cannot
        demote to const. C3's PT001 rule fires on the qualifier check.

        Historic note: an INNER-TYPE mismatch (``123`` = const<int> vs
        const<string>) is NOT enforced by C3 today — ``_rule_pt001`` only
        validates qualifier promotion, not inner-type compatibility. Tracked
        by bead ``OpenBBTechnical-b29`` ("C3: inner-type validation for
        strategy.* signatures") — that bead's acceptance flip is exactly
        this test: strengthen it from the current qualifier-only check to
        an inner-type rejection when C3 gains inner-type awareness.
        """
        src = _STRATEGY_HEAD + 'strategy.entry(close, "long")\n'
        with pytest.raises(PineTypeError) as excinfo:
            compile_pine(src, use_cache=False)
        # PT001 = the qualifier-lattice rule. Rule attribution must be
        # PT001 (or the ``undefined`` fallback if C3 rejects earlier);
        # accept either but require SOME rule.
        assert excinfo.value.rule in ("PT001", "undefined"), (
            f"expected PT001 (qualifier) rule; got {excinfo.value.rule!r}"
        )

    def test_entry_accepts_string_literal_direction(self) -> None:
        """``strategy.entry("id", "sideways")`` — value-set validation
        ("long"/"short" only) is deferred: the Signature dataclass carries
        no enum-value axis. So a bareword string literal MUST type-check
        cleanly (only the inner-type is checked, and both "sideways" and
        "long" are const<string>). This is a REGRESSION guard so a future
        bead that adds value-set validation can find & flip this test."""
        _compile_strategy_body('strategy.entry("id", "sideways")\n')

    def test_entry_rejects_series_direction(self) -> None:
        """A series<string>-qualified value cannot promote to
        const<string>. Pine has few series<string> sources today; we
        synthesise one by feeding an ``input.source`` (series<float>)
        through a str.tostring — but that's series-qualified. Instead the
        cleanest smoke: use ``close`` (series<float>) which is inner-type
        mismatched AND series-qualified, and expect any PineTypeError."""
        src = _STRATEGY_HEAD + 'strategy.entry("id", close)\n'
        with pytest.raises(PineTypeError):
            compile_pine(src, use_cache=False)


# ---------------------------------------------------------------------------
# 5. builtins_used population (D1 §3.1 contract — flows to CompiledModule)
# ---------------------------------------------------------------------------


class TestBuiltinsUsedPopulation:
    """Every ``strategy.*`` reference — call or constant — must land in
    ``builtins_used`` so PRD §3.4 L0.5 wild-corpus coverage attributes
    strategy-scoped scripts correctly.

    We inspect the type checker's output DIRECTLY here (via the low-level
    ``check`` entry) rather than through ``compile_pine`` — because
    ``compile_pine`` raises PF010 at codegen and the resulting
    ``CompiledModule.builtins_used`` set is never surfaced to the caller.
    """

    def _typecheck_body(self, body: str):
        """Lex + parse + type-check a strategy-scoped script; return the
        TypeCheckResult. Skips codegen entirely."""
        from openbb_pine.compiler.lexer import tokenize
        from openbb_pine.compiler.parser import parse
        from openbb_pine.compiler.type_checker import check

        src = _STRATEGY_HEAD + body
        program = parse(tokenize(src), pine_version=6)
        return check(program, pine_version=6)

    def test_entry_call_recorded(self) -> None:
        result = self._typecheck_body(
            'strategy.entry("id", "long", qty=1.0)\n'
        )
        assert "strategy.entry" in result.builtins_used

    def test_direction_constant_recorded(self) -> None:
        result = self._typecheck_body('d = strategy.long\n')
        assert "strategy.long" in result.builtins_used

    def test_qty_type_constant_recorded(self) -> None:
        result = self._typecheck_body(
            'qt = strategy.percent_of_equity\n'
        )
        assert "strategy.percent_of_equity" in result.builtins_used

    def test_multiple_strategy_names_recorded(self) -> None:
        """A realistic strategy body — entry + exit + close_all + a
        direction constant — populates ``builtins_used`` with every name
        referenced."""
        result = self._typecheck_body(
            'strategy.entry("long_1", strategy.long, qty=1)\n'
            'strategy.exit("exit_1", from_entry="long_1", stop=90.0)\n'
            'strategy.close_all()\n'
        )
        for name in (
            "strategy.entry",
            "strategy.exit",
            "strategy.close_all",
            "strategy.long",
        ):
            assert name in result.builtins_used, (
                f"missing {name!r} in {result.builtins_used!r}"
            )
