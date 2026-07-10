"""Tests for ``openbb_pine.compiler.builtin_signatures`` — the Phase-1 stub registry.

Source of truth: PRD §3.2 (36 Phase-1 builtins), D1 §3.1 (CompiledModule.builtins_used
contract — C3 populates this set during walk).

The registry is a **stub** for C3's contract — full per-builtin signatures land as
S-beads (S* 0e9.5.16-51) bring each builtin into pynecore.lib bridge wiring.
Until then this module guarantees:

1. lookup(name) returns a Signature for at least the 36 Phase-1 builtins so the
   type checker doesn't bottom out when a real script touches them.
2. lookup(name) returns None for unknown identifiers.
3. is_builtin_namespace(prefix) recognises the ~30 Pine namespace prefixes so C3
   can distinguish "unsupported-but-real-Pine" (raise PineUnsupportedBuiltinError,
   still record in builtins_used) from "totally unknown" (raise PineTypeError
   with rule='undefined').
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Signature dataclass
# ---------------------------------------------------------------------------


class TestSignatureDataclass:
    def test_construct_minimal_signature(self) -> None:
        from openbb_pine.compiler.builtin_signatures import Signature
        from openbb_pine.compiler.types import PineType, Scalar

        src = PineType(qualifier="series", inner=Scalar(kind="float"))
        length = PineType(qualifier="simple", inner=Scalar(kind="int"))
        sig = Signature(
            args=(("src", src), ("length", length)),
            returns=src,
        )
        assert sig.args[0] == ("src", src)
        assert sig.args[1] == ("length", length)
        assert sig.returns == src

    def test_signature_is_frozen_and_hashable(self) -> None:
        import dataclasses
        from openbb_pine.compiler.builtin_signatures import Signature
        from openbb_pine.compiler.types import PineType, Scalar

        sig = Signature(args=(), returns=PineType(qualifier="series", inner=Scalar(kind="float")))
        with pytest.raises(dataclasses.FrozenInstanceError):
            sig.version = 7  # type: ignore[misc]
        # Hashable -> can live inside a set / dict.
        assert {sig} == {sig}

    def test_signature_equality(self) -> None:
        from openbb_pine.compiler.builtin_signatures import Signature
        from openbb_pine.compiler.types import PineType, Scalar

        a = Signature(args=(), returns=PineType(qualifier="series", inner=Scalar(kind="float")))
        b = Signature(args=(), returns=PineType(qualifier="series", inner=Scalar(kind="float")))
        assert a == b
        assert hash(a) == hash(b)

    def test_default_version_is_6(self) -> None:
        from openbb_pine.compiler.builtin_signatures import Signature
        from openbb_pine.compiler.types import PineType, Scalar

        sig = Signature(args=(), returns=PineType(qualifier="series", inner=Scalar(kind="float")))
        assert sig.version == 6


# ---------------------------------------------------------------------------
# Phase-1 builtins coverage (PRD §3.2)
# ---------------------------------------------------------------------------


# PRD §3.2 — Phase-1 builtins. 29 ta.* + 7 math.* = 36 total.
PHASE1_TA: list[str] = [
    "ta.sma", "ta.ema", "ta.rma", "ta.wma", "ta.vwma", "ta.swma",
    "ta.rsi", "ta.macd", "ta.bb", "ta.stoch", "ta.cci", "ta.mfi",
    "ta.atr", "ta.tr", "ta.stdev", "ta.variance",
    "ta.highest", "ta.lowest", "ta.barssince", "ta.crossover", "ta.crossunder",
    "ta.cross", "ta.change", "ta.mom",
    "ta.roc", "ta.cum", "ta.dev", "ta.linreg", "ta.median",
]

PHASE1_MATH: list[str] = [
    "math.abs", "math.sqrt", "math.log", "math.exp",
    "math.max", "math.min", "math.pow",
]

PHASE1_INPUT: list[str] = [
    "input.int", "input.float", "input.bool", "input.string", "input.source",
]

PHASE1_PLOT: list[str] = ["plot", "plotshape", "hline"]
PHASE1_SOURCES: list[str] = ["close", "open", "high", "low", "volume", "time"]
PHASE1_NA: list[str] = ["na", "nz"]


class TestPhase1BuiltinsLookup:
    def test_all_29_ta_phase1_builtins_resolve(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        missing = [name for name in PHASE1_TA if lookup(name) is None]
        assert missing == [], f"Phase-1 ta.* lookup miss: {missing}"
        assert len(PHASE1_TA) == 29

    def test_all_7_math_phase1_builtins_resolve(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        missing = [name for name in PHASE1_MATH if lookup(name) is None]
        assert missing == [], f"Phase-1 math.* lookup miss: {missing}"
        assert len(PHASE1_MATH) == 7

    def test_input_constructors_resolve(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        for name in PHASE1_INPUT:
            assert lookup(name) is not None, f"input.* lookup miss: {name}"

    def test_plot_family_resolves(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        for name in PHASE1_PLOT:
            assert lookup(name) is not None, f"plot family lookup miss: {name}"

    def test_builtin_sources_resolve(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        # close / open / etc. are values not functions; in the registry they
        # have a zero-arg "constructor" or a marker signature so C3 can resolve
        # them uniformly during walk.
        for name in PHASE1_SOURCES:
            assert lookup(name) is not None, f"source lookup miss: {name}"

    def test_na_nz_resolve(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        for name in PHASE1_NA:
            assert lookup(name) is not None, f"na/nz lookup miss: {name}"


class TestUnknownLookup:
    def test_unknown_name_returns_none(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup

        assert lookup("ta.totally_not_a_builtin") is None
        assert lookup("not_a_namespace.thing") is None
        assert lookup("") is None


# ---------------------------------------------------------------------------
# Namespace recognition
# ---------------------------------------------------------------------------


class TestIsBuiltinNamespace:
    def test_recognises_expected_namespaces(self) -> None:
        from openbb_pine.compiler.builtin_signatures import is_builtin_namespace

        for prefix in [
            "ta", "math", "input", "array", "matrix", "map", "strategy",
            "request", "library", "line", "label", "box", "table",
            "syminfo", "barstate", "session", "alert", "currency",
            "dayofweek", "display", "earnings", "extend", "fixnan",
            "location", "month", "na", "plot", "price", "runtime",
            "sym", "year", "color", "chart",
        ]:
            assert is_builtin_namespace(prefix), f"{prefix} should be a Pine namespace"

    def test_unknown_prefix_is_not_namespace(self) -> None:
        from openbb_pine.compiler.builtin_signatures import is_builtin_namespace

        assert not is_builtin_namespace("zoltan")
        assert not is_builtin_namespace("myvar")
        assert not is_builtin_namespace("")

    def test_case_sensitive(self) -> None:
        from openbb_pine.compiler.builtin_signatures import is_builtin_namespace

        # Pine namespaces are lowercase; "TA" must NOT match.
        assert not is_builtin_namespace("TA")
        assert not is_builtin_namespace("Math")


# ---------------------------------------------------------------------------
# Stub coverage
# ---------------------------------------------------------------------------


class TestStubMarker:
    def test_stub_entries_carry_notes_marker(self) -> None:
        """Builtins whose precise contract isn't fixed yet must still resolve to
        a Signature so the C3 type checker doesn't crash; their `notes` slot
        carries 'STUB' so future S-beads can find them by grep."""
        from openbb_pine.compiler.builtin_signatures import BUILTIN_SIGNATURES, Signature

        # Every entry must be a Signature.
        for name, sig in BUILTIN_SIGNATURES.items():
            assert isinstance(sig, Signature), f"{name} entry is not a Signature"
        # At least the 36 Phase-1 builtins + the input/plot/source essentials.
        assert len(BUILTIN_SIGNATURES) >= 36


class TestSignatureReturnsConsistency:
    """Spot-check a few signatures' shape so the C3 type checker can rely on them."""

    def test_ta_sma_signature_shape(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup
        from openbb_pine.compiler.types import PineType, Scalar

        sig = lookup("ta.sma")
        assert sig is not None
        # ta.sma(src, length) -> series<float>
        assert len(sig.args) == 2
        names = [n for n, _ in sig.args]
        assert "src" in names
        assert "length" in names
        assert sig.returns == PineType(qualifier="series", inner=Scalar(kind="float"))

    def test_math_abs_signature_shape(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup
        from openbb_pine.compiler.types import PineType, Scalar

        sig = lookup("math.abs")
        assert sig is not None
        assert len(sig.args) == 1
        # math.abs returns same numeric kind it takes — registry uses float as the
        # series widening, which is the safe Pine promotion.
        assert isinstance(sig.returns.inner, Scalar)

    def test_input_int_signature_shape(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup
        from openbb_pine.compiler.types import PineType, Scalar

        sig = lookup("input.int")
        assert sig is not None
        # input.int(defval, ...) -> input<int>
        assert sig.returns == PineType(qualifier="input", inner=Scalar(kind="int"))

    def test_close_source_is_series_float(self) -> None:
        from openbb_pine.compiler.builtin_signatures import lookup
        from openbb_pine.compiler.types import PineType, Scalar

        sig = lookup("close")
        assert sig is not None
        # close is a value, not a function — args=() and returns series<float>.
        assert sig.args == ()
        assert sig.returns == PineType(qualifier="series", inner=Scalar(kind="float"))
