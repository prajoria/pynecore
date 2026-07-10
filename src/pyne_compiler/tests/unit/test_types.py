"""Tests for ``openbb_pine.compiler.types`` — PineType lattice + CompiledModule contract.

Source of truth: D1 §3.1 (CompiledModule) and §4.1-§4.2 (PineType + qualifier
lattice). The af08128d3 cross-doc consolidation requires builtins_used,
security_contexts, and cache_status as part of the codegen contract — D2/D3
consume them.
"""

from __future__ import annotations

import dataclasses

import pytest


# ---------------------------------------------------------------------------
# PineType + InnerType + qualifier lattice (D1 §4.1-§4.2)
# ---------------------------------------------------------------------------


class TestQualifierLattice:
    """Qualifier promotion follows ``const → input → simple → series``."""

    def test_can_promote_along_lattice(self) -> None:
        from openbb_pine.compiler.types import can_promote

        for src, dst in [
            ("const", "input"),
            ("const", "simple"),
            ("const", "series"),
            ("input", "simple"),
            ("input", "series"),
            ("simple", "series"),
        ]:
            assert can_promote(src, dst), f"{src} should promote to {dst}"

    def test_can_promote_is_reflexive(self) -> None:
        from openbb_pine.compiler.types import can_promote

        for q in ("const", "input", "simple", "series"):
            assert can_promote(q, q)

    def test_cannot_demote(self) -> None:
        from openbb_pine.compiler.types import can_promote

        for src, dst in [
            ("series", "simple"),
            ("series", "input"),
            ("series", "const"),
            ("simple", "input"),
            ("simple", "const"),
            ("input", "const"),
        ]:
            assert not can_promote(src, dst), f"{src} must not demote to {dst}"


class TestScalarAndPineType:
    """``PineType`` is the ``(qualifier, inner)`` pair from D1 §4.1."""

    def test_construct_series_float(self) -> None:
        from openbb_pine.compiler.types import PineType, Scalar

        t = PineType(qualifier="series", inner=Scalar(kind="float"))
        assert t.qualifier == "series"
        assert t.inner.kind == "float"

    def test_pinetype_frozen(self) -> None:
        from openbb_pine.compiler.types import PineType, Scalar

        t = PineType(qualifier="const", inner=Scalar(kind="int"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.qualifier = "series"  # type: ignore[misc]

    def test_pinetype_equality_and_hash(self) -> None:
        from openbb_pine.compiler.types import PineType, Scalar

        a = PineType(qualifier="simple", inner=Scalar(kind="int"))
        b = PineType(qualifier="simple", inner=Scalar(kind="int"))
        c = PineType(qualifier="series", inner=Scalar(kind="int"))
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        # Slots → no __dict__
        assert not hasattr(a, "__dict__")


class TestUnifyQualifiers:
    """``unify`` returns the max-qualifier supertype (D1 §4.2)."""

    def test_unify_promotes_to_max_qualifier(self) -> None:
        from openbb_pine.compiler.types import PineType, Scalar, unify

        a = PineType(qualifier="const", inner=Scalar(kind="float"))
        b = PineType(qualifier="series", inner=Scalar(kind="float"))
        u = unify(a, b)
        assert u.qualifier == "series"
        assert u.inner == Scalar(kind="float")

    def test_unify_rejects_incompatible_inners(self) -> None:
        from openbb_pine.errors import PineTypeError
        from openbb_pine.compiler.types import PineType, Scalar, unify

        a = PineType(qualifier="const", inner=Scalar(kind="float"))
        b = PineType(qualifier="const", inner=Scalar(kind="string"))
        with pytest.raises(PineTypeError):
            unify(a, b)


class TestInnerTypes:
    """Every InnerType variant from D1 §4.1 constructs cleanly."""

    def test_array_and_map_and_matrix(self) -> None:
        from openbb_pine.compiler.types import (
            ArrayT,
            MapT,
            MatrixT,
            PineType,
            Scalar,
        )

        elem = PineType(qualifier="series", inner=Scalar(kind="float"))
        arr = ArrayT(element=elem)
        mat = MatrixT(element=elem)
        m = MapT(key=PineType(qualifier="const", inner=Scalar(kind="string")), value=elem)
        assert arr.element is elem
        assert mat.element is elem
        assert m.key.inner == Scalar(kind="string")
        assert m.value is elem

    def test_reference_kinds(self) -> None:
        from openbb_pine.compiler.types import Reference

        for kind in ("line", "label", "box", "table", "polyline", "linefill"):
            assert Reference(kind=kind).kind == kind  # type: ignore[arg-type]

    def test_udt_and_function_and_tuple_and_na_and_unknown(self) -> None:
        from openbb_pine.compiler.types import (
            FunctionT,
            NaT,
            PineType,
            Scalar,
            TupleT,
            UDT,
            UnknownT,
        )

        scalar_int = PineType(qualifier="simple", inner=Scalar(kind="int"))
        scalar_bool = PineType(qualifier="series", inner=Scalar(kind="bool"))
        udt = UDT(name="MyBar")
        fn = FunctionT(
            params=(scalar_int, scalar_bool),
            return_type=scalar_bool,
            type_params=("T",),
        )
        tup = TupleT(elements=(scalar_int, scalar_bool))
        na = NaT()
        unk = UnknownT(var_id=42)

        assert udt.name == "MyBar"
        assert fn.params[0] is scalar_int
        assert fn.return_type is scalar_bool
        assert fn.type_params == ("T",)
        assert tup.elements == (scalar_int, scalar_bool)
        assert isinstance(na, NaT)
        assert unk.var_id == 42


# ---------------------------------------------------------------------------
# CompiledModule (D1 §3.1 + af08128d3 consolidation)
# ---------------------------------------------------------------------------


class TestCompiledModule:
    """``CompiledModule`` carries the full downstream contract."""

    def _make(self, **overrides):
        from openbb_pine.compiler.types import CompiledModule

        defaults = dict(
            source="from pynecore.lib import close\n",
            sha="blake2b:abc",
            pine_version=6,
            compiler_version="0.1.0",
            builtins_used=frozenset({"ta.sma", "input.int"}),
            security_contexts=None,
            cache_status="miss",
        )
        defaults.update(overrides)
        return CompiledModule(**defaults)

    def test_constructs_with_all_required_fields(self) -> None:
        m = self._make()
        assert m.source.startswith("from pynecore.lib")
        assert m.sha == "blake2b:abc"
        assert m.pine_version == 6
        assert m.compiler_version == "0.1.0"
        assert m.builtins_used == frozenset({"ta.sma", "input.int"})
        assert m.security_contexts is None
        assert m.cache_status == "miss"

    def test_frozen(self) -> None:
        m = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.cache_status = "hit"  # type: ignore[misc]

    def test_round_trips_through_asdict(self) -> None:
        """``CompiledModule`` must be a real dataclass round-trippable for D3 wire layer."""
        from openbb_pine.compiler.types import CompiledModule

        m = self._make()
        d = dataclasses.asdict(m)
        assert d["source"] == m.source
        assert d["sha"] == "blake2b:abc"
        assert d["pine_version"] == 6
        # frozenset survives asdict as a set-like; convert for ordering-stable test.
        assert set(d["builtins_used"]) == {"ta.sma", "input.int"}
        assert d["security_contexts"] is None
        assert d["cache_status"] == "miss"
        assert dataclasses.is_dataclass(CompiledModule)

    def test_builtins_used_is_frozenset(self) -> None:
        m = self._make()
        assert isinstance(m.builtins_used, frozenset)

    def test_cache_status_accepts_three_values(self) -> None:
        for status in ("hit", "miss", "bypass"):
            m = self._make(cache_status=status)
            assert m.cache_status == status

    def test_security_contexts_dict(self) -> None:
        from openbb_pine.compiler.types import SecurityContext

        ctx = SecurityContext(symbol="AAPL", timeframe="1D", expr="close")
        m = self._make(security_contexts={"ctx_0": ctx})
        assert m.security_contexts is not None
        assert m.security_contexts["ctx_0"].symbol == "AAPL"
        assert m.security_contexts["ctx_0"].timeframe == "1D"

    def test_compiledmodule_uses_slots(self) -> None:
        m = self._make()
        assert not hasattr(m, "__dict__")
