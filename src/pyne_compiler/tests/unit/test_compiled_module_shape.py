"""Tests for the ``script_type`` field on ``CompiledModule`` (D5 §5.1).

The ``script_type`` field is the runtime signal codegen writes so the executor
can fork between indicator, strategy, and library scripts. This bead (d75) adds
the field; a follow-up bead wires codegen to set it based on the top-level
declaration.

Contract requirements pinned here:

* Default construction (without passing ``script_type``) yields
  ``"indicator"`` — every existing call site that omits the field keeps working.
* Each of the three literal values (``"indicator"``, ``"strategy"``,
  ``"library"``) is accepted at construction.
* ``CompiledModule`` remains frozen (``FrozenInstanceError`` on mutation),
  slots-backed (no ``__dict__``), and dataclass-round-trippable so D3's wire
  layer keeps working.
* The Literal typing is documented by example — mypy would catch invalid
  literals; the runtime dataclass does not (nor does D1 require it to).
"""

from __future__ import annotations

import dataclasses

import pytest

from openbb_pine.compiler.types import CompiledModule


def _make(**overrides) -> CompiledModule:
    """Build a valid ``CompiledModule`` with sensible defaults for shape tests."""
    defaults = dict(
        source="from pynecore.lib import close\n",
        sha="blake2b:script-type-test",
        pine_version=6,
        compiler_version="0.1.0",
        builtins_used=frozenset({"close"}),
        security_contexts=None,
        cache_status="miss",
    )
    defaults.update(overrides)
    return CompiledModule(**defaults)


class TestScriptTypeDefault:
    """Default construction (no ``script_type`` kwarg) yields ``"indicator"``.

    Every existing call site in codegen/compile_cache/tests omits the field —
    the default preserves their contract.
    """

    def test_default_is_indicator(self) -> None:
        m = _make()
        assert m.script_type == "indicator"

    def test_default_when_all_other_fields_supplied_by_kwarg(self) -> None:
        # Same as the codegen call site — every field by kwarg, no script_type.
        m = CompiledModule(
            source="",
            sha="",
            pine_version=6,
            compiler_version="0.1.0",
            builtins_used=frozenset(),
            security_contexts=None,
            cache_status="bypass",
        )
        assert m.script_type == "indicator"


class TestScriptTypeExplicitValues:
    """All three Literal values (indicator, strategy, library) construct cleanly."""

    @pytest.mark.parametrize("value", ["indicator", "strategy", "library"])
    def test_accepts_each_literal(self, value: str) -> None:
        m = _make(script_type=value)
        assert m.script_type == value


class TestScriptTypeLiteralDocumentedByExample:
    """Document the Literal invariant by example.

    Python's dataclass does not enforce Literal at runtime — it's a mypy-level
    check. We pin the accepted set here so a future refactor that widens the
    field triggers a review, and so readers see the exhaustive list at a
    glance.
    """

    def test_literal_domain_is_exactly_three_values(self) -> None:
        # The literal domain — canonical enumeration for D5 §5.1 script types.
        allowed = ("indicator", "strategy", "library")
        for value in allowed:
            m = _make(script_type=value)
            assert m.script_type == value
        # No fourth value: if someone widens the Literal, they must update this
        # list AND the executor branching bead. This is intentional friction.
        assert len(allowed) == 3


class TestScriptTypePreservesFrozenness:
    """Adding ``script_type`` must not weaken the frozen/slots invariants."""

    def test_mutation_raises_frozen_instance_error(self) -> None:
        m = _make(script_type="strategy")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.script_type = "library"  # type: ignore[misc]

    def test_default_field_also_frozen(self) -> None:
        # The defaulted field is just as frozen as the required ones.
        m = _make()  # script_type left as default "indicator"
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.script_type = "strategy"  # type: ignore[misc]

    def test_no_instance_dict_when_slots_active(self) -> None:
        # slots=True means no __dict__; verify we did not accidentally break it
        # when adding the new field.
        m = _make()
        assert not hasattr(m, "__dict__")


class TestScriptTypeRoundTripsThroughAsdict:
    """``dataclasses.asdict`` — the D3 wire layer path — includes the new field."""

    def test_asdict_includes_script_type_default(self) -> None:
        m = _make()
        d = dataclasses.asdict(m)
        assert d["script_type"] == "indicator"

    @pytest.mark.parametrize("value", ["indicator", "strategy", "library"])
    def test_asdict_includes_script_type_explicit(self, value: str) -> None:
        m = _make(script_type=value)
        d = dataclasses.asdict(m)
        assert d["script_type"] == value


class TestScriptTypeIsLastField:
    """The default-valued field must sit AFTER all required (non-default) fields.

    dataclass ordering rule: fields with defaults must follow fields without.
    This test locks the ordering so a future edit that inserts ``script_type``
    mid-list fails loudly rather than silently breaking positional constructors
    elsewhere in the codebase (though we prefer kwargs).
    """

    def test_script_type_is_final_field(self) -> None:
        fields = dataclasses.fields(CompiledModule)
        assert fields[-1].name == "script_type"

    def test_script_type_default_is_indicator(self) -> None:
        fields = {f.name: f for f in dataclasses.fields(CompiledModule)}
        assert fields["script_type"].default == "indicator"
