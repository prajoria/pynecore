"""Pine type system + ``CompiledModule`` codegen contract.

Authoritative source: D1 §3.1 (``CompiledModule``) and §4.1-§4.2 (``PineType``
+ qualifier lattice). The af08128d3 cross-doc consolidation makes
``builtins_used``, ``security_contexts``, and ``cache_status`` part of the
codegen contract — D2 (runtime) and D3 (platform) read them.

Pine's type system is two orthogonal axes per D1 §4:

* **Value type** — int, float, bool, string, color, line, label, box, table,
  polyline, linefill, array, matrix, map, UDT, tuple, na, unknown.
* **Qualifier** — ``const | input | simple | series``; only direction of
  implicit promotion is up the lattice.

We encode types as a ``(qualifier, inner)`` pair (D1 §4.1). The ``inner`` is
one of the ``InnerType`` subclasses below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openbb_pine.errors import PineTypeError

__all__ = [
    "Qualifier",
    "PineType",
    "InnerType",
    "Scalar",
    "Reference",
    "ArrayT",
    "MatrixT",
    "MapT",
    "UDT",
    "FunctionT",
    "TupleT",
    "NaT",
    "UnknownT",
    "AnyT",
    "can_promote",
    "unify",
    "unify_inner",
    "inner_compatible",
    "SecurityContext",
    "CompiledModule",
]

Qualifier = Literal["const", "input", "simple", "series"]


# ---------------------------------------------------------------------------
# Inner type variants (D1 §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InnerType:
    """Marker base for the inner value-type axis."""


@dataclass(frozen=True, slots=True)
class Scalar(InnerType):
    """Scalar values: int, float, bool, string, color."""

    kind: Literal["int", "float", "bool", "string", "color"]


@dataclass(frozen=True, slots=True)
class Reference(InnerType):
    """Mutable drawing references: line, label, box, table, polyline, linefill."""

    kind: Literal["line", "label", "box", "table", "polyline", "linefill"]


@dataclass(frozen=True, slots=True)
class ArrayT(InnerType):
    element: "PineType"


@dataclass(frozen=True, slots=True)
class MatrixT(InnerType):
    element: "PineType"


@dataclass(frozen=True, slots=True)
class MapT(InnerType):
    key: "PineType"
    value: "PineType"


@dataclass(frozen=True, slots=True)
class UDT(InnerType):
    """User-defined type (``type Foo``)."""

    name: str


@dataclass(frozen=True, slots=True)
class FunctionT(InnerType):
    params: tuple["PineType", ...]
    return_type: "PineType"
    type_params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TupleT(InnerType):
    elements: tuple["PineType", ...]


@dataclass(frozen=True, slots=True)
class NaT(InnerType):
    """The polymorphic ``na`` value, pre-unification."""


@dataclass(frozen=True, slots=True)
class UnknownT(InnerType):
    """Fresh inference variable; resolved during type checking."""

    var_id: int


@dataclass(frozen=True, slots=True)
class AnyT(InnerType):
    """A polymorphic *bound* type variable used by signatures whose return
    type matches an incoming operand's type (e.g. ``request.security``'s
    ``expression`` argument and return type share the shape).

    Distinct from :class:`UnknownT` — ``UnknownT(-1)`` is used at four
    fallback sites in the type checker as "compiler could not infer";
    conflating "polymorphic" and "unresolved" causes silent unification of
    unrelated code paths under structural equality (``UnknownT(-1) ==
    UnknownT(-1)`` is True). ``AnyT`` unifies with anything by design (same
    ``na``-style propagation lattice), so it's safe to embed in the shared
    ``_SERIES_ANY`` sentinel without aliasing to unrelated ``UnknownT``
    fallbacks (see PR #322 review — comment ID 3522847216).
    """


# ---------------------------------------------------------------------------
# PineType pair (D1 §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PineType:
    """The ``(qualifier, inner)`` pair that codegen forks on (D1 §4.1)."""

    qualifier: Qualifier
    inner: InnerType


# ---------------------------------------------------------------------------
# Qualifier lattice (D1 §4.2)
# ---------------------------------------------------------------------------


_RANK: dict[str, int] = {"const": 0, "input": 1, "simple": 2, "series": 3}


def can_promote(src: Qualifier, dst: Qualifier) -> bool:
    """Return True iff ``src`` may implicitly promote to ``dst`` (D1 §4.2)."""
    return _RANK[src] <= _RANK[dst]


def inner_compatible(a: InnerType, b: InnerType) -> bool:
    """Return True iff two inner types can unify (structural equality for now)."""
    # Na unifies with anything (PT006: na propagates the other side's type).
    if isinstance(a, NaT) or isinstance(b, NaT):
        return True
    # AnyT is a bound polymorphic sentinel (see docstring on AnyT); it
    # unifies with anything, mirroring the na-propagation lattice.
    if isinstance(a, AnyT) or isinstance(b, AnyT):
        return True
    return a == b


def unify_inner(a: InnerType, b: InnerType) -> InnerType:
    """Unify two compatible inner types; ``na`` collapses to the other side."""
    if isinstance(a, NaT):
        return b
    if isinstance(b, NaT):
        return a
    if isinstance(a, AnyT):
        return b
    if isinstance(b, AnyT):
        return a
    return a


def unify(a: PineType, b: PineType) -> PineType:
    """Pine's type join: promote both to the max-qualifier supertype (D1 §4.2).

    Raises ``PineTypeError`` if the inner types are incompatible.
    """
    if not inner_compatible(a.inner, b.inner):
        raise PineTypeError(f"cannot unify {a!r} with {b!r}")
    return PineType(
        qualifier=max(a.qualifier, b.qualifier, key=_RANK.__getitem__),
        inner=unify_inner(a.inner, b.inner),
    )


# ---------------------------------------------------------------------------
# Compiled-module contract (D1 §3.1 + af08128d3 consolidation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SecurityContext:
    """A lowered ``request.security(symbol, timeframe, expr)`` directive.

    Codegen rewrites each such call into a structured ``SecurityContext`` so
    D2 §5.1 step 2 can set up multi-symbol dispatch ahead of
    ``ScriptRunner.run_iter()``.

    The ``dynamic_*`` flags implement D5 §4.4's fully-dynamic
    symbol/timeframe support: when C3 cannot statically resolve the symbol
    or timeframe argument (e.g. ``request.security(syminfo.ticker, tf,
    expr)``), it sets the corresponding flag so the runtime dispatcher
    (D2's ``security_dispatcher``) knows to fall back to lazy per-bar fetch
    instead of prefetching. Documented perf caveat per D5 §4.4:
    dynamic-symbol scripts run 5-10× slower than static-symbol.

    Both flags default to ``False`` to preserve backward compatibility with
    existing constructions like
    ``SecurityContext(symbol="AAPL", timeframe="1D", expr="close")``.
    """

    symbol: str
    timeframe: str
    expr: str  # serialized lowered expression; D2 reads opaquely
    dynamic_symbol: bool = False
    """True when C3 could not statically resolve the symbol (e.g.
    ``request.security(syminfo.ticker, ...)``). Signals runtime to defer
    to lazy per-bar fetch (D5 §4.4)."""
    dynamic_timeframe: bool = False
    """True when C3 could not statically resolve the timeframe. Same
    runtime consequence as ``dynamic_symbol`` (D5 §4.4)."""


@dataclass(frozen=True, slots=True)
class CompiledModule:
    """The codegen contract — fields downstream D2 / D3 consume (D1 §3.1).

    Mutating this shape is a D1 decision; consumers downstream may not add
    fields. The af08128d3 cross-doc consolidation makes ``builtins_used``,
    ``security_contexts``, and ``cache_status`` part of the contract.
    """

    source: str
    """The Python text written under D1 §6's cache key."""

    sha: str
    """blake2b digest: ``source ‖ params ‖ compiler_version ‖ pine_version``."""

    pine_version: int
    """5 or 6 — the source's ``//@version=`` pragma."""

    compiler_version: str
    """``openbb_pine.__version__`` at the time of compile."""

    builtins_used: frozenset[str]
    """Pine builtin identifiers the script references (e.g. ``{"ta.sma"}``)."""

    security_contexts: dict[str, SecurityContext] | None
    """Per-call security context map, or None for scripts without ``request.security``."""

    cache_status: Literal["hit", "miss", "bypass"]
    """Was this compilation a cache hit, miss, or bypass? D2 §6.3 reads this."""

    script_type: Literal["indicator", "strategy", "library"] = "indicator"
    """Top-level Pine declaration kind (D5 §5.1 — script-type detection).

    Defaults to ``"indicator"`` so existing constructor call sites (codegen,
    compile_cache, tests) that pre-date the M2 strategy engine keep working
    unchanged. Codegen will overwrite this based on which top-level declaration
    fires (``indicator(...)`` / ``strategy(...)`` / ``library(...)``); the
    executor forks on it to enable strategy vs. indicator branches.
    """
