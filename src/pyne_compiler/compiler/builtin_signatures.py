"""Phase-1 stub registry of Pine builtin signatures (D1 §3.1 contract).

C3's type checker pattern-matches a CallExpr's resolved name (e.g. ``"ta.sma"``)
against :data:`BUILTIN_SIGNATURES` here so it can:

1. Type-check the argument list against the formal parameter types.
2. Annotate the call's return type onto the IR sidecar.
3. Add the qualified name to :class:`CompiledModule.builtins_used` so the wild-
   corpus coverage metric (PRD §3.4 L0.5) credits the script for touching it,
   regardless of whether we currently emit code for it.

Per the parent's bead spec: this is a **stub** — full signatures land as the
S-beads (0e9.5.16-51) bring each builtin into the pynecore.lib bridge. Until
then, every entry is marked ``notes="STUB"`` so future workers can find them
by grep. The registry **only** needs to be precise enough that:

- The 36 Phase-1 builtins (29 ta.* + 7 math.* per PRD §3.2) resolve cleanly.
- Essential surfaces (input.*, plot family, na/nz, OHLCV sources) resolve.
- :func:`is_builtin_namespace` distinguishes "unsupported-but-real-Pine" from
  "totally-unknown identifier" so C3 can pick the right error class:
    * `ta.ichimoku` -> not in registry but namespace 'ta' is real ->
      PineUnsupportedBuiltinError + add to builtins_used.
    * `foobar.baz` -> namespace 'foobar' unknown -> PineTypeError(rule="undefined").

Design hot spot: the per-builtin signature **shape** is the same as what
the S-bead replacement will use. So when 0e9.5.16 lands ta.sma's *real*
PyneCore-compatible signature, the diff against this file is a one-line
override of the same key — no schema migration.
"""

from __future__ import annotations

from dataclasses import dataclass

from openbb_pine.compiler.types import PineType, Scalar, TupleT

__all__ = [
    "Signature",
    "BUILTIN_SIGNATURES",
    "PINE_NAMESPACES",
    "lookup",
    "is_builtin_namespace",
]


# ---------------------------------------------------------------------------
# Signature dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signature:
    """A callable Pine builtin's signature.

    ``args`` is an ordered tuple of ``(name, type)`` pairs. Positional vs
    keyword discipline is enforced by C3 at call sites — Pine permits both
    forms for almost every builtin parameter. For value-like builtins (e.g.
    ``close``, ``open``), ``args=()`` and ``returns`` carries the value's
    inferred type so C3 can resolve a bare ``Name`` against the registry.

    ``version`` is the Pine version the signature appeared in (5 or 6); the
    Phase-1 stub uses ``6`` uniformly. ``notes`` is the audit channel — every
    STUB-marker entry is a future S-bead's responsibility to refine.
    """

    args: tuple[tuple[str, PineType], ...]
    returns: PineType
    version: int = 6
    notes: str | None = "STUB"


# ---------------------------------------------------------------------------
# Common type shorthands
# ---------------------------------------------------------------------------


_SERIES_FLOAT = PineType(qualifier="series", inner=Scalar(kind="float"))
_SERIES_INT = PineType(qualifier="series", inner=Scalar(kind="int"))
_SERIES_BOOL = PineType(qualifier="series", inner=Scalar(kind="bool"))
_SIMPLE_INT = PineType(qualifier="simple", inner=Scalar(kind="int"))
_SIMPLE_FLOAT = PineType(qualifier="simple", inner=Scalar(kind="float"))
_SIMPLE_BOOL = PineType(qualifier="simple", inner=Scalar(kind="bool"))
_SIMPLE_STRING = PineType(qualifier="simple", inner=Scalar(kind="string"))
_INPUT_INT = PineType(qualifier="input", inner=Scalar(kind="int"))
_INPUT_FLOAT = PineType(qualifier="input", inner=Scalar(kind="float"))
_INPUT_BOOL = PineType(qualifier="input", inner=Scalar(kind="bool"))
_INPUT_STRING = PineType(qualifier="input", inner=Scalar(kind="string"))
_CONST_COLOR = PineType(qualifier="const", inner=Scalar(kind="color"))


def _src_length() -> tuple[tuple[str, PineType], ...]:
    """Common (src, length) shape — used by ta.sma/ema/rma/wma/atr/etc."""
    return (("src", _SERIES_FLOAT), ("length", _SIMPLE_INT))


# ---------------------------------------------------------------------------
# The registry — 36 Phase-1 builtins + essential surfaces
# ---------------------------------------------------------------------------


BUILTIN_SIGNATURES: dict[str, Signature] = {
    # === 29 ta.* builtins per PRD §3.2 ===========================================
    "ta.sma":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    "ta.ema":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    "ta.rma":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    "ta.wma":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    "ta.vwma":       Signature(args=_src_length(), returns=_SERIES_FLOAT),
    "ta.swma":       Signature(args=(("src", _SERIES_FLOAT),), returns=_SERIES_FLOAT),
    "ta.rsi":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    # MACD returns a 3-tuple ``(macd_line, signal_line, hist)``. Wave 5B-1
    # lifted the stub to its real tuple shape; C3's tuple-destructuring path
    # (type_checker._visit_subscript / assignment tuple unpack) reads
    # TupleT.elements to route each element to its LHS binding.
    "ta.macd":       Signature(
        args=(
            ("src", _SERIES_FLOAT),
            ("fastlen", _SIMPLE_INT),
            ("slowlen", _SIMPLE_INT),
            ("siglen", _SIMPLE_INT),
        ),
        returns=PineType(
            qualifier="series",
            inner=TupleT(elements=(_SERIES_FLOAT, _SERIES_FLOAT, _SERIES_FLOAT)),
        ),
        notes="IMPLEMENTED",
    ),
    # Bollinger Bands returns a 3-tuple ``(basis, upper, lower)``. Wave 5B-3
    # (bead 0e9.5.22) lifted the stub to its real tuple shape; same pattern
    # as Wave 5B-1's ``ta.macd`` — C3's tuple-destructuring path reads
    # TupleT.elements to route each element to its LHS binding.
    "ta.bb":         Signature(
        args=(
            ("src", _SERIES_FLOAT),
            ("length", _SIMPLE_INT),
            ("mult", _SIMPLE_FLOAT),
        ),
        returns=PineType(
            qualifier="series",
            inner=TupleT(elements=(_SERIES_FLOAT, _SERIES_FLOAT, _SERIES_FLOAT)),
        ),
        notes="IMPLEMENTED",
    ),
    "ta.stoch":      Signature(
        args=(
            ("src", _SERIES_FLOAT),
            ("high", _SERIES_FLOAT),
            ("low", _SERIES_FLOAT),
            ("length", _SIMPLE_INT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.cci":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    "ta.mfi":        Signature(args=_src_length(), returns=_SERIES_FLOAT, notes="IMPLEMENTED"),
    # ta.adx is NOT in the PRD §3.2 Phase-1 list — Wave 5B-2 (bead 0e9.5.26)
    # added its registry entry alongside the bridge because PyneCore has no
    # standalone ``adx`` primitive; the bridge synthesises it from
    # ``ta.dmi(dilen, adxlen)[2]``. Signature deviation from most ta.*: no
    # ``src`` parameter — Pine's ``ta.adx`` takes only the two length
    # parameters and reads ``high``/``low`` off the primary OHLCV stream
    # transitively through ``ta.dmi``.
    "ta.adx":        Signature(
        args=(
            ("dilen", _SIMPLE_INT),
            ("adxlen", _SIMPLE_INT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.atr":        Signature(
        # Wave 5B-3 (bead 0e9.5.23). Signature deviation from most ta.*:
        # NO ``src`` parameter — ``ta.atr`` reads ``high``/``low``/``close``
        # transitively through ``ta.tr`` and applies Wilder's ``rma`` on top.
        args=(("length", _SIMPLE_INT),),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.tr":         Signature(
        # Wave 5B-3 (bead 0e9.5.38). PyneCore models ``tr`` as a
        # ``@module_property`` so both ``ta.tr`` (bare identifier) and
        # ``ta.tr(handle_na)`` compile. Registry declares the callable
        # form with ``handle_na=False`` — the default matches PyneCore.
        args=(("handle_na", _SIMPLE_BOOL),),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.stdev":      Signature(
        # Wave 5B-3 (bead 0e9.5.34). Adds the third ``biased`` parameter
        # matching PyneCore's ``stdev(source, length, biased=True)``. The
        # ``biased=True`` default (population stdev) matches Pine's
        # documented behaviour; callers who want the sample form pass
        # ``biased=False``.
        args=(("src", _SERIES_FLOAT), ("length", _SIMPLE_INT), ("biased", _SIMPLE_BOOL)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.variance":   Signature(args=_src_length(), returns=_SERIES_FLOAT),
    # ta.obv is NOT in the PRD §3.2 Phase-1 list — Wave 5B-3 (bead 0e9.5.28)
    # added its registry entry alongside the bridge because PyneCore models
    # ``obv`` as a zero-arg ``@module_property`` (Pine scripts write
    # ``plot(ta.obv)`` — a bare identifier). Registry declares the callable
    # form ``args=()`` matching how the bridge is invoked.
    "ta.obv":        Signature(
        args=(),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    # ta.vwap is NOT in the PRD §3.2 Phase-1 list — Wave 5B-3 (bead 0e9.5.29)
    # added its registry entry alongside the bridge. PyneCore's public
    # signature is ``vwap(source, anchor=None, stdev_mult=None)``; the
    # scalar return-type covers the vast majority of Pine calls
    # (``ta.vwap(hlc3)``, ``ta.vwap(close)``). A future stub lift can add
    # the tuple return for the ``stdev_mult`` overload without a bridge
    # change — the bridge already accepts and forwards ``stdev_mult``.
    "ta.vwap":       Signature(
        args=(
            ("source", _SERIES_FLOAT),
            ("anchor", _SIMPLE_BOOL),
            ("stdev_mult", _SIMPLE_FLOAT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    # ta.sar is NOT in the PRD §3.2 Phase-1 list — Wave 5B-3 (bead 0e9.5.39)
    # added its registry entry alongside the bridge. Signature deviation
    # from most ta.*: no ``src`` — ``ta.sar`` reads ``high``/``low`` off
    # the primary OHLCV stream. The parameter named ``max`` shadows
    # Python's builtin — preserved to match PyneCore's public signature
    # so keyword-form calls resolve.
    "ta.sar":        Signature(
        args=(
            ("start", _SIMPLE_FLOAT),
            ("inc", _SIMPLE_FLOAT),
            ("max", _SIMPLE_FLOAT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.highest":    Signature(
        # Wave 5B-4 (bead 0e9.5.32). Bridge lives at openbb_pine.stdlib.ta.highest.
        # PyneCore names the first arg ``source``; the registry mirrors that
        # so keyword-form Pine calls resolve. PyneCore also defines a
        # single-arg overload ``ta.highest(length)`` (defaults source to
        # ``high``); Phase-1 stub covers the two-arg form only — codegen's
        # arg-fill handles the sugar form in a later phase.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.lowest":     Signature(
        # Wave 5B-4 (bead 0e9.5.33). Symmetric to ``ta.highest``.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.barssince":  Signature(
        # Wave 5B-4 (bead 0e9.5.44). PyneCore param name is ``condition``
        # (the previous stub used ``cond``); flipping to the PyneCore name
        # so keyword-form calls resolve through the bridge.
        args=(("condition", _SERIES_BOOL),),
        returns=_SERIES_INT,
        notes="IMPLEMENTED",
    ),
    "ta.crossover":  Signature(
        # Wave 5B-4 (bead 0e9.5.30). PyneCore uses ``source1``/``source2``;
        # the registry preserves the existing names (no drift).
        args=(("source1", _SERIES_FLOAT), ("source2", _SERIES_FLOAT)),
        returns=_SERIES_BOOL,
        notes="IMPLEMENTED",
    ),
    "ta.crossunder": Signature(
        # Wave 5B-4 (bead 0e9.5.31). Symmetric to ``ta.crossover``.
        args=(("source1", _SERIES_FLOAT), ("source2", _SERIES_FLOAT)),
        returns=_SERIES_BOOL,
        notes="IMPLEMENTED",
    ),
    "ta.cross":      Signature(
        args=(("source1", _SERIES_FLOAT), ("source2", _SERIES_FLOAT)),
        returns=_SERIES_BOOL,
    ),
    "ta.change":     Signature(
        # Wave 5B-4 (bead 0e9.5.35). PyneCore's public signature is
        # ``change(source, length=1)``; the stub had only ``src`` (missing
        # the trailing ``length``). Registry does not model per-arg
        # defaults — the ``length=1`` default lives in the bridge; C3's
        # stub type check tolerates the omitted trailing positional. Param
        # name flipped ``src`` → ``source`` for keyword-form calls.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.mom":        Signature(
        # Wave 5B-4 (bead 0e9.5.36). Semantically identical to
        # ``ta.change(source, length)`` — PyneCore's ``ta.mom`` is a one-line
        # delegation to ``ta.change``. Param name flipped ``src`` →
        # ``source`` for PyneCore-native keyword-form calls.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.roc":        Signature(
        # Wave 5B-4 (bead 0e9.5.37). Rate of change:
        # ``100 * (source - source[length]) / source[length]``. Param name
        # flipped ``src`` → ``source``.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.cum":        Signature(
        # Wave 5B-4 (bead 0e9.5.43). No ``length`` arg; running total from
        # series start. Param name flipped ``src`` → ``source`` to match
        # PyneCore.
        args=(("source", _SERIES_FLOAT),),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.dev":        Signature(args=_src_length(), returns=_SERIES_FLOAT),
    "ta.linreg":     Signature(
        # Wave 5B-4 (bead 0e9.5.40). ``offset`` is REQUIRED (no default) in
        # both PyneCore and the Pine reference — the common
        # "current-bar value" call is ``ta.linreg(source, length, 0)``. Param
        # name flipped ``src`` → ``source`` for PyneCore-native keyword-form
        # calls.
        args=(
            ("source", _SERIES_FLOAT),
            ("length", _SIMPLE_INT),
            ("offset", _SIMPLE_INT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "ta.median":     Signature(
        # Wave 5B-4 (bead 0e9.5.41). Rolling median via PyneCore's two-heap
        # implementation. ``length == 1`` is a short-circuit that returns
        # ``source`` unchanged.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    # ta.percentile_linear_interpolation — Wave 5B-4 (bead 0e9.5.42). Percentile
    # with linear interpolation between adjacent ranks. Pine signature:
    # (source, length, percentage) where percentage is 0..100. This builtin
    # was NOT in the PRD §3.2 Phase-1 enumeration; Wave 5B-4 adds the entry
    # alongside the bridge because the parent bead spec 0e9.5.42 names it
    # explicitly. Third arg is ``percentage`` (NOT ``percentile``) per
    # PyneCore.
    "ta.percentile_linear_interpolation": Signature(
        args=(
            ("source", _SERIES_FLOAT),
            ("length", _SIMPLE_INT),
            ("percentage", _SIMPLE_FLOAT),
        ),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    # === math.* builtins per PRD §3.2 ============================================
    # Bridges landed in Wave 5B group 5 (S-beads 0e9.5.{45..51}).
    # ``math.max`` / ``math.min`` are declared as 2-ary here for C3's stub
    # type check; PyneCore (and the bridge) accept N-ary via ``*numbers``,
    # and C3 tolerates trailing positionals when checking against a stub
    # (see :func:`_check_call_args`).
    "math.abs":      Signature(
        args=(("number", _SERIES_FLOAT),),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.sqrt":     Signature(
        args=(("number", _SERIES_FLOAT),),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.log":      Signature(args=(("number", _SERIES_FLOAT),), returns=_SERIES_FLOAT),
    "math.exp":      Signature(args=(("number", _SERIES_FLOAT),), returns=_SERIES_FLOAT),
    "math.max":      Signature(
        args=(("a", _SERIES_FLOAT), ("b", _SERIES_FLOAT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.min":      Signature(
        args=(("a", _SERIES_FLOAT), ("b", _SERIES_FLOAT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.pow":      Signature(
        args=(("base", _SERIES_FLOAT), ("exponent", _SERIES_FLOAT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.round":    Signature(
        # ``precision`` is optional; PyneCore treats an omitted precision as
        # NA(int) (defaulting to integer rounding). The stub declares it as
        # a normal simple<int> arg; C3's positional-check tolerates the
        # single-arg call because we do not enforce arity min for the stub.
        args=(("number", _SERIES_FLOAT), ("precision", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    "math.sum":      Signature(
        # Rolling sum over ``length`` bars — same shape as ta.sma/ta.rma.
        args=(("source", _SERIES_FLOAT), ("length", _SIMPLE_INT)),
        returns=_SERIES_FLOAT,
        notes="IMPLEMENTED",
    ),
    # === input.* constructors ====================================================
    "input.int":     Signature(
        args=(("defval", _SIMPLE_INT),), returns=_INPUT_INT,
    ),
    "input.float":   Signature(
        args=(("defval", _SIMPLE_FLOAT),), returns=_INPUT_FLOAT,
    ),
    "input.bool":    Signature(
        args=(("defval", _SIMPLE_BOOL),), returns=_INPUT_BOOL,
    ),
    "input.string":  Signature(
        args=(("defval", _SIMPLE_STRING),), returns=_INPUT_STRING,
    ),
    "input.source":  Signature(
        args=(("defval", _SERIES_FLOAT),), returns=_SERIES_FLOAT,
    ),
    # === plot family ============================================================
    # plot/plotshape/hline are statements in Pine — they return na (no
    # meaningful expression value). The stub returns simple<float>; codegen
    # ignores the return when emitted in statement position.
    "plot":          Signature(
        args=(("series", _SERIES_FLOAT),), returns=_SIMPLE_FLOAT,
    ),
    "plotshape":     Signature(
        args=(("series", _SERIES_BOOL),), returns=_SIMPLE_FLOAT,
    ),
    "hline":         Signature(
        args=(("price", _SIMPLE_FLOAT),), returns=_SIMPLE_FLOAT,
    ),
    # === na / nz polymorphic ====================================================
    "na":            Signature(
        args=(("x", _SERIES_FLOAT),), returns=_SERIES_BOOL,
    ),
    "nz":            Signature(
        args=(("x", _SERIES_FLOAT),), returns=_SERIES_FLOAT,
    ),
    # === Color constants (a few key ones; rest live in color.* namespace) ======
    "color.red":     Signature(args=(), returns=_CONST_COLOR),
    "color.green":   Signature(args=(), returns=_CONST_COLOR),
    "color.blue":    Signature(args=(), returns=_CONST_COLOR),
    "color.black":   Signature(args=(), returns=_CONST_COLOR),
    "color.white":   Signature(args=(), returns=_CONST_COLOR),
    "color.orange":  Signature(args=(), returns=_CONST_COLOR),
    "color.yellow":  Signature(args=(), returns=_CONST_COLOR),
    # === OHLCV sources (treated as zero-arg "constructors") =====================
    # PyneCore injects these at runtime; the codegen layer references them as
    # globals via GLOBAL_NAME_ALLOWLIST. The type checker resolves them here.
    "close":         Signature(args=(), returns=_SERIES_FLOAT),
    "open":          Signature(args=(), returns=_SERIES_FLOAT),
    "high":          Signature(args=(), returns=_SERIES_FLOAT),
    "low":           Signature(args=(), returns=_SERIES_FLOAT),
    "volume":        Signature(args=(), returns=_SERIES_FLOAT),
    "time":          Signature(args=(), returns=_SERIES_INT),
    "hl2":           Signature(args=(), returns=_SERIES_FLOAT),
    "hlc3":          Signature(args=(), returns=_SERIES_FLOAT),
    "ohlc4":         Signature(args=(), returns=_SERIES_FLOAT),
    "bar_index":     Signature(args=(), returns=_SERIES_INT),
}


# ---------------------------------------------------------------------------
# Recognised Pine namespaces
# ---------------------------------------------------------------------------


# These are the namespace prefixes the Pine reference manual documents. When
# C3 sees an Attribute like ``ta.ichimoku`` where ``ta`` is in this set but
# the full name isn't in BUILTIN_SIGNATURES, it raises
# PineUnsupportedBuiltinError (PU0xx) — but still adds ``"ta.ichimoku"`` to
# CompiledModule.builtins_used so wild-corpus coverage (PRD §3.4 L0.5)
# attributes the request.
PINE_NAMESPACES: frozenset[str] = frozenset({
    "ta", "math", "input", "array", "matrix", "map", "strategy",
    "request", "library", "line", "label", "box", "table",
    "syminfo", "barstate", "session", "alert", "currency",
    "dayofweek", "display", "earnings", "extend", "fixnan",
    "location", "month", "na", "plot", "price", "runtime",
    "sym", "year", "color", "chart", "timeframe", "string",
    "polyline", "linefill",
})


# ---------------------------------------------------------------------------
# Public lookup surface
# ---------------------------------------------------------------------------


def lookup(name: str) -> Signature | None:
    """Resolve a qualified Pine builtin name to its signature, or return None.

    ``name`` is the fully-qualified Pine identifier (e.g. ``"ta.sma"``,
    ``"input.int"``, ``"close"``). Returns the registered :class:`Signature`
    when present and :data:`None` otherwise.
    """
    return BUILTIN_SIGNATURES.get(name)


def is_builtin_namespace(prefix: str) -> bool:
    """Return True iff ``prefix`` names a Pine namespace (e.g. ``"ta"``,
    ``"math"``, ``"input"``). C3 uses this to distinguish

    - unsupported-but-real-Pine (e.g. ``ta.ichimoku``) → raise PU0xx, still
      record in :class:`CompiledModule.builtins_used` so the wild-corpus
      coverage metric credits the script.
    - totally-unknown (e.g. ``zoltan.xyz``) → raise PineTypeError with
      ``rule="undefined"``.

    Case-sensitive — Pine namespaces are lowercase. The match must be exact.
    """
    return prefix in PINE_NAMESPACES
