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

from openbb_pine.compiler.types import AnyT, NaT, PineType, Scalar, TupleT

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
    kwargs: dict[str, PineType] | None = None
    """Optional keyword-only parameters (D5 §7.2). Distinct from ``args``
    which enumerates positional-or-keyword parameters. C3 uses this to
    type-check keyword-only slots like ``request.security(..., gaps=?,
    lookahead=?)``. Defaults to ``None`` so pre-D5 signatures keep working
    unchanged — every existing entry only uses ``args``.
    """


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
# _CONST_BOOL / _CONST_STRING / _SERIES_ANY — shared shorthands added by
# bead y86 (request.security signature, D5 §7.2). _CONST_STRING is used by
# h14's strategy.entry id/direction args; _CONST_BOOL by request.security's
# gaps/lookahead kwargs; _SERIES_ANY by request.security's polymorphic
# ``expression`` shape. See :class:`AnyT` for why _SERIES_ANY uses a dedicated
# sentinel instead of ``UnknownT(-1)`` (four other type_checker fallback sites
# use ``UnknownT(-1)`` for "compiler could not infer"; structural equality
# would silently unify unrelated code paths).
_CONST_BOOL = PineType(qualifier="const", inner=Scalar(kind="bool"))
_CONST_STRING = PineType(qualifier="const", inner=Scalar(kind="string"))
_SERIES_ANY = PineType(qualifier="series", inner=AnyT())
# Pseudo-void return marker — Pine's strategy.* order-management calls don't
# yield a meaningful expression value. Modelled as ``const<na>`` so a script
# doing ``x = strategy.entry(...)`` binds ``x`` to ``const<NaT>`` (semantically
# "no value") rather than the misleading ``simple<float>`` a plot-style alias
# would produce. Codegen ignores the return when the call sits in statement
# position. Mirror of ``type_checker._NA``. This is a distinct sentinel — an
# alias like ``_VOID = _SIMPLE_FLOAT`` would silently change meaning if a
# future refactor ever touched ``_SIMPLE_FLOAT`` and would let an assignment
# like ``x = strategy.entry(...)`` claim ``x: simple<float>`` (a value that
# is never usable).
_VOID = PineType(qualifier="const", inner=NaT())


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
    # ---------------------------------------------------------------------------
    # request.security signature (bead 0e9.6.y86 — D5 §4.1, §7.2)
    # ---------------------------------------------------------------------------
    # Static-symbol form: ``request.security("SPY", "1D", close)`` — C3
    # populates one SecurityContext per call site (dynamic_symbol=False,
    # dynamic_timeframe=False). Fully-dynamic form per D5 §4.4:
    # ``request.security(syminfo.ticker, "1D", close)`` — C3 flags
    # dynamic_symbol=True; runtime falls back to lazy per-bar fetch.
    # Return type is _SERIES_ANY so the polymorphic ``expression`` shape
    # (series<float> for ``close``, series<bool> for a comparison, etc.)
    # flows through the type checker without a rejection.
    "request.security": Signature(
        args=(
            ("symbol",     _SIMPLE_STRING),
            ("timeframe",  _SIMPLE_STRING),
            ("expression", _SERIES_ANY),
        ),
        kwargs={
            "gaps":      _CONST_BOOL,
            "lookahead": _CONST_BOOL,
        },
        returns=_SERIES_ANY,
        notes="IMPLEMENTED",
    ),

    # strategy.* signatures (bead 0e9.6.h14 — D5 §7.2)
    # ---------------------------------------------------------------------------
    #
    # Order-management calls (``strategy.entry`` etc.) are declared here so C3
    # type-checks arg types and populates ``builtins_used`` with the qualified
    # names. Codegen for these entries is deferred — bead ``aeh`` will lift
    # the ``PineUnsupportedFeatureError`` PF010 stub in ``codegen.py`` and
    # wire the calls through the ``openbb_pine.stdlib.strategy`` bridge. Until
    # then a script that uses these WILL type-check cleanly but fail at
    # codegen with a clear PF010 error pointing at the M2 tracking issue.
    #
    # Signature shape notes:
    #
    # * The :class:`Signature` dataclass has no distinct ``kwargs`` field.
    #   ``args`` is the union of positional AND keyword-only parameters.
    #   :func:`type_checker._check_call_args` walks positional args by index,
    #   then keyword args by name-match against the same tuple — so declaring
    #   ``qty`` / ``limit`` / ``stop`` / etc. as trailing entries yields the
    #   correct kwarg-binding behaviour with no schema change.
    #
    # * Void-returning calls use ``_VOID`` (an alias for ``_SIMPLE_FLOAT``)
    #   matching the ``plot`` / ``plotshape`` / ``hline`` convention; codegen
    #   ignores the return when the call sits in statement position.
    #
    # * ``strategy.long`` / ``strategy.short`` / ``strategy.fixed`` /
    #   ``strategy.cash`` / ``strategy.percent_of_equity`` are declared as
    #   zero-arg ``Signature`` entries mirroring the ``color.red`` pattern
    #   (attribute-access as a const-typed value). This is what makes
    #   ``direction=strategy.long`` and ``default_qty_type=strategy.percent_of_equity``
    #   type-check cleanly instead of raising PineUnsupportedBuiltinError.
    #
    # Every entry carries ``notes="SIGNATURE_ONLY"`` per the D5 §7.2
    # convention for signature-landed / codegen-deferred surfaces. The
    # ``IMPLEMENTED`` marker is reserved for entries where the bridge has
    # landed and the script would ``run unedited`` today; strategy scripts
    # still hit PF010 at codegen. Bead ``aeh`` will flip these to
    # ``IMPLEMENTED`` when it lifts the PF010 stub. This convention keeps
    # ``_coverage_manifest.py::BUILTINS_IMPLEMENTED`` and the PRD §3.4 L0.5
    # wild-corpus coverage metric honest — ``notes="IMPLEMENTED"`` alone must
    # not double-count strategy scripts as "would run unedited" when they
    # actually crash at codegen.

    # --- Order-management calls ------------------------------------------------
    "strategy.entry": Signature(
        # Pine v6 signature — matches PyneCore's ``entry()`` in
        # ``third_party/pynecore/src/pynecore/lib/strategy/__init__.py:3034``:
        #   strategy.entry(id, direction, qty, limit, stop, oca_name,
        #                  oca_type, comment, alert_message)
        # ``direction`` is const<string> because it must be ``strategy.long``
        # or ``strategy.short`` (both const<string>). Value-set validation
        # (rejecting "sideways" etc.) is not enforced here — the Signature
        # type system carries no enum-value axis; a later bead may add
        # per-signature literal-value validation. For now, type-level we
        # require const<string> so bare barewords and series-qualified
        # values are still rejected.
        #
        # ``oca_type`` is declared const<string> even though PyneCore uses
        # the ``_oca.Oca`` StrLiteral sentinel. Rationale: M2 accepts any
        # const<string>; strict enum validation is deferred to bead ``aeh``
        # when codegen enforces the enum axis at emission (the bridge will
        # translate ``"cancel"`` / ``"none"`` / ``"reduce"`` into the
        # ``_oca.Oca`` sentinel).
        #
        # NOTE: PyneCore's ``entry()`` has NO ``disable_alert`` param — it
        # was in earlier drafts of this signature but would crash at aeh
        # codegen with ``TypeError: entry() got an unexpected keyword
        # argument 'disable_alert'``. Dropped to match PyneCore.
        args=(
            ("id", _CONST_STRING),
            ("direction", _CONST_STRING),
            ("qty", _SIMPLE_FLOAT),
            ("limit", _SERIES_FLOAT),
            ("stop", _SERIES_FLOAT),
            ("oca_name", _CONST_STRING),
            ("oca_type", _CONST_STRING),
            ("comment", _CONST_STRING),
            ("alert_message", _CONST_STRING),
        ),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),
    "strategy.exit": Signature(
        # Pine v6 signature — the widest kwarg surface in the strategy.*
        # namespace:
        #   strategy.exit(id, from_entry, qty, qty_percent, profit, limit,
        #                 loss, stop, trail_price, trail_points, trail_offset,
        #                 oca_name, comment, comment_profit, comment_loss,
        #                 comment_trailing, alert_message, alert_profit,
        #                 alert_loss, alert_trailing, disable_alert)
        args=(
            ("id", _CONST_STRING),
            ("from_entry", _CONST_STRING),
            ("qty", _SIMPLE_FLOAT),
            ("qty_percent", _SIMPLE_FLOAT),
            ("profit", _SIMPLE_FLOAT),
            ("limit", _SERIES_FLOAT),
            ("loss", _SIMPLE_FLOAT),
            ("stop", _SERIES_FLOAT),
            ("trail_price", _SERIES_FLOAT),
            ("trail_points", _SIMPLE_FLOAT),
            ("trail_offset", _SIMPLE_FLOAT),
            ("oca_name", _CONST_STRING),
            ("comment", _CONST_STRING),
            ("comment_profit", _CONST_STRING),
            ("comment_loss", _CONST_STRING),
            ("comment_trailing", _CONST_STRING),
            ("alert_message", _CONST_STRING),
            ("alert_profit", _CONST_STRING),
            ("alert_loss", _CONST_STRING),
            ("alert_trailing", _CONST_STRING),
            ("disable_alert", _SIMPLE_BOOL),
        ),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),
    "strategy.close": Signature(
        # Close a specific position by ``id``. Matches PyneCore's ``close()``
        # in ``third_party/pynecore/src/pynecore/lib/strategy/__init__.py:2932``:
        #   strategy.close(id, comment, qty, qty_percent, alert_message,
        #                  immediately)
        # NOTE: earlier drafts of this signature declared ``when`` at position
        # 2 (a v4 hangover) and ``disable_alert`` at the tail. Both were
        # DROPPED to match PyneCore:
        # * ``when`` was removed in Pine v6 and PyneCore's ``close()`` never
        #   exposed it — declaring it here corrupted positional binding so
        #   ``strategy.close("id", "closing long")`` would misbind
        #   ``"closing long"`` (const<string>) to formal ``when``
        #   (series<bool>) and raise PT001 with a confusing "cannot demote
        #   const<string> to series<bool>" message.
        # * ``disable_alert`` would type-check cleanly here but crash at aeh
        #   codegen with ``TypeError: close() got an unexpected keyword
        #   argument 'disable_alert'``.
        args=(
            ("id", _CONST_STRING),
            ("comment", _CONST_STRING),
            ("qty", _SIMPLE_FLOAT),
            ("qty_percent", _SIMPLE_FLOAT),
            ("alert_message", _CONST_STRING),
            ("immediately", _SIMPLE_BOOL),
        ),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),
    "strategy.close_all": Signature(
        # Close every open position. Matches PyneCore's ``close_all()`` in
        # ``third_party/pynecore/src/pynecore/lib/strategy/__init__.py:2994``:
        #   strategy.close_all(comment, alert_message, immediately)
        # NOTE: earlier drafts declared ``disable_alert`` at the tail; DROPPED
        # to match PyneCore (would crash at aeh codegen with the same
        # ``TypeError`` pattern as ``strategy.close``).
        args=(
            ("comment", _CONST_STRING),
            ("alert_message", _CONST_STRING),
            ("immediately", _SIMPLE_BOOL),
        ),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),
    "strategy.cancel": Signature(
        # Cancel a specific pending order by ``id``. Single-arg Pine v6 call.
        args=(("id", _CONST_STRING),),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),
    "strategy.cancel_all": Signature(
        # Cancel every pending order. Zero-arg — no filter surface in Pine.
        args=(),
        returns=_VOID,
        notes="SIGNATURE_ONLY",
    ),

    # --- Namespace constants (direction + qty-type enums) ----------------------
    # These are lookup values, not callable functions. The type checker walks
    # ``_visit_attribute`` -> ``lookup_builtin("strategy.long")`` and reads
    # ``sig.returns`` as the value's type. Mirror of the ``color.red`` pattern
    # (bare Attribute yielding a const<T>). Downstream consumers:
    #   * ``strategy.entry(id, strategy.long)`` — direction kwarg receives a
    #     const<string> value; PT001 accepts it against the const<string>
    #     formal (const promotes to any weaker qualifier).
    #   * ``strategy(default_qty_type=strategy.percent_of_equity)`` — the
    #     top-level directive's kwarg (parsed by C2's ScriptDirective
    #     handler). C3 doesn't type-check the directive's kwargs today
    #     (they're passed opaquely to the emitted ``@script.strategy``
    #     decorator) but the attribute still needs to resolve cleanly; else
    #     the type checker raises PineUnsupportedBuiltinError before ever
    #     reaching the directive walk.
    "strategy.long":              Signature(args=(), returns=_CONST_STRING, notes="SIGNATURE_ONLY"),
    "strategy.short":             Signature(args=(), returns=_CONST_STRING, notes="SIGNATURE_ONLY"),
    "strategy.fixed":             Signature(args=(), returns=_CONST_STRING, notes="SIGNATURE_ONLY"),
    "strategy.cash":              Signature(args=(), returns=_CONST_STRING, notes="SIGNATURE_ONLY"),
    "strategy.percent_of_equity": Signature(args=(), returns=_CONST_STRING, notes="SIGNATURE_ONLY"),
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
