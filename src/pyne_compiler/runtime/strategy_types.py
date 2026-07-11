"""Trade + open-position wire shapes for the strategy ``OBBject.extra`` pathway.

Authoritative source: D5 §1.4 (data shapes) + §2.2 (OBBject.extra fields).

These are the **serialized-for-``.extra``** shapes, not runtime data structures.
PyneCore owns runtime state (``pynecore.lib.strategy.Trade`` and ``SimPosition``
mutate in place during ``ScriptRunner.run_iter()`` iteration); this module owns
the wire format the OBBject exposes to callers.

The two dataclasses (`TradeSummary`, `OpenPositionSummary`) are:

* ``frozen=True`` — post-run snapshots. Callers must not mutate what came off
  the wire; a mutation attempt should fail loudly rather than diverge from the
  PyneCore truth.
* ``slots=True`` — memory-tight (a wild-corpus backtest with 9,000 closed trades
  materializes 9,000 ``TradeSummary`` instances; slots trim ~200 bytes each).
* ``dataclasses.asdict()``-round-trippable — that's the D3 wire layer's
  serialization path (matches ``CompiledModule``'s pattern per D5 §5.1).

The serializer bridge functions (`trade_to_summary`, `position_to_summary`)
translate the mutable PyneCore objects into these immutable snapshots. They
use ``if TYPE_CHECKING`` PyneCore imports so this module doesn't couple its
import path to PyneCore being resolvable at import time — the same pattern
`_pynecore_glue.py` uses for ``SymInfo``. Runtime resolution is via
duck-typing on the attributes the bridge reads.

Field mapping is documented next to each field so the wire shape stays
traceable back to the PyneCore source of truth. Bead ``liz`` (executor
branch) reads from this module; ``5k0`` (StrategyStatistics serializer) is
a sibling wire-shape module that reads from PyneCore's ``strategy_stats``.

Clean-room: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover -- typing-only imports
    from pynecore.lib.strategy import SimPosition, Trade


__all__ = [
    "TradeSummary",
    "OpenPositionSummary",
    "trade_to_summary",
    "position_to_summary",
    "serialize_strategy_statistics",
    "capture_equity_snapshot",
]


# ---------------------------------------------------------------------------
# StrategyStatistics serializer (bd-5k0) — dict-form for OBBject.extra["stats"]
# ---------------------------------------------------------------------------
#
# Sourced from ``pynecore.core.strategy_stats.StrategyStatistics`` (see
# ``src/pynecore/core/strategy_stats.py:17``). That dataclass carries ~70
# fields; we surface the Pine-canonical core set on the wire with stable
# key names so the OBBject.extra["stats"] shape is agnostic to whether
# PyneCore renames its internal attributes over time.
#
# Key aliasing: PyneCore uses ``total_trades`` (matches TV's "Total
# Trades" row) and ``max_equity_drawdown`` (matches TV's "Max Drawdown");
# on the wire we normalize to ``closed_trades`` and ``max_drawdown`` for
# symmetry with the ``TradeSummary`` / ``.extra["orders"]`` naming used
# elsewhere in this module. Callers passing a duck-typed shape may use
# either alias — the serializer prefers the canonical PyneCore name and
# falls back to the wire name.
#
# The canonical field list (see ``_STATS_FIELDS`` below) is a curated
# subset. The full StrategyStatistics dataclass includes long/short
# breakouts, per-side avg-bar counts, etc.; those are omitted on the
# public wire until a caller asks for them. Add fields here — do not
# blanket-``asdict()`` the source, because that leaks PyneCore-internal
# helper attributes (e.g. ``margin_calls``) that TV's Strategy Tester
# does not expose in its stats panel.

# (wire_key, pynecore_attr, wire_key_fallback)
# The ``wire_key_fallback`` field is set when the wire key differs from
# the pynecore attribute and duck-typed callers may pass the wire name
# directly (test doubles, mock stats). The serializer probes both.
_STATS_FIELDS: tuple[tuple[str, str, str | None], ...] = (
    ("net_profit", "net_profit", None),
    ("net_profit_percent", "net_profit_percent", None),
    ("gross_profit", "gross_profit", None),
    ("gross_profit_percent", "gross_profit_percent", None),
    ("gross_loss", "gross_loss", None),
    ("gross_loss_percent", "gross_loss_percent", None),
    ("max_drawdown", "max_equity_drawdown", "max_drawdown"),
    ("max_drawdown_percent", "max_equity_drawdown_percent", "max_drawdown_percent"),
    ("max_runup", "max_equity_runup", "max_runup"),
    ("max_runup_percent", "max_equity_runup_percent", "max_runup_percent"),
    ("buy_and_hold_return", "buy_and_hold_return", None),
    ("buy_and_hold_return_percent", "buy_and_hold_return_percent", None),
    ("sharpe_ratio", "sharpe_ratio", None),
    ("sortino_ratio", "sortino_ratio", None),
    ("profit_factor", "profit_factor", None),
    ("closed_trades", "total_trades", "closed_trades"),
    ("winning_trades", "winning_trades", None),
    ("losing_trades", "losing_trades", None),
    ("percent_profitable", "percent_profitable", None),
    ("avg_trade", "avg_trade", None),
    ("avg_winning_trade", "avg_winning_trade", None),
    ("avg_losing_trade", "avg_losing_trade", None),
    ("largest_winning_trade", "largest_winning_trade", None),
    ("largest_losing_trade", "largest_losing_trade", None),
    ("avg_bars_in_trades", "avg_bars_in_trades", None),
    ("commission_paid", "commission_paid", None),
    ("max_contracts_held", "max_contracts_held", None),
    ("open_trades", "total_open_trades", "open_trades"),
    ("max_cons_winning_trades", "max_cons_winning_trades", None),
    ("max_cons_losing_trades", "max_cons_losing_trades", None),
    ("ratio_avg_win_loss", "ratio_avg_win_loss", None),
)


def serialize_strategy_statistics(stats: object | None) -> dict | None:
    """Convert a ``StrategyStatistics``-shaped object to a plain dict for
    ``OBBject.extra["stats"]``.

    Returns ``None`` when ``stats`` is ``None`` (an indicator run has no
    strategy statistics to serialize — the executor should omit the
    ``stats`` key entirely, or set it to ``None`` explicitly).

    Duck-typed: reads only attributes listed in ``_STATS_FIELDS``.
    Handles the PyneCore canonical names (e.g. ``total_trades``,
    ``max_equity_drawdown``) and their wire-name equivalents
    (``closed_trades``, ``max_drawdown``) so hand-rolled test mocks
    using the wire vocabulary work identically to real
    ``StrategyStatistics`` instances.

    Missing attributes are silently skipped (rather than defaulted to
    zero) so a caller passing a partial mock does not accumulate
    zero-valued keys that the caller never actually set. Real
    ``StrategyStatistics`` always has every field (dataclass defaults),
    so partial output only occurs for test doubles.

    :param stats: A ``pynecore.core.strategy_stats.StrategyStatistics``
        or any duck-typed object exposing the same field names.
    :returns: A dict of the canonical Pine stats keys, or ``None`` when
        the input was ``None``.
    """
    if stats is None:
        return None
    out: dict = {}
    for wire_key, pyne_attr, wire_fallback in _STATS_FIELDS:
        if hasattr(stats, pyne_attr):
            out[wire_key] = getattr(stats, pyne_attr)
        elif wire_fallback is not None and hasattr(stats, wire_fallback):
            out[wire_key] = getattr(stats, wire_fallback)
    return out


def capture_equity_snapshot(
    curve: list,
    bar_index: int,
    equity: float,
    drawdown: float,
) -> None:
    """Append one equity-curve snapshot to ``curve`` in place.

    Snapshot shape is a plain dict: ``{"bar_index": int, "equity":
    float, "drawdown": float}``. The executor calls this once per
    strategy bar (after ``ScriptRunner`` advances) to build the
    ``OBBject.extra["equity_curve"]`` list.

    The caller owns the ``curve`` list — this function just adds
    structure so every snapshot on the wire has the same key names.
    Passing a fresh ``[]`` at the start of the run and reading it at
    end-of-run yields the per-bar equity trajectory.

    :param curve: Target list (mutated in place — append semantics).
    :param bar_index: Bar index of the snapshot (0-based).
    :param equity: Running equity value in currency (typically
        ``strategy.equity`` from PyneCore, which includes closed P&L +
        unrealized).
    :param drawdown: Running drawdown magnitude in currency (non-negative;
        ``0.0`` when equity is at a new high). Matches PyneCore's
        ``max_equity_drawdown`` semantics on a per-bar basis.
    """
    curve.append({"bar_index": bar_index, "equity": equity, "drawdown": drawdown})


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeSummary:
    """One closed round-trip trade — the ``.extra["orders"]`` list element.

    Sourced from ``pynecore.lib.strategy.Trade`` (see PyneCore
    ``strategy/__init__.py:238`` for the full mutable shape). We collapse
    that ~25-attribute internal record into the 14-field public wire shape
    documented in D5 §1.4 (as amended — see below).

    Timestamps are ``pandas.Timestamp`` (UTC) so callers using pandas for
    downstream analysis (openbb-backtest bridge, notebook workflows) get
    native tz-aware types — ``dataclasses.asdict()`` preserves the type,
    JSON serializers can call ``.isoformat()`` themselves. PyneCore stores
    entry/exit_time as int ms-epoch, which we convert once here so nobody
    downstream has to.

    Deltas vs. D5 §1.4 spec (all intentional; see PR #321 review reply):

    * Field renames for consistency with ``OpenPositionSummary`` and the
      ``qty`` / ``pnl`` / ``bars_held`` naming used elsewhere in this module:
      ``side`` → ``direction``; ``entry_id`` → ``id``; ``pnl_percent`` →
      ``pnl_pct``; ``bars_in_trade`` → ``bars_held``.
    * ``entry_signal`` + ``exit_signal`` collapsed into a single ``comment``
      field (PyneCore's ``Trade.exit_comment`` with fallback to
      ``entry_comment``) — dropping the two-field split keeps the wire flat.
    * ``trade_num`` dropped — it is a list index, cheaply reconstructible by
      the caller (bead ``liz``) as ``enumerate(orders, start=1)`` at
      ``asdict`` time. Keeping it here would force this dataclass to know
      its own position in a list it doesn't own.
    * ``slippage`` dropped — PyneCore does not surface per-trade slippage on
      ``Trade`` (it is folded into fill price at ``strategy(slippage=…)``
      time). Reintroduce only when PyneCore starts tracking it separately.
    * ``runup`` / ``drawdown`` **restored** — free data from
      ``Trade.max_runup`` / ``Trade.max_drawdown``.
    * ``datetime`` → ``pd.Timestamp`` for tz-aware pandas interop.
    """

    # Identity / direction --------------------------------------------------
    id: str
    """The Pine ``strategy.entry(id=...)`` identifier. Empty string if the
    entry was placed without an explicit id (rare but permitted)."""

    direction: Literal["long", "short"]
    """``"long"`` when the entry was a buy (``Trade.sign == 1.0``);
    ``"short"`` for a sell (``Trade.sign == -1.0``). Flat sign (``0.0``)
    is never a closed-trade state; the serializer raises ``ValueError``
    at ``_direction_from_sign`` if a zero-sign closed trade is ever
    encountered (invariant violation from PyneCore)."""

    # Entry side ------------------------------------------------------------
    entry_time: pd.Timestamp
    """UTC bar timestamp when the entry order filled."""

    entry_price: float
    """Price at which the entry filled."""

    # Exit side -------------------------------------------------------------
    exit_time: pd.Timestamp
    """UTC bar timestamp when the closing order filled."""

    exit_price: float
    """Price at which the exit filled."""

    # Sizing / P&L ----------------------------------------------------------
    qty: float
    """Absolute contract / share count that transacted (Pine's ``qty``).
    Always non-negative — direction lives in the ``direction`` field."""

    pnl: float
    """Gross profit-and-loss in currency, signed.
    Sourced from ``Trade.profit``. Positive = winning trade."""

    pnl_pct: float
    """Percent P&L relative to initial capital (matches
    ``StrategyStatistics.net_profit_percent`` semantics: pnl / initial_capital
    * 100). PyneCore stores per-trade ``profit_percent`` measured against
    entry value (see ``strategy/__init__.py:1011``), which is a DIFFERENT
    reference — the serializer computes this field explicitly from
    ``initial_capital`` so the sum of all trade ``pnl_pct`` values equals
    ``StrategyStatistics.net_profit_percent`` on the run."""

    # Timing / cost ---------------------------------------------------------
    bars_held: int
    """Number of bars the position was open, from entry fill to exit fill.
    Computed as ``exit_bar_index - entry_bar_index``. A single-bar trade
    (entered and exited on the same bar via close-on-signal) yields ``0``."""

    commission: float
    """Total commission paid across both entry and exit legs."""

    # Excursion (MAE/MFE) ---------------------------------------------------
    runup: float
    """Maximum favorable excursion during the trade life — the peak paper
    profit the position ever reached before it closed. Sourced from
    ``Trade.max_runup`` (see ``strategy/__init__.py:280``). Always
    non-negative. Together with ``drawdown`` this feeds R-multiple /
    MFE analysis without a re-run."""

    drawdown: float
    """Maximum adverse excursion during the trade life — the deepest paper
    loss the position ever reached before it closed. Sourced from
    ``Trade.max_drawdown`` (see ``strategy/__init__.py:278``). Always
    non-negative (PyneCore stores it as an unsigned magnitude)."""

    # Metadata --------------------------------------------------------------
    comment: str | None
    """The ``comment=`` argument from the closing ``strategy.exit()`` or
    ``strategy.close()`` call. ``None`` if the exit did not carry a comment
    (empty string on ``Trade.exit_comment`` normalizes to ``None`` here so
    JSON callers see ``null`` uniformly). Falls back to ``Trade.entry_comment``
    if the trade closed without an explicit exit comment (e.g. margin call).
    """


@dataclass(frozen=True, slots=True)
class OpenPositionSummary:
    """The single running position at end-of-run — ``.extra["open_position"]``.

    Sourced from ``pynecore.lib.strategy.SimPosition``. Only produced when
    ``SimPosition.size != 0`` at the last bar; the serializer returns
    ``None`` for a flat position rather than emitting a synthetic entry
    (D5 §1.4 says "one running position (long / short / flat)" but flat
    means nothing to serialize).
    """

    id: str
    """The entry id of the first (oldest) open trade contributing to the
    running position. When multiple entries are stacked (Pine's ``pyramiding``
    setting), this is the id of the **first entry to fire** — the head of
    ``SimPosition.open_trades``, which PyneCore appends to as new entries
    fire. This mirrors TV's Strategy Tester "Open P&L" panel convention of
    anchoring to the oldest entry. Empty string if the entry was id-less."""

    direction: Literal["long", "short"]
    """``"long"`` when ``SimPosition.sign == 1.0``; ``"short"`` when
    ``-1.0``. Flat position (``sign == 0.0``) is filtered upstream — the
    serializer returns ``None`` before constructing this dataclass, and
    ``_direction_from_sign`` raises ``ValueError`` if a zero-sign value
    ever reaches it (defense against a PyneCore invariant break)."""

    entry_time: pd.Timestamp
    """UTC bar timestamp of the first (oldest) open trade's entry fill.
    When multiple entries are stacked, this is the earliest entry (the head
    of ``SimPosition.open_trades``), which matches TV's Strategy Tester
    "Open P&L" panel convention."""

    entry_price: float
    """Weighted-average entry price for the running position, sourced from
    ``SimPosition.avg_price``. When only one entry is open this equals
    that entry's fill price; with pyramiding, this is the size-weighted
    average across all open entries."""

    qty: float
    """Absolute contract / share count of the running position, sourced
    from ``abs(SimPosition.size)``. Always non-negative — direction lives
    in the ``direction`` field, matching ``TradeSummary``."""

    unrealized_pnl: float
    """Mark-to-market P&L against the last bar's close, sourced directly
    from ``SimPosition.openprofit`` (which PyneCore maintains as
    ``size * (close - avg_price) * pointvalue`` — see
    ``strategy/__init__.py:1064, 1128, 1184, 2481``). Reading the attribute
    verbatim rather than recomputing here ensures the wire value carries
    the ``syminfo.pointvalue`` (contract multiplier), so futures / forex
    symbols with ``pv != 1.0`` agree with PyneCore's own ``openprofit``,
    with ``StrategyStatistics``, and with the TV Strategy Tester panel.
    """

    unrealized_pnl_pct: float
    """Percent unrealized P&L relative to initial capital
    (``unrealized_pnl / initial_capital * 100``). Same denominator as
    ``TradeSummary.pnl_pct`` so a caller can sum the closed-side
    ``pnl_pct`` values across ``.extra["orders"]`` and add
    ``.extra["open_position"].unrealized_pnl_pct`` to get a cumulative
    total-return contribution. Matches TV's "Net Profit %" convention.
    Zero when ``initial_capital <= 0`` (defensive guard)."""

    bars_held: int
    """Bars elapsed since the first (oldest) open entry filled. Matches
    the ``TradeSummary.bars_held`` semantic on the closed side —
    ``current_bar_index - oldest_open_trade.entry_bar_index``. Requires
    the caller to pass ``current_bar_index`` because ``SimPosition`` does
    not carry a running bar counter on its own surface."""


# ---------------------------------------------------------------------------
# Serializer bridge
# ---------------------------------------------------------------------------


def _direction_from_sign(sign: float) -> Literal["long", "short"]:
    """Map a PyneCore ``sign`` (``+1.0`` / ``-1.0``) to our
    ``"long"`` / ``"short"`` literal.

    Raises ``ValueError`` on a zero-sign input. PyneCore's flat state
    (``sign == 0.0``) should never reach the wire shapes — closed trades
    always transacted (non-zero size), open positions with sign 0 are
    filtered by ``position_to_summary``. If a zero-sign ever surfaces here
    it means the caller skipped its own flat check (a bug in the executor)
    or PyneCore's invariants have broken. Silent fall-through to a "long"
    default would produce a mystery zero-sign row in downstream analytics
    that filter on ``direction``, so we fail loud instead.
    """
    if sign > 0.0:
        return "long"
    if sign < 0.0:
        return "short"
    raise ValueError(
        f"_direction_from_sign received sign={sign!r}: expected +1.0 or -1.0. "
        "A zero sign means the caller passed a flat position/trade — the flat "
        "check should happen upstream (see position_to_summary)."
    )


def _ms_epoch_to_ts(ms_epoch: int) -> pd.Timestamp:
    """Convert PyneCore's ms-since-epoch int into a UTC ``pandas.Timestamp``.

    PyneCore stores ``Trade.entry_time`` / ``Trade.exit_time`` as integer
    milliseconds since the Unix epoch (see ``strategy/__init__.py:304``
    where the CSV writer converts back). We standardize on tz-aware UTC
    on the wire so downstream pandas ops work naturally with the primary
    OHLCV DataFrame's tz-aware index (D2 §2.3).

    Raises ``ValueError`` on a negative epoch. PyneCore uses ``-1`` as the
    sentinel for an unset ``exit_time`` (``Trade.__init__`` line 271); if
    that sentinel reaches this helper it means the caller passed an
    unclosed trade to ``trade_to_summary``, which would otherwise produce
    a plausible-looking ``1969-12-31T23:59:59.999+00:00`` timestamp that
    downstream analytics silently trust. Fail loud rather than emit a
    fake pre-epoch timestamp.
    """
    if ms_epoch < 0:
        raise ValueError(
            f"_ms_epoch_to_ts received ms_epoch={ms_epoch}: negative epoch is "
            "PyneCore's -1 sentinel for an unset time. Passing an unclosed "
            "Trade to trade_to_summary is a caller bug."
        )
    return pd.Timestamp(datetime.fromtimestamp(ms_epoch / 1000.0, tz=timezone.utc))


def trade_to_summary(
    trade: "Trade",
    initial_capital: float,
) -> TradeSummary:
    """Convert a PyneCore ``Trade`` (closed round-trip) into a wire-shaped
    ``TradeSummary``.

    :param trade: A completed ``pynecore.lib.strategy.Trade`` from
        ``SimPosition.closed_trades``. Must have ``exit_bar_index >= 0``
        and a non-negative ``exit_time`` (i.e. the closing leg filled).
        Passing an unclosed trade fails loud through ``_ms_epoch_to_ts``
        with ``ValueError`` rather than emitting a sentinel 1969 timestamp.
    :param initial_capital: The strategy's starting capital, used to
        compute ``pnl_pct`` (per-trade percent return against initial
        capital, matching ``StrategyStatistics.net_profit_percent``
        component semantics). Must be positive; a zero or negative value
        yields ``pnl_pct = 0.0`` (defensive — matches PyneCore's
        ``strategy_stats.py:214`` guard).

    Duck-types the ``trade`` argument: this function reads only the
    attributes documented on ``Trade.__slots__`` (see
    ``pynecore/lib/strategy/__init__.py:243``), so a hand-rolled test
    mock with the same attributes works identically to a real ``Trade``
    from ``SimPosition``.
    """
    # ``entry_comment`` fallback: TV's Strategy Tester shows the entry
    # comment when a trade closes without an explicit exit label (margin
    # call, close-on-strategy-end). We normalize empty string → None so
    # JSON callers see ``null`` uniformly rather than mixing "" and null.
    exit_comment_raw = getattr(trade, "exit_comment", "") or ""
    entry_comment_raw = getattr(trade, "entry_comment", None)
    comment: str | None
    if exit_comment_raw:
        comment = exit_comment_raw
    elif entry_comment_raw:
        comment = entry_comment_raw
    else:
        comment = None

    # Percent P&L against initial capital — matches D5 §3.1's stat semantics
    # (``StrategyStatistics.net_profit_percent = net_profit / initial_capital``).
    # PyneCore's per-trade ``profit_percent`` is measured against entry value,
    # which is NOT what we want on this wire shape.
    pnl_pct = (trade.profit / initial_capital * 100.0) if initial_capital > 0.0 else 0.0

    # bars_held from bar-index deltas; guard against a possible negative
    # count (should never happen for a closed trade with exit_bar_index
    # already validated above via _ms_epoch_to_ts, but clamp is cheap).
    entry_bar = int(trade.entry_bar_index)
    exit_bar = int(trade.exit_bar_index)
    bars_held = max(0, exit_bar - entry_bar)

    return TradeSummary(
        id=trade.entry_id or "",
        direction=_direction_from_sign(trade.sign),
        entry_time=_ms_epoch_to_ts(int(trade.entry_time)),
        entry_price=float(trade.entry_price),
        exit_time=_ms_epoch_to_ts(int(trade.exit_time)),
        exit_price=float(trade.exit_price),
        qty=abs(float(trade.size)),
        pnl=float(trade.profit),
        pnl_pct=pnl_pct,
        bars_held=bars_held,
        commission=float(trade.commission),
        runup=float(trade.max_runup),
        drawdown=float(trade.max_drawdown),
        comment=comment,
    )


def position_to_summary(
    pos: "SimPosition",
    current_bar_index: int,
    initial_capital: float,
) -> OpenPositionSummary | None:
    """Convert a PyneCore ``SimPosition`` into an ``OpenPositionSummary``.

    Returns ``None`` if the position is flat (``size == 0``) — the caller
    should either drop the field or emit ``null`` for ``.extra["open_position"]``.
    D5 §1.4 mentions a ``"flat"`` literal, but a flat position has no
    entry price / no unrealized P&L / no meaningful entry time, so
    materializing the row adds noise rather than signal. Returning None
    lets the executor write ``extra["open_position"] = None`` explicitly.

    :param pos: A ``pynecore.lib.strategy.SimPosition`` (see
        ``pynecore/lib/strategy/__init__.py:662``) that was mutated in
        place by ``ScriptRunner.run_iter()`` during a strategy run.
        Duck-typed — the function reads ``size``, ``sign``, ``avg_price``,
        ``openprofit``, and ``open_trades[0]`` (for entry_time /
        entry_bar_index).
    :param current_bar_index: Bar index of the last bar the strategy saw.
        Used to compute ``bars_held``. The executor sources this from
        ``lib.bar_index`` immediately after ``run_iter()`` exhausts.
    :param initial_capital: The strategy's starting capital, used to
        compute ``unrealized_pnl_pct``. Same denominator as
        ``TradeSummary.pnl_pct`` so the closed + open sides are additive.
        Zero or negative values yield ``unrealized_pnl_pct = 0.0``.

    ``current_bar_close`` is intentionally NOT a parameter: unrealized P&L
    is read from ``SimPosition.openprofit`` (which PyneCore maintains as
    ``size * (close - avg_price) * pointvalue``) so the wire value already
    carries the ``syminfo.pointvalue`` contract multiplier. Recomputing
    from ``current_bar_close`` here would silently drop the ``pv`` factor
    on futures / forex.
    """
    # Deferred import — the wire-shape module keeps its import graph light
    # (see the ``TYPE_CHECKING`` block at the top). NA is only needed at
    # runtime inside the na-guard below.
    from pynecore.types.na import NA

    # Flat guard — no direction, no entry, nothing to serialize.
    if float(pos.size) == 0.0:
        return None

    # NA-guard for avg_price. Symmetric to the open_trades empty guard
    # below: SimPosition.avg_price is typed PyneFloat (NA[float] | float)
    # and is set to na_float on the ZeroDivisionError fallback branch
    # (strategy/__init__.py:1182) even when size != 0. ``float(na_float)``
    # would raise ``TypeError: NA cannot be converted to float`` and
    # abort the whole serialization pass; instead we treat it as
    # "PyneCore invariant violation" and return None like we do for the
    # empty-open_trades case.
    if isinstance(pos.avg_price, NA):
        return None

    # Head of open_trades = the oldest still-open entry. When pyramiding is
    # disabled (default), there's only one; with pyramiding, TV's Strategy
    # Tester anchors the "Open P&L" panel to the oldest entry, and we mirror
    # that convention.
    open_trades = getattr(pos, "open_trades", None) or []
    if not open_trades:
        # Defensive: SimPosition.size != 0 with an empty open_trades list is a
        # PyneCore invariant violation. Rather than raise (which would blow up
        # a whole backtest at serialization time for a hard-to-diagnose edge
        # case), we return None so the caller sees "no open position summary"
        # and can log the discrepancy separately.
        return None
    oldest = open_trades[0]

    qty = abs(float(pos.size))
    avg_price = float(pos.avg_price)
    # Read openprofit directly — PyneCore already applies syminfo.pointvalue
    # (contract multiplier), which recomputing (close - avg) * size would
    # silently drop for futures / forex with pv != 1.0.
    unrealized_pnl = float(pos.openprofit)
    # Percent unrealized against initial capital — same denominator as
    # TradeSummary.pnl_pct so closed and open sides can be summed.
    unrealized_pnl_pct = (
        (unrealized_pnl / initial_capital * 100.0) if initial_capital > 0.0 else 0.0
    )

    entry_bar = int(oldest.entry_bar_index)
    bars_held = max(0, int(current_bar_index) - entry_bar)

    return OpenPositionSummary(
        id=(oldest.entry_id or ""),
        direction=_direction_from_sign(pos.sign),
        entry_time=_ms_epoch_to_ts(int(oldest.entry_time)),
        entry_price=avg_price,
        qty=qty,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        bars_held=bars_held,
    )
