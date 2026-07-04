"""Tests for ``openbb_pine.runtime.strategy_types`` — D5 §1.4.

Covers the wire-shape invariants (frozen, slots, asdict roundtrip) for
``TradeSummary`` + ``OpenPositionSummary`` and the PyneCore-serializer
functions (``trade_to_summary``, ``position_to_summary``).

The serializer tests use hand-rolled mock ``Trade`` and ``SimPosition``
classes rather than instantiating the real PyneCore objects — the bridge
functions are duck-typed on the attributes documented in D5 §1.4, so a
minimal namespace suffices. This keeps the tests independent of PyneCore's
import setup and cuts the test-time cost of exercising every branch.

Clean-room: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from openbb_pine.runtime.strategy_types import (
    OpenPositionSummary,
    TradeSummary,
    position_to_summary,
    trade_to_summary,
)


# ---------------------------------------------------------------------------
# Fixtures — hand-rolled mocks that satisfy the duck-typed contract
# ---------------------------------------------------------------------------


def _mock_trade(
    *,
    entry_id: str = "long1",
    size: float = 100.0,
    entry_bar_index: int = 10,
    entry_time_ms: int = 1_700_000_000_000,  # 2023-11-14T22:13:20Z
    entry_price: float = 50.0,
    exit_id: str = "long1",
    exit_bar_index: int = 20,
    exit_time_ms: int = 1_700_864_000_000,  # 10 days later, roughly
    exit_price: float = 55.0,
    commission: float = 2.5,
    entry_comment: str | None = "buy",
    exit_comment: str = "sell",
    profit: float | None = None,
    max_runup: float = 0.0,
    max_drawdown: float = 0.0,
) -> Any:
    """Build a duck-typed mock ``Trade`` with just the attributes the
    serializer reads. Matches ``pynecore.lib.strategy.Trade.__slots__``
    field-for-field but only for the subset the bridge accesses.

    ``profit`` defaults to ``(exit_price - entry_price) * size`` (long-side
    identity); pass an explicit value to override for short-side tests.
    ``sign`` is derived from ``size`` (positive = long, negative = short),
    matching PyneCore's ``Trade.__init__`` convention.
    ``max_runup`` / ``max_drawdown`` default to 0.0 (both non-negative
    magnitudes on PyneCore's ``Trade``).
    """
    sign = 0.0 if size == 0.0 else (1.0 if size > 0.0 else -1.0)
    if profit is None:
        profit = (exit_price - entry_price) * size
    return SimpleNamespace(
        entry_id=entry_id,
        size=size,
        sign=sign,
        entry_bar_index=entry_bar_index,
        entry_time=entry_time_ms,
        entry_price=entry_price,
        entry_comment=entry_comment,
        exit_id=exit_id,
        exit_bar_index=exit_bar_index,
        exit_time=exit_time_ms,
        exit_price=exit_price,
        exit_comment=exit_comment,
        commission=commission,
        profit=profit,
        max_runup=max_runup,
        max_drawdown=max_drawdown,
    )


def _mock_sim_position(
    *,
    size: float = 100.0,
    avg_price: float = 50.0,
    open_trades: list[Any] | None = None,
    openprofit: float | None = None,
    current_bar_close: float | None = None,
    pointvalue: float = 1.0,
) -> Any:
    """Build a duck-typed mock ``SimPosition`` with just the attributes
    the serializer reads. ``sign`` is derived from ``size``.

    ``open_trades`` defaults to a single-entry list whose head matches the
    ``size`` / ``avg_price`` convention (bar 10, ms epoch as in ``_mock_trade``).
    Pass an explicit list to override — the serializer reads
    ``open_trades[0]`` for the oldest-entry anchor.

    ``openprofit`` defaults to PyneCore's identity
    ``size * (current_bar_close - avg_price) * pointvalue``. Callers that
    exercise the pv-multiplier path pass ``pointvalue != 1.0`` and (optionally)
    ``current_bar_close`` to make the mock stand in for what a real PyneCore
    ``SimPosition`` would have after ``ScriptRunner.run_iter()`` computed
    ``self.openprofit = self.size * (self.c - self.avg_price) * pv``
    (see ``strategy/__init__.py:1064``). Callers that want to test the
    "read openprofit verbatim" behavior pass an explicit ``openprofit``
    that would NOT match the (close-avg)*size formula.
    """
    sign = 0.0 if size == 0.0 else (1.0 if size > 0.0 else -1.0)
    if open_trades is None:
        open_trades = [
            SimpleNamespace(
                entry_id="pos_entry",
                entry_bar_index=10,
                entry_time=1_700_000_000_000,
            )
        ]
    if openprofit is None:
        # Default to PyneCore's formula so tests that don't care about pv
        # get a sensible value (size * (close - avg_price) * pv).
        # When current_bar_close is None we fall back to 0.0 (flat P&L).
        if current_bar_close is None:
            openprofit = 0.0
        else:
            openprofit = size * (current_bar_close - avg_price) * pointvalue
    return SimpleNamespace(
        size=size,
        sign=sign,
        avg_price=avg_price,
        openprofit=openprofit,
        open_trades=open_trades,
    )


# ---------------------------------------------------------------------------
# TradeSummary — dataclass invariants
# ---------------------------------------------------------------------------


class TestTradeSummaryConstruction:
    """Construction fills every field and ``asdict()`` returns those keys."""

    def _make(self) -> TradeSummary:
        return TradeSummary(
            id="entry1",
            direction="long",
            entry_time=pd.Timestamp("2024-01-01T10:00:00Z"),
            entry_price=100.0,
            exit_time=pd.Timestamp("2024-01-05T10:00:00Z"),
            exit_price=110.0,
            qty=50.0,
            pnl=500.0,
            pnl_pct=5.0,
            bars_held=4,
            commission=1.25,
            runup=650.0,
            drawdown=100.0,
            comment="tp",
        )

    def test_all_fields_populated(self) -> None:
        summary = self._make()
        # Spot-check each field surface — the dataclass would raise if any
        # required kwarg was missing, so a successful construction proves the
        # field list is complete.
        assert summary.id == "entry1"
        assert summary.direction == "long"
        assert summary.entry_price == 100.0
        assert summary.exit_price == 110.0
        assert summary.qty == 50.0
        assert summary.pnl == 500.0
        assert summary.pnl_pct == 5.0
        assert summary.bars_held == 4
        assert summary.commission == 1.25
        assert summary.runup == 650.0
        assert summary.drawdown == 100.0
        assert summary.comment == "tp"

    def test_asdict_returns_expected_keys(self) -> None:
        d = dataclasses.asdict(self._make())
        assert set(d.keys()) == {
            "id",
            "direction",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "qty",
            "pnl",
            "pnl_pct",
            "bars_held",
            "commission",
            "runup",
            "drawdown",
            "comment",
        }


class TestTradeSummaryFrozenAndSlots:
    """Frozen dataclass with ``slots=True`` — post-run snapshot invariants."""

    def _make(self) -> TradeSummary:
        return TradeSummary(
            id="e",
            direction="long",
            entry_time=pd.Timestamp("2024-01-01T00:00:00Z"),
            entry_price=1.0,
            exit_time=pd.Timestamp("2024-01-02T00:00:00Z"),
            exit_price=2.0,
            qty=1.0,
            pnl=1.0,
            pnl_pct=1.0,
            bars_held=1,
            commission=0.0,
            runup=0.0,
            drawdown=0.0,
            comment=None,
        )

    def test_mutation_raises_frozen_instance_error(self) -> None:
        summary = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.id = "other"  # type: ignore[misc]

    def test_no_instance_dict_when_slots_active(self) -> None:
        summary = self._make()
        assert not hasattr(summary, "__dict__")


# ---------------------------------------------------------------------------
# OpenPositionSummary — dataclass invariants
# ---------------------------------------------------------------------------


class TestOpenPositionSummaryConstruction:
    """Construction fills every field and ``asdict()`` returns those keys."""

    def _make(self) -> OpenPositionSummary:
        return OpenPositionSummary(
            id="entry1",
            direction="long",
            entry_time=pd.Timestamp("2024-01-01T10:00:00Z"),
            entry_price=100.0,
            qty=50.0,
            unrealized_pnl=250.0,
            unrealized_pnl_pct=5.0,
            bars_held=3,
        )

    def test_all_fields_populated(self) -> None:
        s = self._make()
        assert s.id == "entry1"
        assert s.direction == "long"
        assert s.entry_price == 100.0
        assert s.qty == 50.0
        assert s.unrealized_pnl == 250.0
        assert s.unrealized_pnl_pct == 5.0
        assert s.bars_held == 3

    def test_asdict_returns_expected_keys(self) -> None:
        d = dataclasses.asdict(self._make())
        assert set(d.keys()) == {
            "id",
            "direction",
            "entry_time",
            "entry_price",
            "qty",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "bars_held",
        }


class TestOpenPositionSummaryFrozenAndSlots:
    """Frozen dataclass with ``slots=True`` — same invariants as TradeSummary."""

    def _make(self) -> OpenPositionSummary:
        return OpenPositionSummary(
            id="p",
            direction="short",
            entry_time=pd.Timestamp("2024-01-01T00:00:00Z"),
            entry_price=1.0,
            qty=1.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            bars_held=0,
        )

    def test_mutation_raises_frozen_instance_error(self) -> None:
        s = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.qty = 2.0  # type: ignore[misc]

    def test_no_instance_dict_when_slots_active(self) -> None:
        s = self._make()
        assert not hasattr(s, "__dict__")


# ---------------------------------------------------------------------------
# trade_to_summary — serializer bridge for closed trades
# ---------------------------------------------------------------------------


class TestTradeToSummary:
    """Mock ``Trade`` → correctly-shaped ``TradeSummary``."""

    def test_long_trade_populates_all_fields(self) -> None:
        # Long: bought 100 @ 50, sold @ 55 → +$500 profit on $10k capital = +5%.
        trade = _mock_trade(
            entry_id="buy1",
            size=100.0,
            entry_bar_index=10,
            exit_bar_index=20,
            entry_price=50.0,
            exit_price=55.0,
            commission=2.5,
            entry_comment="signal_buy",
            exit_comment="signal_sell",
        )
        summary = trade_to_summary(trade, initial_capital=10_000.0)

        assert summary.id == "buy1"
        assert summary.direction == "long"
        assert summary.entry_price == 50.0
        assert summary.exit_price == 55.0
        assert summary.qty == 100.0
        assert summary.pnl == pytest.approx(500.0)
        assert summary.pnl_pct == pytest.approx(5.0)
        assert summary.bars_held == 10
        assert summary.commission == 2.5
        assert summary.comment == "signal_sell"

    def test_short_trade_pnl_sign(self) -> None:
        # Short: sold 100 @ 60, bought back @ 55 → +$500 profit.
        # size is negative for short → profit = (exit - entry) * size
        # = (55 - 60) * -100 = 500. sign=-1.0 → direction="short".
        trade = _mock_trade(
            entry_id="short1",
            size=-100.0,
            entry_price=60.0,
            exit_price=55.0,
            entry_bar_index=5,
            exit_bar_index=8,
        )
        summary = trade_to_summary(trade, initial_capital=10_000.0)

        assert summary.direction == "short"
        assert summary.qty == 100.0  # abs value on the wire
        assert summary.pnl == pytest.approx(500.0)
        assert summary.bars_held == 3

    def test_short_trade_losing_pnl_sign(self) -> None:
        # Short that lost: sold 100 @ 50, bought back @ 55 → -$500.
        # (55 - 50) * -100 = -500.
        trade = _mock_trade(
            entry_id="short_bad",
            size=-100.0,
            entry_price=50.0,
            exit_price=55.0,
        )
        summary = trade_to_summary(trade, initial_capital=10_000.0)
        assert summary.direction == "short"
        assert summary.pnl < 0.0
        assert summary.pnl_pct < 0.0

    def test_pnl_pct_uses_initial_capital(self) -> None:
        # Fixed $1000 profit at two different initial_capital settings:
        # $10k → 10%, $100k → 1%. Confirms formula is pnl / initial_capital.
        trade = _mock_trade(
            entry_id="e",
            size=100.0,
            entry_price=10.0,
            exit_price=20.0,  # $1000 profit
        )
        s_small = trade_to_summary(trade, initial_capital=10_000.0)
        s_large = trade_to_summary(trade, initial_capital=100_000.0)
        assert s_small.pnl_pct == pytest.approx(10.0)
        assert s_large.pnl_pct == pytest.approx(1.0)

    def test_pnl_pct_zero_initial_capital_defensive(self) -> None:
        # Guard: initial_capital == 0 → pnl_pct == 0 rather than ZeroDivisionError.
        trade = _mock_trade(size=10.0, entry_price=1.0, exit_price=2.0)
        s = trade_to_summary(trade, initial_capital=0.0)
        assert s.pnl_pct == 0.0

    def test_entry_time_is_utc_timestamp(self) -> None:
        # ms epoch → tz-aware pd.Timestamp in UTC.
        trade = _mock_trade(entry_time_ms=0, exit_time_ms=86_400_000)
        s = trade_to_summary(trade, initial_capital=10_000.0)
        assert isinstance(s.entry_time, pd.Timestamp)
        assert s.entry_time.tzinfo is not None
        assert s.entry_time == pd.Timestamp("1970-01-01T00:00:00Z")
        assert s.exit_time == pd.Timestamp("1970-01-02T00:00:00Z")

    def test_comment_falls_back_to_entry_when_exit_empty(self) -> None:
        # Margin-call / end-of-run close: no exit_comment set. Wire should
        # surface the entry_comment as a fallback so callers see *some*
        # context rather than a bare null.
        trade = _mock_trade(entry_comment="original_signal", exit_comment="")
        s = trade_to_summary(trade, initial_capital=10_000.0)
        assert s.comment == "original_signal"

    def test_comment_none_when_both_missing(self) -> None:
        trade = _mock_trade(entry_comment=None, exit_comment="")
        s = trade_to_summary(trade, initial_capital=10_000.0)
        assert s.comment is None

    def test_empty_entry_id_normalizes_to_empty_string(self) -> None:
        # Pine allows id-less entries; PyneCore stores None for entry_id in
        # that case. Our wire surface uses "" (empty string) uniformly so
        # downstream code does not have to null-check.
        trade = _mock_trade(entry_id=None)  # type: ignore[arg-type]
        s = trade_to_summary(trade, initial_capital=10_000.0)
        assert s.id == ""

    def test_bars_held_never_negative(self) -> None:
        # Defensive: entry_bar > exit_bar shouldn't happen for a closed trade
        # but if a mock passes it, we clamp to 0 rather than propagating a
        # negative count downstream.
        trade = _mock_trade(entry_bar_index=20, exit_bar_index=10)
        s = trade_to_summary(trade, initial_capital=10_000.0)
        assert s.bars_held == 0

    def test_asdict_roundtrips(self) -> None:
        trade = _mock_trade()
        summary = trade_to_summary(trade, initial_capital=10_000.0)
        d = dataclasses.asdict(summary)
        # 14 fields per the D5 §1.4 spec (as amended in PR #321):
        # renames + restored runup/drawdown; dropped trade_num/slippage.
        assert len(d) == 14
        # Timestamps come through as pd.Timestamp (JSON callers must
        # normalize; we confirm the type here).
        assert isinstance(d["entry_time"], pd.Timestamp)
        assert isinstance(d["exit_time"], pd.Timestamp)

    def test_asdict_json_serializable_after_isoformat(self) -> None:
        # Once the caller replaces the two Timestamps with .isoformat()
        # strings, the whole dict is JSON-clean. Mirrors what the
        # executor (bead ``liz``) will do when writing extra.
        trade = _mock_trade()
        d = dataclasses.asdict(trade_to_summary(trade, initial_capital=10_000.0))
        d["entry_time"] = d["entry_time"].isoformat()
        d["exit_time"] = d["exit_time"].isoformat()
        # No error → JSON-clean.
        json.dumps(d)


# ---------------------------------------------------------------------------
# position_to_summary — serializer bridge for running position
# ---------------------------------------------------------------------------


class TestPositionToSummary:
    """Mock ``SimPosition`` → ``OpenPositionSummary`` or None."""

    def test_flat_position_returns_none(self) -> None:
        pos = _mock_sim_position(size=0.0, avg_price=0.0, open_trades=[])
        result = position_to_summary(pos, current_bar_index=50, initial_capital=10_000.0)
        assert result is None

    def test_long_position_populates_unrealized_pnl(self) -> None:
        # Long 100 @ 50, current close 55 → +$500 unrealized on $10k capital
        # → +5% unrealized_pnl_pct (same denominator as TradeSummary.pnl_pct).
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)

        assert result is not None
        assert result.direction == "long"
        assert result.qty == 100.0
        assert result.entry_price == 50.0
        assert result.unrealized_pnl == pytest.approx(500.0)
        assert result.unrealized_pnl_pct == pytest.approx(5.0)
        # entry_bar_index=10 on the default mock, current_bar_index=25 → 15 bars held.
        assert result.bars_held == 15
        assert result.id == "pos_entry"

    def test_short_position_unrealized_pnl_sign(self) -> None:
        # Short 100 (size=-100) @ 60, current close 55 → +$500 unrealized
        # for the short side ((55 - 60) * -100 = 500).
        pos = _mock_sim_position(
            size=-100.0, avg_price=60.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=15, initial_capital=10_000.0)

        assert result is not None
        assert result.direction == "short"
        assert result.qty == 100.0
        assert result.unrealized_pnl == pytest.approx(500.0)  # positive: short winning
        assert result.unrealized_pnl_pct > 0.0

    def test_short_position_losing_unrealized_pnl_sign(self) -> None:
        # Short that's underwater: sold @ 50, current close 60 → -$1000.
        pos = _mock_sim_position(
            size=-100.0, avg_price=50.0, current_bar_close=60.0
        )
        result = position_to_summary(pos, current_bar_index=15, initial_capital=10_000.0)

        assert result is not None
        assert result.direction == "short"
        assert result.unrealized_pnl == pytest.approx(-1000.0)
        assert result.unrealized_pnl_pct < 0.0

    def test_unrealized_pnl_pct_zero_when_initial_capital_zero(self) -> None:
        # Defensive: initial_capital == 0 → pct == 0 rather than ZeroDivisionError.
        # (Matches the TradeSummary.pnl_pct guard so both sides align.)
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=5, initial_capital=0.0)
        assert result is not None
        assert result.unrealized_pnl_pct == 0.0

    def test_empty_open_trades_returns_none(self) -> None:
        # Defensive: SimPosition invariant violation (size != 0 but no open
        # trades). Should return None rather than crashing — same guard as
        # flat position.
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, open_trades=[], current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is None

    def test_bars_held_from_current_bar_index(self) -> None:
        # bars_held = current_bar_index - oldest_open_trade.entry_bar_index.
        oldest = SimpleNamespace(
            entry_id="e0", entry_bar_index=50, entry_time=1_700_000_000_000
        )
        pos = _mock_sim_position(
            size=1.0, avg_price=1.0, open_trades=[oldest], current_bar_close=1.0
        )
        result = position_to_summary(pos, current_bar_index=57, initial_capital=10_000.0)
        assert result is not None
        assert result.bars_held == 7

    def test_bars_held_clamped_to_zero(self) -> None:
        # Defensive: current_bar_index < entry_bar_index → 0 rather than negative.
        oldest = SimpleNamespace(
            entry_id="e0", entry_bar_index=100, entry_time=1_700_000_000_000
        )
        pos = _mock_sim_position(
            size=1.0, avg_price=1.0, open_trades=[oldest], current_bar_close=1.0
        )
        result = position_to_summary(pos, current_bar_index=50, initial_capital=10_000.0)
        assert result is not None
        assert result.bars_held == 0

    def test_entry_time_is_utc_timestamp(self) -> None:
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is not None
        assert isinstance(result.entry_time, pd.Timestamp)
        assert result.entry_time.tzinfo is not None

    def test_empty_entry_id_normalizes_to_empty_string(self) -> None:
        oldest = SimpleNamespace(
            entry_id=None, entry_bar_index=10, entry_time=1_700_000_000_000
        )
        pos = _mock_sim_position(
            size=1.0, avg_price=1.0, open_trades=[oldest], current_bar_close=1.0
        )
        result = position_to_summary(pos, current_bar_index=20, initial_capital=10_000.0)
        assert result is not None
        assert result.id == ""

    def test_asdict_roundtrips(self) -> None:
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is not None
        d = dataclasses.asdict(result)
        # 8 fields per the D5 §1.4 spec.
        assert len(d) == 8
        assert isinstance(d["entry_time"], pd.Timestamp)

    def test_asdict_json_serializable_after_isoformat(self) -> None:
        pos = _mock_sim_position(
            size=100.0, avg_price=50.0, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is not None
        d = dataclasses.asdict(result)
        d["entry_time"] = d["entry_time"].isoformat()
        json.dumps(d)


# ---------------------------------------------------------------------------
# Regression tests — findings from PR #321 review
# ---------------------------------------------------------------------------


class TestPointValueMultiplier:
    """Regression for PR #321 review [BUG] finding.

    PyneCore computes ``SimPosition.openprofit = size * (close - avg_price) * pv``
    where ``pv`` is ``syminfo.pointvalue`` (contract multiplier — 50 for ES
    futures, 20 for NQ, etc). The old serializer recomputed
    ``(close - avg_price) * size`` locally, silently dropping ``pv`` for any
    non-equity symbol. Reading ``pos.openprofit`` directly preserves it.
    """

    def test_futures_pointvalue_50_preserved(self) -> None:
        # ES futures: pv=50. Long 2 contracts @ 4500, current close 4510.
        # PyneCore's identity: openprofit = 2 * (4510 - 4500) * 50 = 1000.
        # A naive (close - avg) * size recompute yields (4510 - 4500) * 2 = 20,
        # which is off by 50x (the pv factor). The serializer must surface 1000.
        pos = _mock_sim_position(
            size=2.0,
            avg_price=4500.0,
            current_bar_close=4510.0,
            pointvalue=50.0,
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=100_000.0)
        assert result is not None
        assert result.unrealized_pnl == pytest.approx(1000.0)
        # And NOT the naive recompute value (would have been 20 with pv dropped).
        assert result.unrealized_pnl != pytest.approx(20.0)

    def test_forex_pointvalue_100k_preserved(self) -> None:
        # EURUSD standard lot: 100_000 units per contract. Long 1 @ 1.1000,
        # close 1.1050 → 1 * 0.005 * 100_000 = 500.
        pos = _mock_sim_position(
            size=1.0,
            avg_price=1.1000,
            current_bar_close=1.1050,
            pointvalue=100_000.0,
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=100_000.0)
        assert result is not None
        assert result.unrealized_pnl == pytest.approx(500.0)

    def test_reads_openprofit_verbatim_ignoring_recompute(self) -> None:
        # Explicit proof: if the mock's openprofit is set to a value that does
        # NOT match the (close - avg) * size identity, the serializer must
        # surface the openprofit value (not the recompute). This locks in the
        # "read verbatim" contract even if a future PyneCore change breaks
        # the identity in some edge case (e.g. mid-bar partial fills).
        pos = _mock_sim_position(
            size=100.0,
            avg_price=50.0,
            current_bar_close=55.0,  # ignored; openprofit is authoritative
            openprofit=12345.67,     # intentionally not (55-50)*100 = 500
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is not None
        assert result.unrealized_pnl == pytest.approx(12345.67)


class TestPositionToSummaryEdgeGuards:
    """Regression for PR #321 review [QUESTION] on ``avg_price == na_float``."""

    def test_na_avg_price_returns_none(self) -> None:
        # PyneCore sets SimPosition.avg_price = na_float on the
        # ZeroDivisionError fallback branch (strategy/__init__.py:1182) even
        # when size != 0. Without a guard, float(na_float) raises TypeError
        # and aborts the whole serialization pass. Symmetric to the existing
        # empty-open_trades guard: return None instead.
        from pynecore.types.na import na_float

        pos = _mock_sim_position(
            size=100.0, avg_price=na_float, current_bar_close=55.0
        )
        result = position_to_summary(pos, current_bar_index=25, initial_capital=10_000.0)
        assert result is None


class TestDirectionFromSignGuards:
    """Regression for PR #321 review [NIT] — ``_direction_from_sign(0.0)`` must raise."""

    def test_zero_sign_raises(self) -> None:
        from openbb_pine.runtime.strategy_types import _direction_from_sign

        with pytest.raises(ValueError, match="expected \\+1.0 or -1.0"):
            _direction_from_sign(0.0)

    def test_positive_sign_returns_long(self) -> None:
        from openbb_pine.runtime.strategy_types import _direction_from_sign

        assert _direction_from_sign(1.0) == "long"

    def test_negative_sign_returns_short(self) -> None:
        from openbb_pine.runtime.strategy_types import _direction_from_sign

        assert _direction_from_sign(-1.0) == "short"


class TestMsEpochGuards:
    """Regression for PR #321 review [NIT] — ``_ms_epoch_to_ts(-1)`` must raise."""

    def test_negative_epoch_raises(self) -> None:
        from openbb_pine.runtime.strategy_types import _ms_epoch_to_ts

        with pytest.raises(ValueError, match="-1 sentinel"):
            _ms_epoch_to_ts(-1)

    def test_zero_epoch_returns_1970(self) -> None:
        # Zero is a valid epoch (1970-01-01), only negative should raise.
        from openbb_pine.runtime.strategy_types import _ms_epoch_to_ts

        ts = _ms_epoch_to_ts(0)
        assert ts == pd.Timestamp("1970-01-01T00:00:00Z")

    def test_unclosed_trade_via_trade_to_summary_raises(self) -> None:
        # End-to-end: passing a trade with exit_time = -1 (PyneCore's unset
        # sentinel) must fail loud through _ms_epoch_to_ts rather than emit
        # a plausible 1969-12-31 timestamp downstream analytics would trust.
        trade = _mock_trade(exit_time_ms=-1)
        with pytest.raises(ValueError, match="-1 sentinel"):
            trade_to_summary(trade, initial_capital=10_000.0)


class TestTradeSummaryExcursionFields:
    """Regression for PR #321 review [QUESTION] — runup/drawdown restored."""

    def test_runup_and_drawdown_populated_from_trade(self) -> None:
        # Trade with a $200 favorable excursion and $50 adverse excursion —
        # both surfaced verbatim from Trade.max_runup / Trade.max_drawdown.
        trade = _mock_trade(
            entry_price=50.0,
            exit_price=55.0,
            size=100.0,
            max_runup=200.0,
            max_drawdown=50.0,
        )
        summary = trade_to_summary(trade, initial_capital=10_000.0)
        assert summary.runup == pytest.approx(200.0)
        assert summary.drawdown == pytest.approx(50.0)

    def test_runup_and_drawdown_default_zero(self) -> None:
        # PyneCore initializes both to 0.0 (see Trade.__init__ line 278/280).
        # A trade that never moved past its entry has zero excursions.
        trade = _mock_trade()  # defaults: max_runup=0.0, max_drawdown=0.0
        summary = trade_to_summary(trade, initial_capital=10_000.0)
        assert summary.runup == 0.0
        assert summary.drawdown == 0.0
