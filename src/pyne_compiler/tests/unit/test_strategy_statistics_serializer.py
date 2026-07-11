"""StrategyStatistics -> .extra['stats'] dict + per-bar equity snapshot (bd-5k0).

Per D5 sec R2, when a strategy compiles+runs, the executor emits a result
object whose .extra['stats'] carries a dict-form of pynecore's
StrategyStatistics, and .extra['equity_curve'] carries per-bar equity
snapshots. Both are optional (only present for script_type='strategy' runs).
"""
from __future__ import annotations

from pyne_compiler.runtime.strategy_types import (
    serialize_strategy_statistics,
    capture_equity_snapshot,
)


def test_serializer_produces_expected_dict_keys():
    """StrategyStatistics dict must include the core Pine strategy stats fields."""

    class FakeStats:
        net_profit = 1234.5
        gross_profit = 2000.0
        gross_loss = -765.5
        max_drawdown = 300.0
        max_drawdown_percent = 3.0
        sharpe_ratio = 1.2
        sortino_ratio = 1.5
        profit_factor = 2.61
        closed_trades = 42
        winning_trades = 25
        losing_trades = 17
        avg_trade = 29.4
        avg_winning_trade = 80.0
        avg_losing_trade = -45.0

    result = serialize_strategy_statistics(FakeStats())
    assert isinstance(result, dict)
    for key in ("net_profit", "gross_profit", "gross_loss",
                "max_drawdown", "sharpe_ratio", "profit_factor",
                "closed_trades", "winning_trades", "losing_trades"):
        assert key in result, f"missing {key} in serialized stats: {list(result.keys())}"
    assert result["net_profit"] == 1234.5
    assert result["closed_trades"] == 42
    assert result["max_drawdown"] == 300.0


def test_serializer_handles_none_input():
    """When no strategy ran (script_type='indicator'), serializer returns None."""
    result = serialize_strategy_statistics(None)
    assert result in (None, {}), f"expected None or empty dict, got {result!r}"


def test_serializer_reads_pynecore_aliased_names():
    """Real pynecore StrategyStatistics uses ``total_trades`` and
    ``max_equity_drawdown``; the serializer must recognize those alias names
    so callers passing an actual ``StrategyStatistics`` instance get the same
    canonical output keys as callers passing a duck-typed shape."""

    class RealShape:
        net_profit = 500.0
        gross_profit = 800.0
        gross_loss = -300.0
        max_equity_drawdown = 150.0
        max_equity_drawdown_percent = 1.5
        sharpe_ratio = 0.9
        sortino_ratio = 1.1
        profit_factor = 2.67
        total_trades = 10
        winning_trades = 6
        losing_trades = 4

    result = serialize_strategy_statistics(RealShape())
    assert result["closed_trades"] == 10, "total_trades should alias to closed_trades"
    assert result["max_drawdown"] == 150.0, "max_equity_drawdown should alias to max_drawdown"


def test_capture_equity_snapshot_appends_to_curve():
    curve: list[dict] = []
    capture_equity_snapshot(curve, bar_index=0, equity=10000.0, drawdown=0.0)
    capture_equity_snapshot(curve, bar_index=1, equity=10250.0, drawdown=0.0)
    capture_equity_snapshot(curve, bar_index=2, equity=9800.0, drawdown=200.0)
    assert len(curve) == 3
    assert curve[0] == {"bar_index": 0, "equity": 10000.0, "drawdown": 0.0}
    assert curve[2]["drawdown"] == 200.0
