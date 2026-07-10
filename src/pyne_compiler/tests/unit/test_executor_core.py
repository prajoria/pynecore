"""E0.3 gate: executor_core must be usable end-to-end with a stub provider,
proving it has NO knowledge of FMP, BYO, or attribution.

The tests here exercise ``executor_core.run_compiled`` in isolation from
every openbb-fork-specific concern (FMP retry, provider-name resolution,
BYO validation, ``POWERED_BY_FULL`` attribution string). A stub provider
that exposes only ``iter_ohlcv()`` — the shape the primary-series loop
consumes today, and the shape both ``FMPOHLCVProvider`` and
``BYODataProvider`` already produce — is enough to drive the core.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
import pytest

from openbb_pine.compiler.types import CompiledModule
from openbb_pine.runtime.executor_core import (
    _collect_results,
    run_compiled,
)


# --- Stub provider ------------------------------------------------------------


class _StubProvider:
    """Minimum-viable primary-series provider for executor_core tests.

    Exposes the two attributes executor_core reads
    (``bars_consumed``, ``provider_used``) plus the ``iter_ohlcv`` generator
    the primary-series loop drives. Deliberately does NOT subclass
    :class:`_DataProviderStub` — that stub's ``stream()`` / ``fetch()``
    contract is the dispatcher's shape (DataFrame return per bead 78w),
    not the primary-series iter_ohlcv shape. Executor_core's job is
    provider-agnostic bar iteration; the dispatcher-abstraction shim lands
    in E3.2 when the real pynecore Provider substitutes for both surfaces.
    """

    def __init__(self, bars: list[Any], *, provider_used: str = "stub") -> None:
        self._bars = list(bars)
        self.provider_used = provider_used
        self.bars_consumed = 0
        self.symbol = "STUB"
        self.interval = "1D"
        self.asset_class = "equity"

    def iter_ohlcv(self) -> Iterator[Any]:
        for bar in self._bars:
            self.bars_consumed += 1
            yield bar


def _stub_bars(rows: int = 5) -> list[Any]:
    from pynecore.types.ohlcv import OHLCV  # noqa: PLC0415

    return [
        OHLCV(timestamp=1704067200 + i * 86400,
              open=100.0 + i, high=101.0 + i, low=99.0 + i,
              close=100.5 + i, volume=1000.0)
        for i in range(rows)
    ]


def _trivial_plot_close_module(*, cache_status: str = "miss") -> CompiledModule:
    source = '''"""
@pyne
"""
from pynecore.lib import script, plot, close

@script.indicator(title="Core")
def main():
    plot(close, "close")
'''
    return CompiledModule(
        source=source, sha="core-smoke", pine_version=6,
        compiler_version="0.1.0",
        builtins_used=frozenset({"plot", "close"}),
        security_contexts=None, cache_status=cache_status,  # type: ignore[arg-type]
    )


# --- End-to-end drive tests ---------------------------------------------------


def test_executor_core_runs_pine_script_against_stub_provider() -> None:
    """Full run_compiled call with the stub provider must yield the same
    per-bar snapshot count as bars consumed."""
    provider = _StubProvider(_stub_bars(rows=5))
    results = run_compiled(
        _trivial_plot_close_module(),
        provider=provider,
        symbol="STUB",
        interval="1D",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
    )
    # run_compiled returns (results_df, exec_ms, alerts).
    df, exec_ms, alerts = results
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 1)
    assert list(df.columns) == ["close"]
    assert alerts == []
    assert isinstance(exec_ms, int)
    assert exec_ms >= 0
    assert provider.bars_consumed == 5


def test_executor_core_alerts_are_captured() -> None:
    """alert() calls inside the script must land in the returned alert list."""
    source = '''"""
@pyne
"""
from pynecore.lib import script, plot, alert, close

@script.indicator(title="Alerter")
def main():
    plot(close, "close")
    alert("ping")
'''
    cm = CompiledModule(
        source=source, sha="core-alert", pine_version=6,
        compiler_version="0.1.0",
        builtins_used=frozenset({"plot", "alert", "close"}),
        security_contexts=None, cache_status="miss",
    )
    provider = _StubProvider(_stub_bars(rows=3))
    _, _, alerts = run_compiled(cm, provider=provider, symbol="STUB", interval="1D")
    assert len(alerts) == 3
    for entry in alerts:
        assert entry["message"] == "ping"


def test_executor_core_forbidden_import_raises_pine_security_error() -> None:
    """T3 second-line-of-defense scan still fires inside the core."""
    from openbb_pine.compiler_errors import PineSecurityError  # noqa: PLC0415

    source = '''"""
@pyne
"""
import os
from pynecore.lib import script, plot, close

@script.indicator(title="Bad")
def main():
    plot(close, "close")
'''
    cm = CompiledModule(
        source=source, sha="core-bad", pine_version=6,
        compiler_version="0.1.0", builtins_used=frozenset(),
        security_contexts=None, cache_status="miss",
    )
    provider = _StubProvider(_stub_bars(rows=1))
    with pytest.raises(PineSecurityError):
        run_compiled(cm, provider=provider, symbol="STUB", interval="1D")


# --- _collect_results shape ---------------------------------------------------


def test_collect_results_empty_yields_bar_indexed_frame() -> None:
    """A script with no plot() calls still returns a DatetimeIndex-carrying frame."""
    from pynecore.types.ohlcv import OHLCV  # noqa: PLC0415

    pairs = [(OHLCV(timestamp=1_704_067_200 + i * 86400, open=0, high=0, low=0,
                    close=0, volume=0), {}) for i in range(3)]
    df = _collect_results(pairs)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert len(df.index) == 3
    assert df.shape[1] == 0


# --- Grep guard: core has no openbb-fork-specific imports ---------------------


def test_executor_core_does_not_import_fmp_or_attribution() -> None:
    """E0.3 gate: executor_core references NOTHING FMP/BYO/attribution-related.

    The test lists every symbol whose presence would prove the split is
    only cosmetic. Update the list intentionally if the shell/core seam
    ever moves — but never widen it to sneak an FMP concern into the core.
    """
    from openbb_pine.runtime import executor_core  # noqa: PLC0415

    src = inspect.getsource(executor_core)
    banned = (
        "FMPOHLCVProvider",
        "FMPRequest",
        "BYODataProvider",
        "POWERED_BY_FULL",
        "call_with_retry",
        "infer_asset_class",
        "from openbb_pine.attribution",
        "from openbb_pine.runtime.fmp_provider",
        "from openbb_pine.runtime.byo_provider",
        "from openbb_pine.runtime.fmp_retry",
        "from openbb_pine.runtime.provider_selection",
    )
    for token in banned:
        assert token not in src, (
            f"executor_core still references {token!r} — E0.3 core/shell "
            "split is incomplete. Core must be provider-agnostic; move "
            "the offending code to executor_shell."
        )


def test_executor_core_does_not_construct_obbject() -> None:
    """The OBBject wrapper is the shell's job — core returns raw pieces."""
    from openbb_pine.runtime import executor_core  # noqa: PLC0415

    src = inspect.getsource(executor_core)
    assert "OBBject(" not in src, (
        "executor_core must NOT construct OBBject; that's the shell's job "
        "so downstream substrates (a future pyne_compiler.execute) can use "
        "the core without importing openbb_core."
    )
