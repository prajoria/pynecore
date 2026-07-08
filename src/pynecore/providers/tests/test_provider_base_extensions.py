"""E1.1: Provider base class now exposes stream() and fetch() concrete methods
with a default file-backed implementation.

The default implementations use the existing download_ohlcv() + load_ohlcv_data()
flow, so existing CCXT + CapitalCom subclasses inherit them for free with no
code change. Subclasses (CSV, SQLite, FMP) can override for direct-query
optimization.

Test names follow the pynecore convention (``python_functions = __test_*__``
in ``pytest.ini``).
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import pytest

from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
from pynecore.providers.provider import Provider
from pynecore.types.ohlcv import OHLCV


class _MinimalProvider(Provider):
    """Fake concrete provider whose download_ohlcv writes 5 known daily bars
    to the underlying .ohlcv file. Verifies the default stream()/fetch() flow
    reads them back correctly.
    """

    # Sane no-op implementations of the abstract methods so the ABCMeta
    # instantiation guard is satisfied.
    @classmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return timeframe

    @classmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        return timeframe

    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        return ["TESTSYM"]

    def update_symbol_info(self):
        # The base class only calls this via get_symbol_info(), which the
        # tests here never trigger. A raise makes accidental invocation loud.
        raise NotImplementedError("_MinimalProvider does not model symbol info")

    @classmethod
    def get_opening_hours_and_sessions(cls) -> tuple[
        list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]
    ]:
        return [], [], []

    def load_config(self) -> None:
        # Skip the providers.toml lookup — tests supply an isolated tmp_path
        # for both ohlv_dir and config_dir and do not need any provider config.
        self.config = {}

    def download_ohlcv(  # type: ignore[override]
        self,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        on_progress: Callable[[datetime], None] | None = None,
        limit: int | None = None,
    ) -> None:
        # Write 5 known daily bars (Jan 1-5 2024 UTC). The parent __enter__
        # opens self.ohlcv_file so writes flush immediately.
        assert self.ohlcv_file is not None, "call inside `with provider:`"
        for i in range(1, 6):
            ts = int(datetime(2024, 1, i, tzinfo=timezone.utc).timestamp())
            self.ohlcv_file.write(OHLCV(
                timestamp=ts,
                open=float(i),
                high=float(i),
                low=float(i),
                close=float(i),
                volume=1.0,
            ))


def _make_provider(tmp_path: Path, symbol: str = "TESTSYM") -> _MinimalProvider:
    """Construct a _MinimalProvider whose .ohlcv file lives under tmp_path.

    Both ohlv_dir and config_dir point at tmp_path — load_config is stubbed
    out so no providers.toml is required.
    """
    return _MinimalProvider(
        symbol=symbol,
        timeframe="1D",
        ohlv_dir=tmp_path,
        config_dir=tmp_path,
    )


def _seed(provider: _MinimalProvider) -> None:
    """Write the 5 fixture bars into provider's underlying .ohlcv file."""
    with provider:
        provider.download_ohlcv()


#
# Presence + return-type checks
#

def __test_provider_exposes_stream_method__() -> None:
    """Provider.stream is a callable attribute of the base class."""
    assert callable(getattr(Provider, "stream", None))


def __test_provider_exposes_fetch_method__() -> None:
    """Provider.fetch is a callable attribute of the base class."""
    assert callable(getattr(Provider, "fetch", None))


def __test_stream_returns_iterator_not_list__(tmp_path: Path) -> None:
    """stream() returns an Iterator (a lazy generator), not a materialized list."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.stream("TESTSYM", "1D")
    # A list is Iterable but not an Iterator; a generator is both.
    assert iter(result) is result, "stream() must return an Iterator"
    assert not isinstance(result, list), "stream() must not materialize a list"


def __test_fetch_returns_list__(tmp_path: Path) -> None:
    """fetch() returns a materialized list, not a lazy iterator."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch("TESTSYM", "1D")
    assert isinstance(result, list), "fetch() must return a concrete list"


#
# Default-impl round-trip through the .ohlcv file
#

def __test_default_stream_yields_download_ohlcv_output__(tmp_path: Path) -> None:
    """After download_ohlcv() writes 5 bars, stream() yields those 5 bars in order."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    bars = list(provider.stream(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
    ))
    assert len(bars) == 5
    assert bars[0].close == 1.0
    assert bars[-1].close == 5.0
    # Chronological order + UTC timestamps (integer seconds since epoch).
    timestamps = [b.timestamp for b in bars]
    assert timestamps == sorted(timestamps)


def __test_default_fetch_matches_stream__(tmp_path: Path) -> None:
    """list(stream(...)) == fetch(...) for closed historical ranges (spec §5.1)."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    streamed = list(provider.stream("TESTSYM", "1D", start=start, end=end))
    fetched = provider.fetch("TESTSYM", "1D", start=start, end=end)
    assert streamed == fetched


#
# Range-filter semantics
#

def __test_stream_returns_empty_for_range_before_data__(tmp_path: Path) -> None:
    """A [start, end] range entirely before the first bar yields nothing."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 2, tzinfo=timezone.utc),
    )
    assert result == []


def __test_stream_returns_empty_for_range_after_data__(tmp_path: Path) -> None:
    """A [start, end] range entirely after the last bar yields nothing."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2030, 1, 1, tzinfo=timezone.utc),
        end=datetime(2030, 1, 2, tzinfo=timezone.utc),
    )
    assert result == []


def __test_stream_reversed_range_returns_empty__(tmp_path: Path) -> None:
    """Per spec §5.4 conformance check #4 (Rev 3): start > end returns [], NOT raise."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 5, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert result == []


def __test_stream_partial_range_slices_bars__(tmp_path: Path) -> None:
    """A [start, end] range that overlaps a subset yields exactly that subset."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    # Bars are Jan 1-5. Ask for Jan 2 through Jan 4 → 3 bars.
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    assert len(result) == 3
    assert result[0].close == 2.0
    assert result[-1].close == 4.0


def __test_stream_unbounded_start_yields_from_beginning__(tmp_path: Path) -> None:
    """start=None means unbounded on the low end (spec §5.1)."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=None,
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert len(result) == 3
    assert result[0].close == 1.0
    assert result[-1].close == 3.0


def __test_stream_unbounded_end_yields_to_end__(tmp_path: Path) -> None:
    """end=None means unbounded on the high end (spec §5.1)."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 3, tzinfo=timezone.utc),
        end=None,
    )
    assert len(result) == 3
    assert result[0].close == 3.0
    assert result[-1].close == 5.0


def __test_stream_both_endpoints_unbounded__(tmp_path: Path) -> None:
    """Both start and end None yields every persisted bar."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    result = provider.fetch("TESTSYM", "1D")
    assert len(result) == 5


#
# Call-parameterization: one provider instance can serve multiple queries
#
# Notes: the current file-backed default impl still resolves (symbol,
# timeframe) at Provider construction time via ohlcv_path. So a genuine
# multi-symbol query against a single instance is only possible for
# subclasses that override stream/fetch and don't rely on the .ohlcv path.
# We assert here that (a) the call-time API accepts the parameters without
# error and (b) repeated calls against the same instance and same
# (symbol, timeframe) are stateless — they return equal results each time.
#

def __test_repeated_fetch_is_stateless__(tmp_path: Path) -> None:
    """Calling fetch() twice on one provider yields identical results
    (no cursor state carried between calls)."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    first = provider.fetch("TESTSYM", "1D", start=start, end=end)
    second = provider.fetch("TESTSYM", "1D", start=start, end=end)
    assert first == second
    assert len(first) == 5


def __test_repeated_stream_is_stateless__(tmp_path: Path) -> None:
    """Calling stream() twice on one provider yields identical bar sequences."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    first = list(provider.stream("TESTSYM", "1D"))
    second = list(provider.stream("TESTSYM", "1D"))
    assert first == second


def __test_stream_and_fetch_accept_positional_call_time_args__(tmp_path: Path) -> None:
    """symbol/timeframe are call-time positional args (spec §5.1 signature)."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    # Both must accept (symbol, timeframe, *, start=..., end=...).
    _ = list(provider.stream("TESTSYM", "1D"))
    _ = provider.fetch("TESTSYM", "1D")


def __test_start_and_end_are_keyword_only__(tmp_path: Path) -> None:
    """start and end must be keyword-only per the spec signature."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    # Positional start/end must fail (they are keyword-only in the spec).
    with pytest.raises(TypeError):
        provider.fetch(
            "TESTSYM", "1D",
            datetime(2024, 1, 1, tzinfo=timezone.utc),  # type: ignore[misc]
            datetime(2024, 1, 5, tzinfo=timezone.utc),  # type: ignore[misc]
        )
    with pytest.raises(TypeError):
        list(provider.stream(
            "TESTSYM", "1D",
            datetime(2024, 1, 1, tzinfo=timezone.utc),  # type: ignore[misc]
            datetime(2024, 1, 5, tzinfo=timezone.utc),  # type: ignore[misc]
        ))


#
# Backward-compat: existing subclasses that only implement download_ohlcv
# inherit stream() / fetch() unchanged.
#

def __test_subclass_without_stream_override_inherits_default__() -> None:
    """A subclass that overrides nothing beyond the abstract methods should
    still expose the default stream() + fetch() from Provider (spec §5)."""
    assert _MinimalProvider.stream is Provider.stream
    assert _MinimalProvider.fetch is Provider.fetch


def __test_subclass_may_override_stream__(tmp_path: Path) -> None:
    """Subclasses MAY override stream() for direct-query optimization
    (spec §5.1 — CSV/SQLite/FMP will do this in E1.2/E1.3/E3.2)."""
    sentinel_bar = OHLCV(
        timestamp=int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()),
        open=99.0, high=99.0, low=99.0, close=99.0, volume=1.0,
    )

    class _OverrideProvider(_MinimalProvider):
        def stream(  # type: ignore[override]
            self,
            symbol: str,
            timeframe: str,
            *,
            start: datetime | None = None,
            end: datetime | None = None,
            include_gaps: bool = False,
        ) -> Iterator[OHLCV]:
            yield sentinel_bar

    provider = _OverrideProvider(
        symbol="TESTSYM", timeframe="1D", ohlv_dir=tmp_path, config_dir=tmp_path,
    )
    # No download_ohlcv needed — the override does not touch the file.
    bars = list(provider.stream("TESTSYM", "1D"))
    assert bars == [sentinel_bar]
    # Default fetch() consumes the overridden stream() → same content.
    fetched = provider.fetch("TESTSYM", "1D")
    assert fetched == [sentinel_bar]


#
# Regression tests for PR #2 review findings (bd-dnf).
#
# BUG #1: default stream() must raise ValueError on mode-1 symbol/timeframe
# mismatch instead of silently returning the construction-time symbol's data
# (spec §5.4 check #5). BUG #2: missing .ohlcv file must return [] instead
# of raw FileNotFoundError (spec §5.4 check #3). QUESTION #3: include_gaps
# opt-in must expose OHLCVReader.read_from's skip_gaps lever. NIT: no test
# for gap-fill skipping or naive-datetime rejection.
#


class _GapFillProvider(_MinimalProvider):
    """Writes 5 daily bars with a synthetic gap on Jan 3 so OHLCVWriter
    auto-fills that day with a volume=-1 sentinel bar (gap-fill marker)."""

    def download_ohlcv(  # type: ignore[override]
        self,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        on_progress: Callable[[datetime], None] | None = None,
        limit: int | None = None,
    ) -> None:
        assert self.ohlcv_file is not None, "call inside `with provider:`"
        # Write bars for Jan 1, 2, 4, 5 (skip Jan 3). OHLCVWriter fills
        # Jan 3 automatically with the previous close and volume=-1 to
        # preserve the uniform 1D interval — this is the gap-fill marker
        # that stream() must skip by default.
        for i in [1, 2, 4, 5]:
            ts = int(datetime(2024, 1, i, tzinfo=timezone.utc).timestamp())
            self.ohlcv_file.write(OHLCV(
                timestamp=ts,
                open=float(i), high=float(i), low=float(i), close=float(i),
                volume=1.0,
            ))


def __test_stream_skips_gap_fill_bars_by_default__(tmp_path: Path) -> None:
    """Default stream() drops volume=-1 gap-fill bars (spec §5 "real
    trading activity"). The underlying file has 5 bars (Jan 1-5 with a
    synthetic vol=-1 marker on Jan 3), stream() yields 4."""
    provider = _GapFillProvider(
        symbol="TESTSYM", timeframe="1D", ohlv_dir=tmp_path, config_dir=tmp_path,
    )
    with provider:
        provider.download_ohlcv()
    # Verify the underlying file did get a gap-fill marker.
    with provider.load_ohlcv_data() as reader:
        raw = list(reader)
    assert len(raw) == 5, "OHLCVWriter should auto-fill Jan 3 with vol=-1"
    assert any(b.volume < 0 for b in raw), "gap-fill marker missing"

    bars = provider.fetch("TESTSYM", "1D")
    assert len(bars) == 4, "gap-fill bar should be skipped by default"
    assert all(b.volume >= 0 for b in bars), "no vol=-1 bars should survive"


def __test_stream_includes_gap_fill_bars_when_opted_in__(tmp_path: Path) -> None:
    """``include_gaps=True`` preserves vol=-1 markers (spec §5 "MAY" clause
    — matches OHLCVReader.read_from's skip_gaps opt-in)."""
    provider = _GapFillProvider(
        symbol="TESTSYM", timeframe="1D", ohlv_dir=tmp_path, config_dir=tmp_path,
    )
    with provider:
        provider.download_ohlcv()

    bars = provider.fetch("TESTSYM", "1D", include_gaps=True)
    assert len(bars) == 5, "include_gaps=True should preserve gap-fill markers"
    assert any(b.volume < 0 for b in bars), "gap-fill marker should be visible"


def __test_stream_raises_on_naive_start_datetime__(tmp_path: Path) -> None:
    """Naive datetimes are ambiguous across timezones; spec §5 requires
    aware. ``.timestamp()`` on naive silently interprets in local tz —
    a cross-machine reproducibility footgun — so we reject explicitly."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    with pytest.raises(TypeError, match="timezone-aware"):
        provider.fetch(
            "TESTSYM", "1D",
            start=datetime(2024, 1, 1),  # no tzinfo
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )


def __test_stream_raises_on_naive_end_datetime__(tmp_path: Path) -> None:
    """Same guard on the end bound."""
    provider = _make_provider(tmp_path)
    _seed(provider)
    with pytest.raises(TypeError, match="timezone-aware"):
        provider.fetch(
            "TESTSYM", "1D",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5),  # no tzinfo
        )


def __test_stream_raises_on_call_time_symbol_mismatch__(tmp_path: Path) -> None:
    """BUG #1: mode-1 default must fail loud when call-time symbol differs
    from construction-time (spec §5.4 check #5). The old impl silently
    returned the construction-time symbol's data — a data-corruption bug."""
    provider = _make_provider(tmp_path, symbol="TESTSYM")
    _seed(provider)
    with pytest.raises(ValueError, match="does not match construction-time"):
        provider.fetch("DIFFERENT_SYM", "1D")
    with pytest.raises(ValueError, match="does not match construction-time"):
        list(provider.stream("DIFFERENT_SYM", "1D"))


def __test_stream_raises_on_call_time_timeframe_mismatch__(tmp_path: Path) -> None:
    """Same guard on timeframe — a mode-1 provider constructed for 1D
    cannot silently return 1D data when asked for 1H."""
    provider = _make_provider(tmp_path, symbol="TESTSYM")
    _seed(provider)
    with pytest.raises(ValueError, match="does not match construction-time"):
        provider.fetch("TESTSYM", "1H")


def __test_stream_returns_empty_when_ohlcv_file_missing__(tmp_path: Path) -> None:
    """BUG #2: missing .ohlcv file → ``[]``, not raw FileNotFoundError
    (spec §5.4 check #3 — 'unknown range must return [] cleanly').
    The old impl leaked an internal file-format concern to the caller."""
    # Construct provider without calling download_ohlcv → no .ohlcv file.
    provider = _make_provider(tmp_path)
    assert not provider.ohlcv_path.exists(), "test premise: file must not exist"

    result = provider.fetch("TESTSYM", "1D")
    assert result == []

    # stream() should also yield nothing (not raise on iteration).
    assert list(provider.stream("TESTSYM", "1D")) == []


def __test_stream_uses_read_from_fast_path_semantics__(tmp_path: Path) -> None:
    """QUESTION #4: default stream() delegates to OHLCVReader.read_from,
    which uses byte-offset positioning (O(range) — not O(file)). We can't
    directly assert perf here, but we can lock in that the semantics still
    match the previous Python-side impl exactly: start after last bar
    yields [], start on the first bar yields all bars from there, etc."""
    provider = _make_provider(tmp_path)
    _seed(provider)

    # Start strictly after the last bar → empty (get_positions clamps
    # start_pos to size-1, so without our post-filter this would yield
    # the last bar spuriously). Our regression test locks the exact bound.
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 6, tzinfo=timezone.utc),
    )
    assert result == []

    # End strictly before the first bar → empty.
    result = provider.fetch(
        "TESTSYM", "1D",
        end=datetime(2023, 12, 31, tzinfo=timezone.utc),
    )
    assert result == []
