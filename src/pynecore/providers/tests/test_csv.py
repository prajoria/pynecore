"""E1.2: CSVProvider reads RFC 4180 CSV files with columns
``timestamp,open,high,low,close,volume``.

Direct-CSV reads bypass the base class's file-backed ``.ohlcv``
round-trip (the CSV IS the data file). All base-class guards (mode-1
mismatch → ValueError, naive datetime → TypeError, reversed range → [],
missing file → []) are replicated by the override so behavior stays
uniform across providers — this is what the E1.4 (bd-cko) conformance
suite will verify.

Test names follow pynecore's convention (``python_functions = __test_*__``
in ``pytest.ini``); functions named ``test_*`` are silently skipped by
the collector, which is the pynecore-side gotcha the plan reference code
missed.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pynecore.providers.csv import CSVProvider


FIXTURE = Path(__file__).parent / "fixtures" / "testsym_1d.csv"


def _make(tmp_path: Path, symbol: str = "TESTSYM", timeframe: str = "1D",
          csv_path: Path | None = None) -> CSVProvider:
    """Construct a CSVProvider whose config lives under tmp_path.

    The default ``load_config`` on CSVProvider is a stub (no providers.toml
    lookup), so tmp_path only serves as a placeholder for the ohlv_dir/
    config_dir contract inherited from ``Provider.__init__``.
    """
    return CSVProvider(
        csv_path=csv_path if csv_path is not None else FIXTURE,
        symbol=symbol,
        timeframe=timeframe,
        ohlv_dir=tmp_path,
        config_dir=tmp_path,
    )


#
# Presence + basic construction
#

def __test_csv_provider_class_is_importable__() -> None:
    """CSVProvider is exposed at ``pynecore.providers.csv.CSVProvider``."""
    assert CSVProvider is not None


def __test_csv_provider_construction_does_not_touch_csv__(tmp_path: Path) -> None:
    """Constructing with a non-existent CSV must NOT raise (spec §5.4
    check #3: missing data → ``[]`` from stream/fetch, not raise). A
    provider defined for a symbol we haven't populated yet is a
    legitimate state, e.g. a config-driven registry."""
    missing = tmp_path / "does-not-exist.csv"
    provider = _make(tmp_path, csv_path=missing)
    # No exception on construction, and the missing file surfaces as an
    # empty result rather than a FileNotFoundError.
    assert provider.fetch("TESTSYM", "1D") == []
    assert list(provider.stream("TESTSYM", "1D")) == []


#
# Happy-path reads against the fixture (Jan 1-5 2024 daily bars)
#

def __test_csv_provider_fetches_all_bars__(tmp_path: Path) -> None:
    provider = _make(tmp_path)
    bars = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
    )
    assert len(bars) == 5
    assert bars[0].close == 101.0
    assert bars[-1].close == 104.5
    # Ordering + timestamps.
    timestamps = [b.timestamp for b in bars]
    assert timestamps == sorted(timestamps)
    assert bars[0].timestamp == 1704067200  # 2024-01-01 UTC epoch


def __test_csv_provider_filters_by_range__(tmp_path: Path) -> None:
    provider = _make(tmp_path)
    bars = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    assert len(bars) == 3
    assert bars[0].close == 101.5
    assert bars[-1].close == 103.5


def __test_csv_provider_unbounded_start__(tmp_path: Path) -> None:
    provider = _make(tmp_path)
    bars = provider.fetch(
        "TESTSYM", "1D",
        start=None,
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert len(bars) == 3
    assert bars[0].close == 101.0
    assert bars[-1].close == 102.5


def __test_csv_provider_unbounded_end__(tmp_path: Path) -> None:
    provider = _make(tmp_path)
    bars = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 3, tzinfo=timezone.utc),
        end=None,
    )
    assert len(bars) == 3
    assert bars[0].close == 102.5
    assert bars[-1].close == 104.5


def __test_csv_provider_both_endpoints_unbounded__(tmp_path: Path) -> None:
    provider = _make(tmp_path)
    bars = provider.fetch("TESTSYM", "1D")
    assert len(bars) == 5


#
# Conformance-suite (E1.4) previews — these MUST pass in every provider
#

def __test_csv_provider_stream_matches_fetch__(tmp_path: Path) -> None:
    """Conformance check #1 (spec §5.4): ``list(stream(...)) == fetch(...)``
    for closed historical ranges."""
    provider = _make(tmp_path)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    streamed = list(provider.stream("TESTSYM", "1D", start=start, end=end))
    fetched = provider.fetch("TESTSYM", "1D", start=start, end=end)
    assert streamed == fetched


def __test_csv_provider_stream_returns_iterator_not_list__(tmp_path: Path) -> None:
    """``stream()`` returns a lazy Iterator; ``fetch()`` returns a list."""
    provider = _make(tmp_path)
    streamed = provider.stream("TESTSYM", "1D")
    assert iter(streamed) is streamed, "stream() must return an Iterator"
    assert not isinstance(streamed, list), "stream() must not materialize a list"
    fetched = provider.fetch("TESTSYM", "1D")
    assert isinstance(fetched, list), "fetch() must return a concrete list"


def __test_csv_provider_reversed_range_returns_empty__(tmp_path: Path) -> None:
    """Conformance check #4 (spec §5.4 Rev 3): ``start > end`` → ``[]``,
    NOT raise. SQL-consistent semantics."""
    provider = _make(tmp_path)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 5, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert result == []
    assert list(provider.stream(
        "TESTSYM", "1D",
        start=datetime(2024, 1, 5, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )) == []


def __test_csv_provider_range_before_data_returns_empty__(tmp_path: Path) -> None:
    """A range entirely before the first bar returns ``[]``."""
    provider = _make(tmp_path)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 2, tzinfo=timezone.utc),
    )
    assert result == []


def __test_csv_provider_range_after_data_returns_empty__(tmp_path: Path) -> None:
    """A range entirely after the last bar returns ``[]``."""
    provider = _make(tmp_path)
    result = provider.fetch(
        "TESTSYM", "1D",
        start=datetime(2030, 1, 1, tzinfo=timezone.utc),
        end=datetime(2030, 1, 2, tzinfo=timezone.utc),
    )
    assert result == []


def __test_csv_provider_symbol_mismatch_raises__(tmp_path: Path) -> None:
    """Conformance check #5 (spec §5.4): mode-1 provider must raise
    ``ValueError`` when call-time symbol differs from construction time,
    rather than silently returning the construction-time symbol's data."""
    provider = _make(tmp_path, symbol="TESTSYM")
    with pytest.raises(ValueError, match="does not match construction-time"):
        provider.fetch("NEVER_EXISTS_XYZ", "1D",
                       start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                       end=datetime(2024, 1, 5, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="does not match construction-time"):
        list(provider.stream("NEVER_EXISTS_XYZ", "1D"))


def __test_csv_provider_timeframe_mismatch_raises__(tmp_path: Path) -> None:
    """Same guard on timeframe — a mode-1 provider built for 1D cannot
    silently serve 1H."""
    provider = _make(tmp_path)
    with pytest.raises(ValueError, match="does not match construction-time"):
        provider.fetch("TESTSYM", "1H")


def __test_csv_provider_naive_start_raises__(tmp_path: Path) -> None:
    """Naive datetimes are ambiguous cross-machine (spec §5)."""
    provider = _make(tmp_path)
    with pytest.raises(TypeError, match="timezone-aware"):
        provider.fetch(
            "TESTSYM", "1D",
            start=datetime(2024, 1, 1),  # no tzinfo
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )


def __test_csv_provider_naive_end_raises__(tmp_path: Path) -> None:
    """Same guard on the end bound."""
    provider = _make(tmp_path)
    with pytest.raises(TypeError, match="timezone-aware"):
        provider.fetch(
            "TESTSYM", "1D",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5),  # no tzinfo
        )


def __test_csv_provider_start_end_are_keyword_only__(tmp_path: Path) -> None:
    """``start``/``end`` are keyword-only per the spec §5.1 signature."""
    provider = _make(tmp_path)
    with pytest.raises(TypeError):
        provider.fetch(
            "TESTSYM", "1D",
            datetime(2024, 1, 1, tzinfo=timezone.utc),  # type: ignore[misc]
            datetime(2024, 1, 5, tzinfo=timezone.utc),  # type: ignore[misc]
        )


def __test_csv_provider_repeated_calls_stateless__(tmp_path: Path) -> None:
    """No cursor state between calls — repeated fetch/stream returns the
    same rows. Regression against a common CSV-provider footgun where a
    single ``DictReader`` is reused across calls and exhausts silently."""
    provider = _make(tmp_path)
    first = provider.fetch("TESTSYM", "1D")
    second = provider.fetch("TESTSYM", "1D")
    assert first == second
    assert len(first) == 5

    streamed_first = list(provider.stream("TESTSYM", "1D"))
    streamed_second = list(provider.stream("TESTSYM", "1D"))
    assert streamed_first == streamed_second


def __test_csv_provider_extends_provider_base__() -> None:
    """CSVProvider is a subclass of the extended Provider base (bd-dnf)."""
    from pynecore.providers.provider import Provider
    assert issubclass(CSVProvider, Provider)


#
# CSV-specific parsing errors
#

def __test_csv_provider_missing_columns_raises__(tmp_path: Path) -> None:
    """CSV missing a required column raises a helpful ``ValueError``."""
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "timestamp,open,high,low,close\n"  # no volume column
        "1704067200,100.0,101.5,99.0,101.0\n",
        encoding="utf-8",
    )
    provider = _make(tmp_path, csv_path=bad)
    with pytest.raises(ValueError, match="missing columns"):
        provider.fetch("TESTSYM", "1D")


def __test_csv_provider_malformed_row_raises_with_line_number__(tmp_path: Path) -> None:
    """A non-numeric value in a data row raises ``ValueError`` and names
    the offending line so operators can locate it without a bisect."""
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1704067200,100.0,101.5,99.0,101.0,1000\n"
        "1704153600,not-a-number,102.0,100.5,101.5,1100\n",  # line 3
        encoding="utf-8",
    )
    provider = _make(tmp_path, csv_path=bad)
    with pytest.raises(ValueError, match=r":3"):
        list(provider.stream("TESTSYM", "1D"))
