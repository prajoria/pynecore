"""Behavioral tests for :class:`pynecore.providers.sqlite.SQLiteProvider`
(bd-l05, Task E1.3).

The E1.4 conformance suite (bd-cko) will re-run a shared set of checks
against every provider; the tests here focus on SQLite-specific behavior
(schema configurability, SQL-injection guards, read-only enforcement,
column-type coercion) and the core spec §5.4 conformance checks (missing
symbol → ``[]``, reversed range → ``[]``, timezone-aware only, closed-
range fetch/stream equivalence).

Test names follow the pynecore convention: ``__test_*__``
(``python_functions = __test_*__`` in ``pytest.ini``). Anything that
doesn't match that pattern is silently skipped by collection.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pynecore.providers.sqlite import (
    SQLiteProvider,
    _validate_identifier,
)


# ------------------------------------------------------------------
# Fixture DB builders
# ------------------------------------------------------------------
# We generate a fresh SQLite DB per test into ``tmp_path`` so tests never
# share state and cleanup is automatic. Two DB shapes: the "standard"
# schema (matches the SQLiteProvider defaults) and a "custom" schema
# (renamed columns + different table name) to exercise the config surface.


def _make_standard_db(
    tmp_path: Path,
    *,
    rows: list[tuple[str, str, int, float, float, float, float, float]] | None = None,
) -> Path:
    """Create a DB with the SQLiteProvider default schema.

    Schema: ``ohlcv(symbol TEXT, timeframe TEXT, timestamp INTEGER,
    open REAL, high REAL, low REAL, close REAL, volume REAL)``

    Defaults to 5 daily MSFT bars for Jan 1-5 2024 UTC, close=i, vol=1.
    """
    if rows is None:
        rows = [
            (
                "MSFT",
                "1D",
                int(datetime(2024, 1, i, tzinfo=timezone.utc).timestamp()),
                float(i),
                float(i),
                float(i),
                float(i),
                1.0,
            )
            for i in range(1, 6)
        ]
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "symbol TEXT, timeframe TEXT, timestamp INTEGER, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL)"
        )
        conn.executemany(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _make_custom_schema_db(tmp_path: Path) -> Path:
    """DB where every user-configurable name has been renamed. Exercises
    the constructor-kwarg / providers.toml override surface."""
    db = tmp_path / "custom.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE bars ("
            "ticker TEXT, tf TEXT, ts INTEGER, "
            "o REAL, h REAL, l REAL, c REAL, v REAL)"
        )
        for i in range(1, 4):
            conn.execute(
                "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "AAPL",
                    "1H",
                    int(datetime(2024, 3, i, tzinfo=timezone.utc).timestamp()),
                    float(i * 10),
                    float(i * 10),
                    float(i * 10),
                    float(i * 10),
                    2.0,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db


def _make_no_tf_column_db(tmp_path: Path) -> Path:
    """Single-timeframe DB — the schema omits the timeframe column so
    users must construct with ``timeframe_column=None``."""
    db = tmp_path / "notf.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "symbol TEXT, timestamp INTEGER, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL)"
        )
        for i in range(1, 4):
            conn.execute(
                "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "SPY",
                    int(datetime(2024, 5, i, tzinfo=timezone.utc).timestamp()),
                    float(i),
                    float(i),
                    float(i),
                    float(i),
                    3.0,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db


# ------------------------------------------------------------------
# Identifier validator (unit-level)
# ------------------------------------------------------------------

def __test_validate_identifier_accepts_plain_ascii__() -> None:
    """Alphanumeric + underscore ascii identifiers pass without change."""
    assert _validate_identifier("ohlcv", "table_name") == "ohlcv"
    assert _validate_identifier("symbol_1", "col") == "symbol_1"
    assert _validate_identifier("_x", "col") == "_x"


def __test_validate_identifier_rejects_empty__() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validate_identifier("", "col")


def __test_validate_identifier_rejects_leading_digit__() -> None:
    with pytest.raises(ValueError, match="digit"):
        _validate_identifier("1col", "col")


def __test_validate_identifier_rejects_sql_injection_attempts__() -> None:
    """SQL-comment / semicolon / quote / whitespace all get rejected."""
    for bad in [
        "col; DROP TABLE ohlcv",
        "col--",
        "col ohlcv",
        "col'",
        'col"',
        "col`",
        "col)",
    ]:
        with pytest.raises(ValueError, match="disallowed characters"):
            _validate_identifier(bad, "col")


# ------------------------------------------------------------------
# Construction — config precedence, sentinel behavior
# ------------------------------------------------------------------

def __test_constructor_uses_defaults_when_no_overrides__(tmp_path: Path) -> None:
    """No kwargs → default table/columns are picked up so the standard
    schema Just Works."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    assert p._table == "ohlcv"
    assert p._sym_col == "symbol"
    assert p._tf_col == "timeframe"
    assert p._ts_col == "timestamp"


def __test_constructor_kwargs_override_defaults__(tmp_path: Path) -> None:
    """Explicit kwargs win over defaults — full-schema rename works."""
    db = _make_custom_schema_db(tmp_path)
    p = SQLiteProvider(
        db_path=db,
        table_name="bars",
        symbol_column="ticker",
        timeframe_column="tf",
        timestamp_column="ts",
        open_column="o",
        high_column="h",
        low_column="l",
        close_column="c",
        volume_column="v",
    )
    assert p._table == "bars"
    assert p._sym_col == "ticker"
    assert p._tf_col == "tf"


def __test_constructor_timeframe_column_none_is_distinct_from_missing__(
    tmp_path: Path,
) -> None:
    """``timeframe_column=None`` → single-timeframe DB (skip WHERE clause).
    Omitting the arg entirely uses the default ``"timeframe"``. The
    sentinel disambiguates these two cases."""
    db = _make_standard_db(tmp_path)

    p_default = SQLiteProvider(db_path=db)
    assert p_default._tf_col == "timeframe"

    p_no_tf = SQLiteProvider(db_path=db, timeframe_column=None)
    assert p_no_tf._tf_col is None


def __test_constructor_rejects_injection_in_table_name__(tmp_path: Path) -> None:
    """Table names go through _validate_identifier — semicolon rejects."""
    db = _make_standard_db(tmp_path)
    with pytest.raises(ValueError, match="disallowed characters"):
        SQLiteProvider(db_path=db, table_name="ohlcv; DROP TABLE ohlcv --")


def __test_constructor_rejects_injection_in_column_name__(tmp_path: Path) -> None:
    """Column names also validated. Every override goes through the same
    choke point in the __init__ loop."""
    db = _make_standard_db(tmp_path)
    with pytest.raises(ValueError, match="disallowed characters"):
        SQLiteProvider(db_path=db, symbol_column="symbol'; DROP TABLE ohlcv --")


# ------------------------------------------------------------------
# stream/fetch: core behavioral contract (spec §5)
# ------------------------------------------------------------------

def __test_fetch_returns_all_bars_for_symbol__(tmp_path: Path) -> None:
    """Baseline: fetch("MSFT", "1D") returns the 5 seeded bars, in
    ascending timestamp order, with close values 1..5."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    bars = p.fetch("MSFT", "1D")
    assert len(bars) == 5
    assert [b.close for b in bars] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)


def __test_stream_yields_iterator_not_list__(tmp_path: Path) -> None:
    """spec §5.1: stream() returns an Iterator, fetch() returns a list."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    result = p.stream("MSFT", "1D")
    # A generator is both Iterable and Iterator (iter(g) is g), a list is not.
    assert iter(result) is result


def __test_fetch_matches_stream__(tmp_path: Path) -> None:
    """list(stream(...)) == fetch(...) for the same range (spec §5.1)."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, tzinfo=timezone.utc)
    assert list(p.stream("MSFT", "1D", start=start, end=end)) == p.fetch(
        "MSFT", "1D", start=start, end=end
    )


def __test_fetch_partial_range_slices_bars__(tmp_path: Path) -> None:
    """Range [Jan 2, Jan 4] over Jan 1-5 seed → 3 bars."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    result = p.fetch(
        "MSFT",
        "1D",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    assert len(result) == 3
    assert result[0].close == 2.0
    assert result[-1].close == 4.0


def __test_fetch_unbounded_start_yields_from_beginning__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    result = p.fetch(
        "MSFT", "1D",
        start=None,
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert len(result) == 3
    assert result[0].close == 1.0


def __test_fetch_unbounded_end_yields_to_end__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    result = p.fetch(
        "MSFT", "1D",
        start=datetime(2024, 1, 3, tzinfo=timezone.utc),
        end=None,
    )
    assert len(result) == 3
    assert result[-1].close == 5.0


def __test_fetch_both_endpoints_unbounded__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    assert len(p.fetch("MSFT", "1D")) == 5


# ------------------------------------------------------------------
# spec §5.4 conformance checks
# ------------------------------------------------------------------

def __test_fetch_missing_symbol_returns_empty_list__(tmp_path: Path) -> None:
    """spec §5.4 check #3: unknown (symbol, timeframe) returns []
    without raising."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    assert p.fetch("NOPE", "1D") == []
    assert list(p.stream("NOPE", "1D")) == []


def __test_fetch_missing_timeframe_returns_empty_list__(tmp_path: Path) -> None:
    """Same, but the symbol exists — only the timeframe combination is
    novel. Still returns [] (partitions are AND'd)."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    assert p.fetch("MSFT", "5m") == []


def __test_fetch_reversed_range_returns_empty__(tmp_path: Path) -> None:
    """spec §5.4 check #4: start > end returns [], does NOT raise."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    result = p.fetch(
        "MSFT", "1D",
        start=datetime(2024, 1, 5, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert result == []


def __test_stream_raises_on_naive_start_datetime__(tmp_path: Path) -> None:
    """spec §5 requires timezone-aware datetime. Naive raises TypeError."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(TypeError, match="timezone-aware"):
        list(p.stream("MSFT", "1D", start=datetime(2024, 1, 1)))


def __test_stream_raises_on_naive_end_datetime__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(TypeError, match="timezone-aware"):
        list(p.stream(
            "MSFT", "1D",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5),
        ))


def __test_fetch_raises_on_naive_datetime__(tmp_path: Path) -> None:
    """fetch() delegates to stream() — same guard runs on both endpoints."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(TypeError, match="timezone-aware"):
        p.fetch("MSFT", "1D", start=datetime(2024, 1, 1))


def __test_start_and_end_are_keyword_only__(tmp_path: Path) -> None:
    """spec signature: (symbol, timeframe, *, start=..., end=...)"""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(TypeError):
        p.fetch("MSFT", "1D", datetime(2024, 1, 1, tzinfo=timezone.utc))  # type: ignore[misc]
    with pytest.raises(TypeError):
        list(p.stream("MSFT", "1D", datetime(2024, 1, 1, tzinfo=timezone.utc)))  # type: ignore[misc]


# ------------------------------------------------------------------
# Mode 2: call-time symbol/timeframe wins (no mismatch guard here)
# ------------------------------------------------------------------
# The default Provider.stream (mode 1) raises ValueError on call-time
# symbol/timeframe mismatch. SQLite overrides stream() precisely so one
# instance can serve many (symbol, timeframe) queries. These tests lock
# that in — regression if a future refactor accidentally re-inherits.

def __test_stream_does_not_enforce_construction_symbol__(tmp_path: Path) -> None:
    """Construct with symbol="MSFT", query "AAPL" — no error, AAPL rows
    (if any) are returned. Missing AAPL yields [] per §5.4 #3."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db, symbol="MSFT")
    # AAPL not in the seed → [], not ValueError.
    assert p.fetch("AAPL", "1D") == []
    # MSFT still works via the same instance.
    assert len(p.fetch("MSFT", "1D")) == 5


def __test_stream_does_not_enforce_construction_timeframe__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db, symbol="MSFT", timeframe="1D")
    # "1H" not in the seed → [], not ValueError.
    assert p.fetch("MSFT", "1H") == []


def __test_one_instance_serves_multiple_symbols__(tmp_path: Path) -> None:
    """A truly multi-symbol query: seed both MSFT and AAPL, verify one
    provider instance returns both without reconstruction."""
    db = _make_standard_db(
        tmp_path,
        rows=[
            (
                "MSFT",
                "1D",
                int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
                1.0, 1.0, 1.0, 1.0, 1.0,
            ),
            (
                "AAPL",
                "1D",
                int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
                2.0, 2.0, 2.0, 2.0, 1.0,
            ),
        ],
    )
    p = SQLiteProvider(db_path=db)
    msft = p.fetch("MSFT", "1D")
    aapl = p.fetch("AAPL", "1D")
    assert len(msft) == 1 and msft[0].close == 1.0
    assert len(aapl) == 1 and aapl[0].close == 2.0


# ------------------------------------------------------------------
# Schema flexibility
# ------------------------------------------------------------------

def __test_custom_schema_renamed_columns_work__(tmp_path: Path) -> None:
    """Every OHLCV column renamed + table renamed — fetches still hit."""
    db = _make_custom_schema_db(tmp_path)
    p = SQLiteProvider(
        db_path=db,
        table_name="bars",
        symbol_column="ticker",
        timeframe_column="tf",
        timestamp_column="ts",
        open_column="o",
        high_column="h",
        low_column="l",
        close_column="c",
        volume_column="v",
    )
    bars = p.fetch("AAPL", "1H")
    assert len(bars) == 3
    # Custom schema seeded close=i*10 for i in 1..3.
    assert [b.close for b in bars] == [10.0, 20.0, 30.0]


def __test_single_timeframe_db_via_none_column__(tmp_path: Path) -> None:
    """timeframe_column=None → WHERE clause omitted → any timeframe arg
    at call time is ignored on the SQL side."""
    db = _make_no_tf_column_db(tmp_path)
    p = SQLiteProvider(db_path=db, timeframe_column=None)
    # Any timeframe string works — the DB has no such column to filter on.
    bars = p.fetch("SPY", "whatever")
    assert len(bars) == 3


# ------------------------------------------------------------------
# Type coercion, stateless calls, gap-fill parity
# ------------------------------------------------------------------

def __test_int_columns_coerced_to_float_ohlcv_fields__(tmp_path: Path) -> None:
    """SQLite is storage-class-loose: a column authored as REAL can
    deliver a Python int if the inserted value has no fractional part.
    The OHLCV NamedTuple annotations expect float; we coerce so
    downstream code that does math on ``bar.close`` doesn't hit surprises.

    Seed 5, 10, 15 as ints — verify all OHLCV floats are actual Python
    float instances after fetch.
    """
    db = tmp_path / "ints.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "symbol TEXT, timeframe TEXT, timestamp INTEGER, "
            "open INTEGER, high INTEGER, low INTEGER, close INTEGER, volume INTEGER)"
        )
        for i, ts in enumerate(
            [
                int(datetime(2024, 1, d, tzinfo=timezone.utc).timestamp())
                for d in (1, 2, 3)
            ],
            start=1,
        ):
            conn.execute(
                "INSERT INTO ohlcv VALUES ('X', '1D', ?, ?, ?, ?, ?, ?)",
                (ts, i * 5, i * 5, i * 5, i * 5, i),
            )
        conn.commit()
    finally:
        conn.close()

    p = SQLiteProvider(db_path=db)
    bars = p.fetch("X", "1D")
    assert len(bars) == 3
    for b in bars:
        assert isinstance(b.open, float)
        assert isinstance(b.high, float)
        assert isinstance(b.low, float)
        assert isinstance(b.close, float)
        assert isinstance(b.volume, float)
        # timestamp stays int (matches the NamedTuple annotation).
        assert isinstance(b.timestamp, int)


def __test_repeated_fetch_is_stateless__(tmp_path: Path) -> None:
    """spec §5.2: no cursor state between calls."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    first = p.fetch("MSFT", "1D", start=start, end=end)
    second = p.fetch("MSFT", "1D", start=start, end=end)
    assert first == second
    assert len(first) == 5


def __test_include_gaps_flag_is_noop_and_accepted__(tmp_path: Path) -> None:
    """SQLite has no gap-fill semantics; include_gaps is accepted for
    API parity and should not affect the result."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    default = p.fetch("MSFT", "1D")
    with_gaps = p.fetch("MSFT", "1D", include_gaps=True)
    assert default == with_gaps


# ------------------------------------------------------------------
# Read-only + missing-DB behaviour
# ------------------------------------------------------------------

def __test_download_ohlcv_raises_not_implemented__(tmp_path: Path) -> None:
    """The DB is the source of truth; download_ohlcv is out of scope."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(NotImplementedError, match="read-only"):
        p.download_ohlcv(None, None)


def __test_update_symbol_info_raises_not_implemented__(tmp_path: Path) -> None:
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    with pytest.raises(NotImplementedError, match="metadata"):
        p.update_symbol_info()


def __test_get_opening_hours_returns_empty_lists__(tmp_path: Path) -> None:
    """DB carries no session metadata → empty lists (documented default)."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    intervals, starts, ends = p.get_opening_hours_and_sessions()
    assert intervals == []
    assert starts == []
    assert ends == []


def __test_get_list_of_symbols_returns_distinct_symbols__(tmp_path: Path) -> None:
    """SELECT DISTINCT symbol — one metadata query SQLite CAN answer."""
    db = _make_standard_db(
        tmp_path,
        rows=[
            ("MSFT", "1D",
             int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
             1.0, 1.0, 1.0, 1.0, 1.0),
            ("AAPL", "1D",
             int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
             1.0, 1.0, 1.0, 1.0, 1.0),
            ("MSFT", "1D",
             int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()),
             1.0, 1.0, 1.0, 1.0, 1.0),
        ],
    )
    p = SQLiteProvider(db_path=db)
    syms = sorted(p.get_list_of_symbols())
    assert syms == ["AAPL", "MSFT"]


def __test_missing_db_file_raises_operational_error_on_query__(tmp_path: Path) -> None:
    """Constructing with a nonexistent DB path is legal — we lazily open
    the connection at query time. First actual query surfaces the
    OperationalError."""
    p = SQLiteProvider(db_path=tmp_path / "does_not_exist.db")
    with pytest.raises(sqlite3.OperationalError):
        p.fetch("MSFT", "1D")


def __test_db_opened_in_read_only_mode__(tmp_path: Path) -> None:
    """A schema-mutating query issued against our connection MUST fail.
    Locks the ``mode=ro`` URI contract so a future refactor can't
    accidentally open in read/write and give a broken provider config
    the power to corrupt user data."""
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    conn = p._connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO ohlcv VALUES ('X', '1D', 0, 0, 0, 0, 0, 0)")
    finally:
        conn.close()
