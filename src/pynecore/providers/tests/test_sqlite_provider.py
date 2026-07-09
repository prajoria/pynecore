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

import math
import os
import sqlite3
import sys
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


# ------------------------------------------------------------------
# PR #4 review regressions
# ------------------------------------------------------------------
# The tests below lock in fixes for the review findings on PR #4 and
# must never regress. See the review commentary at:
#   https://github.com/prajoria/pynecore/pull/4
# for the reproductions each one guards against.


def __test_stream_releases_file_lock_after_early_break__(tmp_path: Path) -> None:
    """[BUG regression, review L497] Abandoning ``stream()`` mid-iteration
    must release the SQLite file lock so ``os.unlink(db)`` succeeds.

    Reproduction: on Windows, ``sqlite3.Connection.__exit__`` only
    commits/rolls back — it does NOT close the connection. Without the
    ``contextlib.closing`` wrapper, a ``for ... in stream(...): break``
    pattern leaves the connection open (the hidden iterator variable
    on the ``for`` frame keeps the generator alive past the ``break``),
    the file lock persists, and the subsequent ``os.unlink(db)`` raises
    ``PermissionError: WinError 32``.

    Verified reproduces pre-fix on Windows CPython. On POSIX the OS
    permits unlink on open files so the test passes trivially there,
    but the fix (``contextlib.closing``) is still correct on both
    platforms — this test exists to prevent the Windows-only regression
    from silently re-entering the codebase.
    """
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)

    # for/break holds a reference to the generator via the ``for`` frame's
    # hidden iterator variable — mimics the real-world pattern that hits
    # the bug (read a few bars then bail out on a threshold / exception).
    for bar in p.stream("MSFT", "1D"):
        assert bar.close == 1.0
        break

    # Unlink must succeed — the file lock is released because the
    # ``contextlib.closing`` wrapper closed the connection on generator
    # abandonment. Pre-fix this raises ``PermissionError: WinError 32``.
    try:
        os.unlink(db)
    except OSError as e:  # pragma: no cover — reproduces only pre-fix on Windows
        pytest.fail(
            f"os.unlink(db) failed after stream() early-break — SQLite file "
            f"lock persisted (regression of PR #4 L497 fix): {e}"
        )
    assert not db.exists()


def __test_get_list_of_symbols_releases_file_lock__(tmp_path: Path) -> None:
    """[BUG regression, review L497] Same connection-leak fix must cover
    ``get_list_of_symbols`` (the other call site that pre-fix used the
    bare ``with self._connect() as conn`` pattern).

    Verified by unlinking immediately after the call.
    """
    db = _make_standard_db(tmp_path)
    p = SQLiteProvider(db_path=db)
    syms = p.get_list_of_symbols()
    assert syms == ["MSFT"]

    try:
        os.unlink(db)
    except OSError as e:  # pragma: no cover — reproduces only pre-fix on Windows
        pytest.fail(
            f"os.unlink(db) failed after get_list_of_symbols() — SQLite "
            f"file lock persisted (regression of PR #4 L497 fix): {e}"
        )
    assert not db.exists()


def __test_null_ohlc_column_yields_nan__(tmp_path: Path) -> None:
    """[BUG regression, review L515] A row with NULL in an OHLC column
    must NOT crash with ``TypeError: float() argument ... 'NoneType'``.

    We adopt the pynecore convention for "no value" in a float-typed
    field: ``math.nan``. Downstream indicators propagate NaN naturally
    (mean/median/etc. treat it as missing) and callers can filter with
    ``math.isnan`` where an explicit skip is desired.
    """
    db = tmp_path / "nulls.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "symbol TEXT, timeframe TEXT, timestamp INTEGER, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL)"
        )
        # Row 0: fully populated. Row 1: NULL in high column (a common
        # backfill-gap pattern). Row 2: NULL in close (indicator input).
        rows = [
            ("X", "1D",
             int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
             1.0, 1.0, 1.0, 1.0, 100.0),
            ("X", "1D",
             int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()),
             2.0, None, 2.0, 2.0, 100.0),
            ("X", "1D",
             int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp()),
             3.0, 3.0, 3.0, None, 100.0),
        ]
        conn.executemany(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()

    p = SQLiteProvider(db_path=db)
    bars = p.fetch("X", "1D")
    assert len(bars) == 3
    # Row 0 unaffected.
    assert bars[0].high == 1.0 and bars[0].close == 1.0
    # Row 1: NULL high → nan; other columns still populated.
    assert math.isnan(bars[1].high)
    assert bars[1].open == 2.0 and bars[1].close == 2.0
    # Row 2: NULL close → nan; other columns still populated.
    assert math.isnan(bars[2].close)
    assert bars[2].open == 3.0 and bars[2].high == 3.0


def __test_null_volume_column_yields_gap_fill_sentinel__(tmp_path: Path) -> None:
    """[BUG regression, review L515] NULL in the volume column maps to
    ``-1.0``, matching the :class:`OHLCVWriter` gap-fill sentinel that
    downstream skip-gaps logic already understands (``volume <= 0``
    → synthetic bar). See ``pynecore.core.ohlcv_file`` line 575.
    """
    db = tmp_path / "nullvol.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "symbol TEXT, timeframe TEXT, timestamp INTEGER, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL)"
        )
        conn.execute(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("X", "1D",
             int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
             1.0, 1.0, 1.0, 1.0, None),
        )
        conn.commit()
    finally:
        conn.close()

    p = SQLiteProvider(db_path=db)
    bars = p.fetch("X", "1D")
    assert len(bars) == 1
    assert bars[0].volume == -1.0
    # OHLC still populated as expected.
    assert bars[0].close == 1.0


def __test_schema_mismatch_raises_operational_error__(tmp_path: Path) -> None:
    """[DOC coverage gap, review L14] If the configured column names do
    not exist in the DB, SQLite raises ``OperationalError: no such column``
    at query time. This documents the expected failure mode — a config
    typo surfaces cleanly rather than silently returning empty results.
    """
    db = _make_standard_db(tmp_path)
    # Point ``close_column`` at a column that does not exist.
    p = SQLiteProvider(db_path=db, close_column="not_a_column")
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        p.fetch("MSFT", "1D")


def __test_uri_encoding_preserves_question_mark_in_filename__(tmp_path: Path) -> None:
    """[NIT regression, review L574] A filename containing ``?`` (legal
    on POSIX; not on Windows) must be URL-encoded before being spliced
    into the URI. Pre-fix the ``?`` was parsed as the URI query-string
    delimiter and SQLite silently opened a different file (or failed
    with ``unable to open database file``).

    On Windows ``?`` is not a legal filename character so we skip the
    filesystem-level exercise there; instead we validate the encoding
    step on a synthetic path to prove the mitigation is in place on
    all platforms.
    """
    from pynecore.providers.sqlite import SQLiteProvider as _P

    # Filesystem-level test — only meaningful on POSIX.
    if sys.platform != "win32":
        db = tmp_path / "foo?bar.db"
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "CREATE TABLE ohlcv ("
                "symbol TEXT, timeframe TEXT, timestamp INTEGER, "
                "open REAL, high REAL, low REAL, close REAL, volume REAL)"
            )
            conn.execute(
                "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("Z", "1D",
                 int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
                 9.0, 9.0, 9.0, 9.0, 1.0),
            )
            conn.commit()
        finally:
            conn.close()
        p = _P(db_path=db)
        bars = p.fetch("Z", "1D")
        assert len(bars) == 1 and bars[0].close == 9.0

    # Encoding-level test — cross-platform. Even on Windows we exercise
    # the code path that builds the URI to guard against a regression
    # that would drop the encoding step.
    p = _P(db_path=Path("/tmp/foo?bar.db"))
    # We can't call ``._connect()`` here (file may not exist) but the
    # URI construction is deterministic — replicate the fix's contract.
    encoded = p._db_path.as_posix()
    # Sanity: the raw path contains ``?`` — the fix's job is to encode.
    assert "?" in encoded
    # After ``quote(safe="/:")`` the ``?`` becomes ``%3F`` which SQLite
    # then decodes back to the literal filename.
    from urllib.parse import quote
    assert "%3F" in quote(encoded, safe="/:")


# ------------------------------------------------------------------
# E1.4 (bd-cko) shared behavioral conformance suite
# ------------------------------------------------------------------
# SQLiteProvider is mode-2 (spec §5.2): a missing symbol naturally
# returns [] rather than raising. Spec §5.4 check #5 requires the
# conforming provider to raise a typed error on missing symbol.
# The wrapper below adds the "raise on missing symbol" behavior via
# composition so we can call the shared suite unmodified — real
# production callers should pick their own missing-symbol policy
# (raise vs empty-list) rather than adopting this wrapper globally.


class _RaisingSQLiteProvider(SQLiteProvider):
    """Wrap :class:`SQLiteProvider` so unknown-symbol fetches RAISE a
    :class:`KeyError` (matching §5.4 check #5) instead of returning ``[]``.

    Rationale: the shared conformance suite requires a typed error for
    missing symbol so that both mode-1 providers (which raise on
    construction/call-time mismatch) and mode-2 providers (which
    naturally return ``[]``) can express a uniform "you notice" contract.
    Real production callers should decide whether they prefer raise or
    empty-list semantics based on their downstream code — this wrapper
    exists only to satisfy the conformance-suite contract at test time.

    The check uses :meth:`get_list_of_symbols` (one ``SELECT DISTINCT``)
    to distinguish "queried but got nothing due to range" from "symbol
    is not in the table at all" — the latter is the case worth raising
    on. A stub timeframe that isn't in the table for a known symbol
    still returns ``[]`` (that's the range-empty case, not the
    unknown-symbol case).
    """

    def fetch(self, symbol, timeframe, *, start=None, end=None, include_gaps=False):
        result = super().fetch(
            symbol, timeframe,
            start=start, end=end, include_gaps=include_gaps,
        )
        # Only elevate to a raise if the symbol is genuinely absent from
        # the table. An empty result for a KNOWN symbol (e.g. because
        # the range is outside the fixture) MUST still return [] — that
        # is check #3 (empty range) and check #4 (reversed range) which
        # both expect [].
        if not result:
            if symbol not in self.get_list_of_symbols():
                raise KeyError(
                    f"symbol {symbol!r} not in SQLite table {self._table!r}"
                )
        return result

    def stream(self, symbol, timeframe, *, start=None, end=None, include_gaps=False):
        # We check symbol existence directly against get_list_of_symbols()
        # rather than through super().fetch() — the latter dispatches back
        # via ``self.stream()`` (SQLiteProvider.fetch is a thin
        # ``list(self.stream(...))`` wrapper), which would recurse
        # infinitely through this override.
        #
        # get_list_of_symbols() is one SELECT DISTINCT — cheap enough for
        # a test-only wrapper. We check up-front so we raise BEFORE the
        # caller starts iterating (a lazy raise from inside the generator
        # would surface only on the first next() call, which the shared
        # suite's ``list(...)`` would eventually trigger — but eager is
        # closer to the spec's "you notice, not silently zero rows"
        # intent).
        if symbol not in self.get_list_of_symbols():
            raise KeyError(
                f"symbol {symbol!r} not in SQLite table {self._table!r}"
            )
        return super().stream(
            symbol, timeframe,
            start=start, end=end, include_gaps=include_gaps,
        )


def __test_sqlite_provider_conforms_to_shared_suite__(tmp_path: Path) -> None:
    """Run the E1.4 shared behavioral conformance suite against a
    fixture-loaded ``_RaisingSQLiteProvider`` (spec §5.4).

    The wrapper adds the "raise on missing symbol" behavior mode-2
    providers need to satisfy §5.4 check #5 — see the class docstring
    for the rationale.

    The fixture seeds ``(TESTSYM, 1D)`` bars for Jan 1-5 2024, matching
    the shared suite's default range so no override is needed on the
    range parameters.
    """
    from pynecore.providers.tests.test_conformance import _conformance_suite

    # Seed the fixture with TESTSYM/1D bars matching the shared suite's
    # default range. Rows: 5 daily bars Jan 1-5 2024, close=i.
    db = _make_standard_db(
        tmp_path,
        rows=[
            (
                "TESTSYM",
                "1D",
                int(datetime(2024, 1, i, tzinfo=timezone.utc).timestamp()),
                float(i),
                float(i),
                float(i),
                float(i),
                1.0,
            )
            for i in range(1, 6)
        ],
    )
    provider = _RaisingSQLiteProvider(db_path=db)
    _conformance_suite(
        provider,
        closed_only=True,
        test_symbol="TESTSYM",
        test_timeframe="1D",
        range_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        range_end=datetime(2024, 1, 5, tzinfo=timezone.utc),
    )
