"""SQLite reference provider (bd-l05, Task E1.3 of the Pine Extraction plan).

Reads OHLCV bars from an existing SQLite database via Python's stdlib
``sqlite3`` module — no third-party dependencies. Ships in pynecore core.

Design rationale
----------------
The SQLite provider is a *reference implementation* of the mode-2 provider
pattern (spec §5.2): a single instance can serve queries for many
``(symbol, timeframe)`` pairs because the storage layer already partitions
by those keys. We therefore override the default :meth:`stream` and
:meth:`fetch` from :class:`Provider` — the file-backed default is mode-1
(single ``.ohlcv`` file per instance) and would reject call-time symbol
mismatches with ``ValueError``.

The **schema is user-configurable** because databases are almost never
canonical:

- ``table_name`` — which table to read from (default ``ohlcv``)
- ``symbol_column``, ``timeframe_column``, ``timestamp_column``,
  ``open_column``, ``high_column``, ``low_column``, ``close_column``,
  ``volume_column`` — column names for each OHLCV field
- ``timeframe_column=None`` — for single-timeframe DBs (skip the
  timeframe WHERE clause entirely)

Config precedence: constructor kwargs override ``providers.toml``
``[sqlite]`` section, which overrides the defaults above.

Behavioral contract (spec §5.4)
-------------------------------
1. **Call-parameterized**: ``stream(symbol, timeframe, ...)`` and
   ``fetch(symbol, timeframe, ...)`` accept ``(symbol, timeframe)`` at call
   time and honor them — one instance may serve many symbols/timeframes.
2. **Missing symbol returns empty**: an unknown ``(symbol, timeframe)`` in
   the DB returns ``[]`` (not a ``KeyError`` or other exception).
3. **Reversed range**: ``start > end`` returns ``[]`` — SQL-consistent
   (spec §5.4 conformance check #4).
4. **Timezone-aware**: naive ``datetime`` for ``start``/``end`` raises
   ``TypeError``. Timestamps stored in SQL are treated as UTC epoch
   seconds — same convention as :class:`~pynecore.types.ohlcv.OHLCV`.
5. **include_gaps**: accepted for API parity with :meth:`Provider.stream`
   but ignored — SQLite has no gap-fill semantics (unlike the ``.ohlcv``
   file format which writes ``volume=-1`` sentinel rows to preserve
   interval regularity).
6. **Read-only**: the DB is opened with URI ``mode=ro`` so a broken
   provider config cannot corrupt user data.

The abstract ``download_ohlcv`` / ``update_symbol_info`` from
:class:`Provider` do not apply to a read-from-user-DB provider — they
raise :class:`NotImplementedError` with a message pointing users to the
correct entry points (``stream``/``fetch``).

Clean-room: I have not viewed TradingView or PyneComp source code.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

# Python 3.12+ ships ``typing.override``; earlier versions get a no-op
# decorator so downstream ``@override`` annotations stay valid.
if sys.version_info >= (3, 12):
    from typing import override
else:
    def override(func):
        return func

from .provider import Provider

from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from ..types.ohlcv import OHLCV

__all__ = ["SQLiteProvider"]


# Defaults for the OHLCV table schema. Users override any subset via
# constructor kwargs (highest precedence) or the ``providers.toml``
# ``[sqlite]`` section (lower precedence).
_DEFAULT_TABLE = "ohlcv"
_DEFAULT_COLUMNS: dict[str, str] = {
    "symbol_column": "symbol",
    "timeframe_column": "timeframe",
    "timestamp_column": "timestamp",
    "open_column": "open",
    "high_column": "high",
    "low_column": "low",
    "close_column": "close",
    "volume_column": "volume",
}

# SQLite identifier whitelist. We NEVER interpolate arbitrary user input
# into SQL identifiers (table + column names) — the ``?`` placeholder only
# works for values. Instead we validate that each identifier is a plain
# ASCII SQL-friendly name, then embed it via f-string. This is safer than
# trying to escape [] or "" quoting rules, and matches SQLite's rule that
# an identifier is a keyword or ``[A-Za-z_][A-Za-z_0-9]*``.
_IDENT_ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
)


# Module-level sentinel — an instance of a private class so ``is``
# comparisons are cheap and unambiguous. Using ``object()`` would work
# but a named class gives cleaner tracebacks and prevents third parties
# from accidentally sharing a "generic" sentinel.
class _MissingType:
    """Marker class for "argument was not supplied" — distinct from
    ``None`` which for ``timeframe_column`` is a legitimate value
    (meaning "single-timeframe DB, skip that WHERE clause").
    """

    _singleton: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<MISSING>"


_MISSING = _MissingType()


def _validate_identifier(name: str, kind: str) -> str:
    """Raise ``ValueError`` if ``name`` is not a plain SQL identifier.

    We refuse anything that isn't ``[A-Za-z_][A-Za-z_0-9]*``. This closes
    the SQL-injection door on the table/column-name configuration surface
    — parameter placeholders (``?``) cannot bind identifiers, so we must
    validate them by hand before embedding.

    Empty strings, names starting with a digit, and names containing
    whitespace / quotes / semicolons / SQL comments all fail here.

    :param name: The identifier candidate.
    :param kind: Human-readable label used in the error message (e.g.
        ``"table_name"``) so config bugs are diagnosable at a glance.
    :return: ``name`` unchanged (so this can be used inline).
    :raises ValueError: If ``name`` is empty, starts with a digit, or
        contains any character outside ``[A-Za-z_0-9]``.
    """
    if not name:
        raise ValueError(f"{kind} must be a non-empty SQL identifier")
    if name[0].isdigit():
        raise ValueError(
            f"{kind}={name!r} must not start with a digit "
            "(not a valid SQL identifier)"
        )
    bad = set(name) - _IDENT_ALLOWED
    if bad:
        raise ValueError(
            f"{kind}={name!r} contains disallowed characters {sorted(bad)!r}; "
            "only ASCII letters, digits, and underscore are permitted"
        )
    return name


class SQLiteProvider(Provider):
    """Read OHLCV bars from a user-supplied SQLite database.

    The provider is *stateless across calls*: one instance can serve
    ``(symbol, timeframe)`` queries for every row in the DB, since the
    storage engine partitions by those keys. Instantiate with the DB
    path and any non-default column names; then call :meth:`stream` or
    :meth:`fetch` per query.

    :param db_path: Absolute or relative path to the SQLite DB file.
        REQUIRED — no default. Must exist and be readable; the DB is
        opened with ``mode=ro`` so writes never leak from us.
    :param symbol: Advisory only for this provider — the SQL layer
        supports true multi-symbol queries via the call-time argument.
        Passed through to :class:`Provider` for API parity.
    :param timeframe: Same — advisory, call-time value wins.
    :param ohlv_dir: Passed through to :class:`Provider` (used only for
        the ``.ohlcv`` file path that this provider does not write).
    :param config_dir: Optional; if provided and ``providers.toml``
        exists in it, config values there are consulted for defaults.
    :param table_name: Override the OHLCV table name (default ``"ohlcv"``).
    :param symbol_column: Column holding the symbol (default ``"symbol"``).
    :param timeframe_column: Column holding the timeframe, or ``None``
        for single-timeframe DBs where every row is implicitly the same
        timeframe (default ``"timeframe"``). Distinguishes "not passed"
        from "explicitly None" via a sentinel.
    :param timestamp_column: Column holding Unix epoch seconds (int)
        (default ``"timestamp"``).
    :param open_column, high_column, low_column, close_column, volume_column:
        Column names for the OHLCV numeric fields (defaults match the
        field name).
    """

    # Advertise the configurable keys so ``providers.toml`` scaffolding
    # in downstream tooling knows what to render. Values are all empty
    # strings — the real defaults live in ``_DEFAULT_TABLE`` /
    # ``_DEFAULT_COLUMNS`` and are applied at ``__init__`` time.
    config_keys: dict[str, str] = {
        "# SQLite reference provider — reads OHLCV rows from a user DB.": "",
        "# All keys optional; constructor kwargs override these.": "",
        "db_path": "",
        "table_name": "",
        "symbol_column": "",
        "timeframe_column": "",
        "timestamp_column": "",
        "open_column": "",
        "high_column": "",
        "low_column": "",
        "close_column": "",
        "volume_column": "",
    }

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """Identity — the SQLite DB stores whatever the user put there;
        we do not attempt a canonicalisation pass.

        Downstream code that needs a canonical TradingView-format
        timeframe should normalize *before* handing the string here.
        """
        return timeframe

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """Identity — see :meth:`to_tradingview_timeframe`.

        SQLite is a passive data store; there is no "exchange format" to
        translate to.
        """
        return timeframe

    @override
    def __init__(
        self,
        *,
        db_path: str | Path,
        symbol: str | None = None,
        timeframe: str | None = None,
        ohlv_dir: Path | None = None,
        config_dir: Path | None = None,
        table_name: str | None = None,
        symbol_column: str | None = None,
        timeframe_column: Any = _MISSING,
        timestamp_column: str | None = None,
        open_column: str | None = None,
        high_column: str | None = None,
        low_column: str | None = None,
        close_column: str | None = None,
        volume_column: str | None = None,
    ) -> None:
        # NOTE: we DO NOT call ``super().__init__`` because the parent
        # constructor unconditionally opens ``providers.toml`` and
        # refuses to instantiate without an ``ohlv_dir``. Both are noisy
        # for a "just read a DB" use case. Instead we replicate the
        # minimum parent bookkeeping needed by API parity, and consult
        # ``providers.toml`` only when a config_dir is supplied.
        self.symbol = symbol
        self.timeframe = timeframe
        self.xchg_timeframe = timeframe  # no translation for SQLite
        self.ohlcv_path = None  # this provider does not write .ohlcv files
        self.ohlcv_file = None
        self.config_dir = config_dir
        self.config = {}

        if config_dir is not None:
            toml_path = Path(config_dir) / "providers.toml"
            if toml_path.exists():
                # Reuse the parent's load_config semantics only when the
                # file is actually present — silent absence is the norm
                # for ad-hoc SQL queries with no ``.openbb_platform`` dir.
                self.load_config()

        # Coalesce config: constructor kwarg > providers.toml > default.
        self._db_path = Path(db_path)
        self._table = _validate_identifier(
            table_name or self.config.get("table_name") or _DEFAULT_TABLE,
            kind="table_name",
        )

        # ``timeframe_column`` uses a sentinel default so users can
        # explicitly pass ``None`` (single-timeframe DB — skip that WHERE
        # clause). ``None`` is a legitimate value here, not just "not
        # provided" — sentinel disambiguates.
        if timeframe_column is _MISSING:
            tfc_cfg = self.config.get("timeframe_column")
            self._tf_col: str | None = (
                _validate_identifier(tfc_cfg, "timeframe_column")
                if tfc_cfg
                else _DEFAULT_COLUMNS["timeframe_column"]
            )
        elif timeframe_column is None:
            self._tf_col = None  # single-timeframe DB
        else:
            self._tf_col = _validate_identifier(
                timeframe_column, "timeframe_column"
            )

        # Fold the remaining column overrides into a single dict for a
        # tight validation loop. Every value flows through
        # ``_validate_identifier`` before it ever touches a SQL string —
        # this is the choke point that protects us from injection through
        # a compromised providers.toml.
        overrides: dict[str, str | None] = {
            "symbol_column": symbol_column,
            "timestamp_column": timestamp_column,
            "open_column": open_column,
            "high_column": high_column,
            "low_column": low_column,
            "close_column": close_column,
            "volume_column": volume_column,
        }
        resolved: dict[str, str] = {}
        for key, kwarg in overrides.items():
            resolved[key] = _validate_identifier(
                kwarg or self.config.get(key) or _DEFAULT_COLUMNS[key],
                kind=key,
            )
        self._sym_col = resolved["symbol_column"]
        self._ts_col = resolved["timestamp_column"]
        self._o_col = resolved["open_column"]
        self._h_col = resolved["high_column"]
        self._l_col = resolved["low_column"]
        self._c_col = resolved["close_column"]
        self._v_col = resolved["volume_column"]

    # ------------------------------------------------------------------
    # Abstract-method stubs
    # ------------------------------------------------------------------
    # ``Provider`` marks these ``@abstractmethod`` so we must define them.
    # The SQLite provider does not download data (the DB is the source of
    # truth) nor update symbol info (the DB has no metadata channel), so
    # they raise with an actionable message rather than silently no-oping.

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """Return every distinct ``symbol`` value present in the OHLCV table.

        This is one of the few metadata operations SQLite CAN answer
        cheaply — a ``SELECT DISTINCT symbol`` uses the same index used
        by the range queries. Convenient for exploratory sessions where
        the caller doesn't yet know what's in the DB.

        :return: List of symbol strings in insertion order (no sort;
            caller can sort if stable ordering is needed).
        """
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT DISTINCT {self._sym_col} FROM {self._table}"
            )
            # ``row[0]`` because we selected exactly one column.
            return [row[0] for row in cur.fetchall()]

    @override
    def update_symbol_info(self) -> SymInfo:
        """SQLite has no exchange metadata channel — ``get_symbol_info``
        cannot synthesise a ``SymInfo`` here.

        A caller that needs full :class:`SymInfo` should hand-write the
        ``.toml`` next to the DB, or use a proper metadata-carrying
        provider (CCXT, Capital.com, FMP).
        """
        raise NotImplementedError(
            "SQLiteProvider does not synthesize symbol metadata. "
            "Provide a hand-authored SymInfo TOML alongside the DB, or "
            "use a metadata-carrying provider (CCXT, Capital.com, FMP)."
        )

    @override
    def get_opening_hours_and_sessions(
        self,
    ) -> tuple[list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """Same rationale as :meth:`update_symbol_info` — the DB does not
        carry trading-session metadata. Return empty lists so downstream
        code that treats sessions as "unknown → no session filter"
        continues to work without a special case.
        """
        return [], [], []

    @override
    def download_ohlcv(
        self,
        time_from: datetime | None,
        time_to: datetime | None,
        on_progress: Callable[[datetime], None] | None = None,
        limit: int | None = None,
    ) -> None:
        """The SQLite provider READS from an existing DB — writing new
        rows is out of scope. Use ``sqlite3`` directly or a dedicated
        ingest tool to populate the DB, then instantiate this provider
        to query.
        """
        raise NotImplementedError(
            "SQLiteProvider is read-only; populate the DB via your own "
            "ingest process (or another provider's download_ohlcv), then "
            "read back with SQLiteProvider.stream()/fetch()."
        )

    # ------------------------------------------------------------------
    # The behavioral contract (spec §5)
    # ------------------------------------------------------------------

    @override
    def stream(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        include_gaps: bool = False,
    ) -> Iterator[OHLCV]:
        """Yield OHLCV bars from the SQLite DB for ``(symbol, timeframe)``.

        This override bypasses the Provider-base file-backed default so
        one instance can serve queries for many ``(symbol, timeframe)``
        pairs (spec §5.2 mode 2). The construction-time ``self.symbol`` /
        ``self.timeframe`` are advisory only; the call-time arguments
        win — no mismatch guard here.

        The query is one ``SELECT ... WHERE symbol = ? [AND timeframe = ?]
        [AND timestamp >= ?] [AND timestamp <= ?] ORDER BY timestamp``,
        parameterised so no user string is interpolated into SQL.

        :param symbol: Value bound to the ``symbol_column`` WHERE clause.
        :param timeframe: Value bound to the ``timeframe_column`` WHERE
            clause. IGNORED (and the WHERE clause omitted) when the
            provider was constructed with ``timeframe_column=None``
            (single-timeframe DB).
        :param start: Optional inclusive lower bound (UTC epoch seconds
            derived via ``int(start.timestamp())``). Naive datetime raises
            ``TypeError``.
        :param end: Optional inclusive upper bound. Naive raises.
        :param include_gaps: Accepted for API parity with
            :meth:`Provider.stream`. SQLite has no gap-fill semantics
            (no ``volume=-1`` sentinel rows) so the flag is a no-op.
        :yields: :class:`OHLCV` rows in ascending timestamp order.
        :raises TypeError: If ``start`` or ``end`` is naive.
        """
        # Guard: naive datetimes silently interpret in local tz — a cross-
        # machine reproducibility footgun. Same rule as Provider.stream.
        if start is not None and start.tzinfo is None:
            raise TypeError(
                "start must be a timezone-aware datetime (spec §5); "
                "got naive datetime which is ambiguous across timezones."
            )
        if end is not None and end.tzinfo is None:
            raise TypeError(
                "end must be a timezone-aware datetime (spec §5); "
                "got naive datetime which is ambiguous across timezones."
            )

        # Reversed range → empty (spec §5.4 check #4). Terminating a
        # generator with a bare ``return`` yields no bars.
        if start is not None and end is not None and start > end:
            return

        # ``include_gaps`` accepted for parity with Provider.stream but
        # ignored here — SQLite has no gap-fill semantics (unlike the
        # .ohlcv file which writes volume=-1 sentinel rows to preserve
        # interval regularity). Callers that need gap synthesis should
        # do it downstream from a DB whose bars are already contiguous,
        # or resample after fetching.
        _ = include_gaps

        # Build the WHERE clause dynamically. All identifiers were
        # validated at __init__ time; user data flows through ``?``
        # placeholders. Order of clauses does not affect the query plan
        # (SQLite reorders by selectivity via the query planner).
        wheres: list[str] = [f"{self._sym_col} = ?"]
        params: list[object] = [symbol]
        if self._tf_col is not None:
            wheres.append(f"{self._tf_col} = ?")
            params.append(timeframe)
        if start is not None:
            wheres.append(f"{self._ts_col} >= ?")
            params.append(int(start.timestamp()))
        if end is not None:
            wheres.append(f"{self._ts_col} <= ?")
            params.append(int(end.timestamp()))

        sql = (
            f"SELECT {self._ts_col}, {self._o_col}, {self._h_col}, "
            f"{self._l_col}, {self._c_col}, {self._v_col} "
            f"FROM {self._table} "
            f"WHERE {' AND '.join(wheres)} "
            f"ORDER BY {self._ts_col} ASC"
        )

        # ``with self._connect()`` closes the connection when the
        # generator is exhausted OR garbage-collected mid-iteration
        # (Python guarantees the finally-block runs when the generator
        # is closed). Missing DB rows return no results without raising.
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            # Iterate cursor lazily so a million-row query does not
            # materialise into memory. ``fetchmany`` could tune the
            # server-side batch — sqlite3 already batches internally, so
            # the naive ``for row in cur`` is optimal for stdlib.
            for row in cur:
                # Cast to the OHLCV field types explicitly. Sqlite is
                # storage-class-loose (REAL vs INTEGER vs TEXT), and a
                # column authored as INTEGER can silently deliver a
                # Python ``int`` where the OHLCV NamedTuple expects a
                # float. int() / float() normalise before yielding.
                yield OHLCV(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )

    @override
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        include_gaps: bool = False,
    ) -> list[OHLCV]:
        """Materialised counterpart to :meth:`stream`.

        Equivalent to ``list(self.stream(...))`` — same query, same
        semantics, but returned as a list. Documented separately per
        spec §5.1 (``stream`` returns an ``Iterator``, ``fetch`` returns
        a ``list``) so callers can pick based on memory-vs-latency
        preferences.

        For SQLite specifically, ``fetch`` does not batch differently
        from ``stream``: the underlying cursor is iterated once and
        results collected. The overhead is one Python list allocation.
        """
        return list(
            self.stream(
                symbol,
                timeframe,
                start=start,
                end=end,
                include_gaps=include_gaps,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only URI connection to the configured DB.

        Using ``mode=ro`` means:

        - A broken provider config cannot mutate the DB.
        - Concurrent readers do not block on writer locks (WAL not
          required; ``mode=ro`` is compatible with any journal mode).
        - Missing DB file raises :class:`sqlite3.OperationalError`
          rather than silently creating an empty one (the default
          ``sqlite3.connect(path)`` behaviour is to create).

        :return: Open :class:`sqlite3.Connection`. Caller is responsible
            for closing (typically via ``with`` block, which
            :meth:`stream` uses).
        :raises sqlite3.OperationalError: If the DB file cannot be
            opened (missing / permissions / not a valid SQLite DB).
        """
        # ``as_posix()`` because sqlite3 URI parsing on Windows chokes on
        # backslashes even in the file: prefix.
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)
