"""CSVProvider — stdlib-only reference provider for RFC 4180 CSV OHLCV files.

Ships in pynecore core (Pine Extraction Design §13.2: stdlib-only providers
ship unconditionally). Reads OHLCV bars directly from a CSV file rather
than the file-backed ``.ohlcv`` round-trip used by the base ``Provider``
default — the CSV *is* the data file, so an intermediate write-then-read
would be wasted I/O.

CSV format (RFC 4180 with header row):
    timestamp,open,high,low,close,volume
    - timestamp: UTC epoch seconds (int)
    - open/high/low/close: float
    - volume: float or int

Rows are assumed already sorted by timestamp ascending; the provider does
not sort. Malformed rows raise ``ValueError`` with the offending line
number so operators can locate the bad row without a bisect.

Mode-1 (instance-scoped) per spec §5.2 — one file, one ``(symbol,
timeframe)``. Call-time ``(symbol, timeframe)`` mismatches raise the
same ``ValueError`` the base class raises, keeping conformance-suite
behavior aligned across providers.
"""
from __future__ import annotations

import csv as _csv
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
from pynecore.providers.provider import Provider
from pynecore.types.ohlcv import OHLCV


class CSVProvider(Provider):
    """Reads OHLCV bars from a single RFC 4180 CSV file.

    Overrides :meth:`stream` and :meth:`fetch` to bypass the base class's
    file-backed ``.ohlcv`` round-trip: the CSV *is* the data file. The
    base-class guards (mode-1 mismatch, naive datetime rejection,
    reversed-range → ``[]``, missing file → ``[]``) are replicated here so
    the two implementations are behaviorally interchangeable for the
    E1.4 conformance suite.

    ``include_gaps`` is accepted for API parity with the base but has no
    effect — CSV rows carry no gap-fill sentinel, so nothing to skip.
    """

    def __init__(
        self,
        csv_path: Path,
        symbol: str,
        timeframe: str,
        ohlv_dir: Path,
        config_dir: Path,
    ) -> None:
        # Store the CSV path BEFORE calling super().__init__, because the
        # base ``__init__`` triggers ``load_config()`` which we override
        # below — the override needs no attributes from ``self``, but
        # future subclasses might, so setting the CSV path first is the
        # defensive order.
        self._csv_path = Path(csv_path)
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            ohlv_dir=ohlv_dir,
            config_dir=config_dir,
        )
        # Deliberately do NOT raise on missing csv_path here — spec §5.4
        # check #3 requires missing data to return ``[]`` from stream/
        # fetch, not raise at construction. A missing file at construction
        # is a legitimate "provider defined for a symbol we haven't
        # populated yet" state.

    # ------------------------------------------------------------------ #
    # Base-class abstract methods — CSV has no exchange concept, so these
    # are no-op / identity stubs. The E1.4 conformance suite only exercises
    # stream()/fetch(); the SymInfo path is out of scope for a
    # file-backed reference provider.
    # ------------------------------------------------------------------ #

    @classmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return timeframe

    @classmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        return timeframe

    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        # A CSVProvider is scoped to a single file/symbol at construction.
        assert self.symbol is not None
        return [self.symbol]

    def update_symbol_info(self):  # pragma: no cover - not exercised
        # CSV files don't ship exchange metadata; delegate the decision
        # to whichever caller needs SymInfo (they shouldn't, for a
        # file-backed reference provider).
        raise NotImplementedError(
            "CSVProvider does not model exchange metadata; "
            "wrap the CSV in a real Provider if you need SymInfo."
        )

    @classmethod
    def get_opening_hours_and_sessions(cls) -> tuple[
        list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]
    ]:
        return [], [], []

    def load_config(self) -> None:
        # Skip the providers.toml lookup — CSVProvider has no per-file
        # config knobs and the base ``load_config()`` would open a
        # non-existent providers.toml under an isolated tmp_path.
        self.config = {}

    def download_ohlcv(  # type: ignore[override]
        self,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        on_progress: Callable[[datetime], None] | None = None,
        limit: int | None = None,
    ) -> None:
        # No-op: the CSV IS the data file. The base-class file-backed flow
        # would round-trip through .ohlcv, but we override stream/fetch
        # below to read the CSV directly, so there's nothing to download.
        return

    # ------------------------------------------------------------------ #
    # Direct-CSV reading — overrides Provider.stream / Provider.fetch
    # ------------------------------------------------------------------ #

    def _iter_csv_bars(self) -> Iterator[OHLCV]:
        """Yield every OHLCV row from the CSV, unfiltered."""
        with self._csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSVProvider: {self._csv_path} missing columns: "
                    f"{sorted(missing)}"
                )
            # Header is line 1; first data row is line 2.
            for lineno, row in enumerate(reader, start=2):
                try:
                    yield OHLCV(
                        timestamp=int(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(
                        f"CSVProvider: malformed row at "
                        f"{self._csv_path}:{lineno}: {e}"
                    ) from e

    def stream(  # type: ignore[override]
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        include_gaps: bool = False,  # accepted for API parity; CSV has no gap markers
    ) -> Iterator[OHLCV]:
        """Yield OHLCV bars from the CSV, filtered to ``[start, end]``.

        Replicates the base-class guards so the E1.4 conformance suite
        sees identical behavior across providers:

        - Mode-1 mismatch (call-time ``(symbol, timeframe)`` differs from
          construction-time) → ``ValueError`` (spec §5.4 check #5).
        - Naive datetimes → ``TypeError`` (spec §5).
        - ``start > end`` → yield nothing (spec §5.4 check #4).
        - Missing CSV file → yield nothing (spec §5.4 check #3).
        - Bounds are inclusive on both sides.
        """
        # Guard: mode-1 symbol/timeframe mismatch. Matches Provider.stream.
        if self.symbol is not None and symbol != self.symbol:
            raise ValueError(
                f"call-time symbol {symbol!r} does not match construction-time "
                f"{self.symbol!r}; CSVProvider is mode-1 (single-symbol per "
                "instance, spec §5.2)."
            )
        if self.timeframe is not None and timeframe != self.timeframe:
            raise ValueError(
                f"call-time timeframe {timeframe!r} does not match construction-"
                f"time {self.timeframe!r}; CSVProvider is mode-1 (single-"
                "timeframe per instance, spec §5.2)."
            )

        # Guard: naive datetimes are ambiguous cross-machine (spec §5).
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

        # Guard: reversed range → empty. An early ``return`` inside a
        # generator terminates it immediately, yielding nothing.
        if start is not None and end is not None and start > end:
            return

        # Guard: missing CSV file → empty (spec §5.4 check #3). This
        # mirrors the base class's FileNotFoundError → [] translation and
        # keeps the two impls interchangeable for the conformance suite.
        if not self._csv_path.is_file():
            return

        # Convert bounds to epoch seconds so we compare like against
        # like — the CSV timestamp is an int, and datetime comparison
        # against int would TypeError.
        start_ts = int(start.timestamp()) if start is not None else None
        end_ts = int(end.timestamp()) if end is not None else None

        for bar in self._iter_csv_bars():
            if start_ts is not None and bar.timestamp < start_ts:
                continue
            if end_ts is not None and bar.timestamp > end_ts:
                # Rows are pre-sorted ascending — safe to short-circuit.
                break
            yield bar

    def fetch(  # type: ignore[override]
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        include_gaps: bool = False,  # accepted for API parity; CSV has no gap markers
    ) -> list[OHLCV]:
        """Materialize ``stream(...)`` as a list.

        Kept as an explicit override (rather than inheriting the base
        default) so the ``include_gaps`` kwarg forwards cleanly and the
        docstring reflects the CSV-specific "no-op include_gaps"
        semantics. Behavior is otherwise identical to
        ``list(self.stream(...))``.
        """
        return list(self.stream(
            symbol, timeframe, start=start, end=end, include_gaps=include_gaps,
        ))
