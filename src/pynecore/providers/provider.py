from typing import Callable, Iterator
from abc import abstractmethod, ABCMeta
from pathlib import Path
from datetime import datetime
import tomllib

from ..types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession, default_mincontract
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader


class Provider(metaclass=ABCMeta):
    """
    Base class for all providers
    """

    timezone = 'UTC'
    """ Timezone of the provider """

    symbol: str | None = None
    """ Symbol of the provider """

    timeframe: str | None = None
    """ Timeframe of the provider """

    xchg_timeframe: str | None = None
    """ TradingView timeframe """

    ohlcv_path: Path | None = None
    """ Directory to save OHLV data """

    mincontract_estimated: bool = False
    """True when the last :meth:`get_symbol_info` fetch had to estimate
    ``mincontract`` because the provider returned no exchange value. The
    download flow then refines the estimate from the downloaded volume data."""

    config_keys = {
        '# Settings for the provider': '',
    }
    """ Key-value pairs to put into providers.toml, if key starts with '#' it is a comment. """

    config: dict[str, str] = {}
    """ Config dict for the exchange loaded from providers.toml """

    @classmethod
    @abstractmethod
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to TradingView fmt
        https://www.tradingview.com/pine-script-reference/v6/#var_timeframe.period
        """

    @classmethod
    @abstractmethod
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """
        Convert timeframe to exchange fmt
        """

    @classmethod
    def get_ohlcv_path(cls, symbol: str, timeframe: str, ohlv_dir: Path, provider_name: str | None = None) -> Path:
        """
        Get the output path of the OHLV data
        """
        return ohlv_dir / (f"{provider_name or cls.__name__.lower().replace('provider', '')}"
                           f"_{symbol.replace('/', '_').replace(':', '_').upper()}"
                           f"_{timeframe}.ohlcv")

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlv_dir: Path | None = None, config_dir: Path | None = None):
        """
        :param symbol: The symbol to get data for
        :param timeframe: The timeframe to get data for in TradingView fmt
        :param ohlv_dir: The directory to save OHLV data
        :param config_dir: The directory to read the config file from
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.xchg_timeframe = self.to_exchange_timeframe(timeframe) if timeframe else None
        self.ohlcv_path = self.get_ohlcv_path(symbol, timeframe, ohlv_dir) if ohlv_dir else None
        self.ohlcv_file = OHLCVWriter(self.ohlcv_path) if self.ohlcv_path else None

        if not config_dir:  # Default config dir from the parent of the ohlcv_dir
            assert self.ohlcv_path is not None
            config_dir = self.ohlcv_path.parent.parent / 'config'
        self.config_dir = config_dir

        self.load_config()

    def __enter__(self) -> OHLCVWriter:
        assert self.ohlcv_file is not None
        return self.ohlcv_file.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.ohlcv_file is not None
        self.ohlcv_file.close()

    @abstractmethod
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        """
        Get list of symbols
        """

    def load_config(self):
        """
        Load config from providers.toml
        """
        with open(self.config_dir / 'providers.toml', 'rb') as f:
            data = tomllib.load(f)
            self.config = data[self.__class__.__name__.replace('Provider', '').lower()]

    @abstractmethod
    def update_symbol_info(self) -> SymInfo:
        """
        Update symbol info from the exchange
        """

    def is_symbol_info_exists(self) -> bool:
        """
        Check if symbol info file exists
        """
        assert self.ohlcv_path is not None
        return self.ohlcv_path.with_suffix('.toml').exists()

    def get_symbol_info(self, force_update=False) -> SymInfo:
        """
        Get market details of a symbol

        :param force_update: Force update the symbol info
        """
        assert self.ohlcv_path is not None
        toml_path = self.ohlcv_path.with_suffix('.toml')
        # Check if file already exists
        if self.is_symbol_info_exists() and not force_update:
            return SymInfo.load_toml(toml_path)

        sym_info = self.update_symbol_info()
        if sym_info.mincontract <= 0.0:
            # No exchange value (providers signal that with 0.0): estimate it.
            # The download flow refines the estimate from the downloaded
            # volume data, see ``mincontract_estimated``.
            sym_info.mincontract = default_mincontract(sym_info.type, sym_info.basecurrency)
            self.mincontract_estimated = True
        sym_info.save_toml(toml_path)
        return sym_info

    @abstractmethod
    def get_opening_hours_and_sessions(self) \
            -> tuple[list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """
        Get opening hours and sessions of a symbol
        """

    def save_ohlcv_data(self, data: OHLCV | list[OHLCV]):
        """
        Save OHLV data to a file

        :param data: OHLV data
        """
        assert self.ohlcv_file is not None
        if isinstance(data, OHLCV):
            self.ohlcv_file.write(data)
        else:
            for candle in data:
                self.ohlcv_file.write(candle)

    @abstractmethod
    def download_ohlcv(self, time_from: datetime | None, time_to: datetime | None,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None):
        """
        Download OHLV data

        In the user code you can call `self.save_ohlcv_data()` to save the data into the data file

        :param time_from: The start time (None to fetch all available data)
        :param time_to: The end time (None to fetch up to the latest)
        :param on_progress: Optional callback to call on progress
        :param limit: Override the automatic chunk size (number of bars per API request)
        """

    def load_ohlcv_data(self) -> OHLCVReader:
        """
        Load OHLV data from the file
        """
        return OHLCVReader(str(self.ohlcv_path))

    def stream(
            self,
            symbol: str,
            timeframe: str,
            *,
            start: datetime | None = None,
            end: datetime | None = None,
            include_gaps: bool = False,
    ) -> Iterator[OHLCV]:
        """Yield OHLCV bars in chronological order for ``(symbol, timeframe)``.

        Default implementation reads from the ``.ohlcv`` file populated by
        :meth:`download_ohlcv`. Subclasses MAY override for direct-query
        optimization (e.g. an FMP subclass hitting REST directly, no file
        intermediate; a SQLite subclass issuing one SELECT).

        Behavioral contract (Pine Extraction Design §5.1 / §5.4):

        - For closed historical ranges, ``list(stream(...))`` MUST equal
          ``fetch(...)``. For live ranges, ``stream()`` MAY include a
          forming bar that ``fetch()`` excludes.
        - ``start`` / ``end`` are inclusive when supplied; ``None`` means
          unbounded on that side. Both must be timezone-aware if provided
          — naive datetimes are ambiguous across timezones and raise
          ``TypeError``.
        - ``start > end`` returns no bars (SQL-consistent — spec §5.4
          conformance check #4, Rev 3). It does NOT raise.
        - Yielded ``OHLCV.timestamp`` values are Unix epoch seconds (int),
          which by convention represent UTC; the ordering is monotonically
          non-decreasing.
        - Gap-filled bars (volume == -1, written by :class:`OHLCVWriter`
          to preserve interval regularity) are skipped by default so
          downstream consumers see only real trading activity. Set
          ``include_gaps=True`` to preserve them (spec §5 "MAY" clause
          — matches :meth:`OHLCVReader.read_from`'s ``skip_gaps`` opt-in).
        - Missing underlying data (e.g. ``.ohlcv`` file was never populated
          by :meth:`download_ohlcv`) returns ``[]`` per spec §5.4 check #3
          — a raw ``FileNotFoundError`` would leak an internal file-format
          concern to the caller.

        The ``symbol`` and ``timeframe`` arguments are accepted at call
        time per spec §5 (call-parameterized, stateless across calls).
        The default file-backed impl still resolves the on-disk path via
        the instance's ``symbol``/``timeframe`` set at construction time,
        so a mode-1 mismatch (call-time values differ from construction-
        time) raises ``ValueError`` to prevent silent wrong-symbol data
        leakage — subclasses that override MAY honor the call-time values
        for true multi-symbol reuse of a single Provider instance
        (spec §5.2 mode 2).
        """
        # Guard: mode-1 (single-symbol) providers must fail loud on a
        # call-time symbol/timeframe mismatch, otherwise the file-backed
        # default silently returns the construction-time symbol's data
        # (verified via a differing-symbol fetch). Only check when the
        # instance has a concrete symbol/timeframe — a Provider built with
        # no symbol is treated as unrestricted (spec §5.2 mode-2 subclasses
        # override this method anyway).
        if self.symbol is not None and symbol != self.symbol:
            raise ValueError(
                f"call-time symbol {symbol!r} does not match construction-time "
                f"{self.symbol!r}; the default file-backed stream() is mode-1 "
                "(single-symbol per instance). Override stream() for multi-"
                "symbol reuse (spec §5.2 mode 2)."
            )
        if self.timeframe is not None and timeframe != self.timeframe:
            raise ValueError(
                f"call-time timeframe {timeframe!r} does not match construction-"
                f"time {self.timeframe!r}; the default file-backed stream() is "
                "mode-1 (single-timeframe per instance). Override stream() for "
                "multi-timeframe reuse (spec §5.2 mode 2)."
            )

        # Guard: naive datetimes are ambiguous (spec §5 requires timezone-
        # aware). ``.timestamp()`` on a naive datetime silently interprets
        # it in the local timezone, which is a footgun for cross-machine
        # reproducibility — reject explicitly with a typed error. This
        # check MUST run before the reversed-range guard below, because
        # comparing naive vs aware raises a generic TypeError that would
        # mask our purpose-specific message.
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

        # Guard: reversed range → empty output (spec §5.4 conformance check #4).
        # An early ``return`` inside a generator terminates it immediately,
        # yielding nothing to the caller.
        if start is not None and end is not None and start > end:
            return

        # Guard: missing .ohlcv file → empty output (spec §5.4 check #3).
        # ``OHLCVReader.open()`` calls ``open(path, 'rb')`` which raises
        # ``FileNotFoundError`` — a leaky internal file-format concern.
        # We convert it to the spec-compliant "unknown range returns []"
        # semantics. The distinction between "never downloaded" and
        # "downloaded but empty range" can still be recovered by the
        # caller via ``is_symbol_info_exists()``.
        try:
            reader_cm = self.load_ohlcv_data()
            reader_cm.open()
        except FileNotFoundError:
            return

        # Convert start/end datetimes → epoch seconds so we can delegate
        # to ``OHLCVReader.read_from``'s O(range) fast-path
        # (``get_positions`` computes byte offsets from the interval
        # header — no full-file scan). ``read_from`` treats a start of
        # ``0`` as "beginning of file" (falsy → get_positions branch),
        # so we pass 0 when start is None.
        start_ts = int(start.timestamp()) if start is not None else None
        end_ts = int(end.timestamp()) if end is not None else None

        try:
            for bar in reader_cm.read_from(
                start_timestamp=start_ts if start_ts is not None else 0,
                end_timestamp=end_ts,
                skip_gaps=not include_gaps,
            ):
                # ``get_positions`` clamps start_pos to ``size - 1`` when
                # start is past the last bar (see ohlcv_file.py:1691), and
                # end_pos to +1 bar of end_diff (see :1704). Both can yield
                # a bar strictly outside the requested [start, end] range.
                # Re-check the bar timestamp to enforce exact inclusive
                # bounds and stop early when we cross ``end`` (chronological).
                if start_ts is not None and bar.timestamp < start_ts:
                    continue
                if end_ts is not None and bar.timestamp > end_ts:
                    break
                yield bar
        finally:
            reader_cm.close()

    def fetch(
            self,
            symbol: str,
            timeframe: str,
            *,
            start: datetime | None = None,
            end: datetime | None = None,
            include_gaps: bool = False,
    ) -> list[OHLCV]:
        """Return a bar range as a materialized list.

        Default implementation: ``list(self.stream(...))``. Subclasses MAY
        override to issue a single batched query (e.g. one SQL SELECT)
        instead of iterating. Ordering + UTC guarantees are identical to
        :meth:`stream`, and ``include_gaps`` is forwarded unchanged.

        For closed historical ranges, ``fetch(...) == list(stream(...))``
        (spec §5.1). For live/forming ranges, ``fetch()`` MAY exclude a
        forming bar that ``stream()`` would yield.
        """
        return list(self.stream(
            symbol, timeframe, start=start, end=end, include_gaps=include_gaps,
        ))
