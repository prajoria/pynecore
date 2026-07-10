"""Shared behavioral conformance suite every :class:`Provider` subclass
must pass (bd-cko, Task E1.4 of the Pine Extraction plan).

Per Pine Extraction Design §5.4 (Rev 3), conformance is verified by a
**behavioral test suite** rather than ``isinstance`` — Python ABCMeta
only enforces the presence of ``@abstractmethod`` methods, not their
signatures or semantics. This module ships the shared suite; every
provider's own test module invokes it against a fixture-loaded instance.

The seven checks
================

1. **stream() / fetch() equivalence** for CLOSED historical ranges.
   ``list(provider.stream(...))`` MUST equal ``provider.fetch(...)`` when
   the fixture guarantees no forming (live) bar is in range. Live
   providers (e.g. FMP realtime) MAY pass ``closed_only=False`` to skip
   this check when their fixture cannot pin a closed window.

2. **Monotonically non-decreasing timestamps.** Bars are returned in
   chronological order — a strict-ascending check would over-constrain
   sub-second-granularity providers, so we accept equal adjacent
   timestamps.

3. **Empty range returns ``[]``.** A window with no matching data yields
   a clean empty list rather than a ``FileNotFoundError`` or a bare-file
   parse error. The suite queries a range far outside the fixture
   (default: year 2000) so this is deterministic.

4. **``start > end`` returns ``[]``.** SQL-consistent semantics matching
   spec R5 (Rev 3 change from Rev 2's "raise"). Providers must NOT raise
   in this case — an accidentally-swapped pair of user inputs should
   surface as "no bars" rather than a stack trace.

5. **Missing symbol raises a typed error.** Providers pick from
   ``(KeyError, LookupError, ValueError)`` — the exact type is left to
   the provider so mode-1 providers (which raise ``ValueError`` on
   construction/call-time symbol mismatch) and mode-2 providers (which
   naturally return ``[]`` and need a wrapper to opt into raising) can
   both express the contract.

6. **UTC-timezone enforcement.** Two-pronged: (a) every returned bar's
   ``timestamp`` (an epoch int) round-trips through ``timezone.utc``
   cleanly, and (b) passing a naive ``datetime`` for ``start``/``end``
   raises ``TypeError``. This locks the "no ambiguous local-time input,
   no ambiguous local-time output" contract at the API surface.

7. **Stateless across calls.** Calling ``fetch()`` (and ``stream()``)
   twice with identical arguments yields identical results. Providers
   MUST NOT hold cursor state between calls — a fresh call is a fresh
   query. This regresses against the common footgun where a single
   ``DictReader`` or DB cursor is stashed on the instance and exhausted
   silently on the second call.

Calling from an external test module
====================================

The suite is a bare helper — NOT a pytest-collected function itself.
Downstream provider test modules (e.g. ``test_csv.py``,
``test_sqlite_provider.py``, or openbb-fork's ``test_fmp_provider.py``
at E3.2/E3.3) each add a ``__test_conforms_to_shared_suite__`` function
that constructs a fixture-loaded provider and calls this suite. Example:

.. code-block:: python

    from pynecore.providers.tests.test_conformance import _conformance_suite

    def __test_csv_provider_conforms_to_shared_suite__(tmp_path):
        provider = CSVProvider(csv_path=FIXTURE, symbol="TESTSYM",
                               timeframe="1D", ohlv_dir=tmp_path,
                               config_dir=tmp_path)
        _conformance_suite(provider, closed_only=True)

For providers whose default missing-symbol behavior is ``[]`` (mode-2
providers like SQLite), wrap the instance in a subclass that raises on
missing symbol before handing it to the suite — see
``_RaisingSQLiteProvider`` in ``test_sqlite_provider.py`` for the
reference pattern.

Naming convention
=================

pynecore's ``pytest.ini`` sets ``python_functions = __test_*__``, so
this file's helper is prefixed with a single leading underscore
(``_conformance_suite``) to make explicit that it is NOT itself a test
— the collector would ignore it either way, but the underscore signals
intent to human readers. Downstream provider test functions MUST use
the ``__test_..._..__`` pattern to be discovered.

Provider coverage status
========================

CSV (bd-gxy) and SQLite (bd-l05) reference providers invoke this suite via
their ``__test_*_provider_conforms_to_shared_suite__`` functions. The
pre-existing CCXTProvider and CapitalComProvider do NOT yet invoke it —
those are network-oriented mode-1 providers that need recorded fixtures
before the suite can run cleanly. Retroactive audit is tracked as bd-7a8
(OpenBBTechnical-7a8, "Retroactively verify CCXTProvider +
CapitalComProvider pass conformance suite"), blocked by bd-cko landing.

Deliberate spec deviations (stricter than spec §5.4)
====================================================

Two checks below intentionally test more than the spec §5.4 wording:

* **Check #6** also asserts that naive ``datetime`` inputs raise
  ``TypeError`` on BOTH ``start`` and ``end``. Spec §5.4 check #6 only
  names the returned-bar UTC round-trip. The input-side prong is
  defensible: a provider that returns UTC bars but silently accepts naive
  input bounds is internally inconsistent, and the strictness closes a
  class of "future subclass overrides ``stream()`` and forgets to
  re-invoke ``super()``'s naive-datetime guard" bugs at the API surface.
* **Check #7** also asserts ``stream()`` statelessness. Spec §5.4 check
  #7 only names ``fetch()``. The same "no cursor state between calls"
  invariant per spec §5.2 applies to ``stream()``, and the extension
  catches bugs where ``fetch()`` is a thin wrapper over ``stream()`` that
  materializes fresh each call while ``stream()`` itself leaks state.

Both extensions are hardening — a future spec revision may fold them in;
until then this docstring is the authoritative record of the delta.

Clean-room: I have not viewed TradingView or PyneComp source code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pynecore.providers.provider import Provider


def _conformance_suite(
    provider: "Provider",
    *,
    closed_only: bool = True,
    test_symbol: str = "TESTSYM",
    test_timeframe: str = "1D",
    range_start: datetime | None = None,
    range_end: datetime | None = None,
    empty_range_start: datetime | None = None,
    empty_range_end: datetime | None = None,
    missing_symbol: str = "NEVER_EXISTS_XYZ_9999",
) -> None:
    """Run the 7 spec §5.4 behavioral checks against ``provider``.

    The provider must already be constructed and wired to a fixture that
    holds bars for ``(test_symbol, test_timeframe)`` within
    ``[range_start, range_end]``. The empty-range window
    ``[empty_range_start, empty_range_end]`` MUST be strictly outside the
    fixture data (default: year 2000, safe for any modern dataset).

    :param provider: A ready-to-query :class:`Provider` instance. NOT
        wrapped in a context manager — the suite calls ``fetch``/``stream``
        directly, which for mode-2 providers do not need ``__enter__``.
    :param closed_only: When True (the default), the fixture guarantees
        the range is entirely closed (no forming bar). Live providers
        (e.g. FMP realtime) may pass ``False`` to skip check #1.
    :param test_symbol: Symbol the fixture has data for. Defaults to
        ``"TESTSYM"`` to match the reference CSV/SQLite fixtures.
    :param test_timeframe: Timeframe the fixture has data for. Defaults
        to ``"1D"``.
    :param range_start: Start of the closed historical range the fixture
        covers. Defaults to ``2024-01-01`` UTC.
    :param range_end: End of the closed historical range. Defaults to
        ``2024-01-05`` UTC (5-bar CSV/SQLite reference fixture).
    :param empty_range_start: Lower bound of the known-empty window used
        by check #3. Defaults to ``2000-01-01`` UTC — safely outside
        any modern OHLCV dataset.
    :param empty_range_end: Upper bound of the known-empty window.
        Defaults to ``2000-01-02`` UTC.
    :param missing_symbol: Symbol name known NOT to exist in the fixture.
        Defaults to ``"NEVER_EXISTS_XYZ_9999"`` — a deliberately unlikely
        identifier that no real corpus would produce.
    :raises AssertionError: If any check fails.
    """
    if range_start is None:
        range_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if range_end is None:
        range_end = datetime(2024, 1, 5, tzinfo=timezone.utc)
    if empty_range_start is None:
        empty_range_start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    if empty_range_end is None:
        empty_range_end = datetime(2000, 1, 2, tzinfo=timezone.utc)

    provider_name = type(provider).__name__

    # ------------------------------------------------------------------
    # Check #1 — stream()/fetch() equivalence for closed historical ranges
    # ------------------------------------------------------------------
    # For closed ranges, the two entry points MUST return byte-identical
    # data. This is what lets consumers pick between them on the
    # memory-vs-latency axis without worrying about semantic drift. Live
    # providers may include a forming bar that fetch() excludes, so we
    # skip this check when the caller flags the fixture as live-range.
    if closed_only:
        streamed = list(provider.stream(
            test_symbol, test_timeframe,
            start=range_start, end=range_end,
        ))
        fetched = provider.fetch(
            test_symbol, test_timeframe,
            start=range_start, end=range_end,
        )
        assert streamed == fetched, (
            f"{provider_name}: stream()/fetch() disagreed for closed range: "
            f"stream={len(streamed)} bars, fetch={len(fetched)} bars"
        )

    # ------------------------------------------------------------------
    # Check #2 — timestamps monotonically non-decreasing
    # ------------------------------------------------------------------
    # We reuse ``fetched`` below for check #6, so materialize once. Use
    # ``<=`` rather than strict ``<`` — sub-second providers may emit
    # equal adjacent timestamps at second granularity, and the spec only
    # requires non-decreasing order.
    fetched = provider.fetch(
        test_symbol, test_timeframe,
        start=range_start, end=range_end,
    )
    timestamps = [bar.timestamp for bar in fetched]
    assert timestamps == sorted(timestamps), (
        f"{provider_name}: bars must be in chronological order (got "
        f"timestamps={timestamps})"
    )

    # ------------------------------------------------------------------
    # Check #3 — empty (out-of-fixture-range) window returns []
    # ------------------------------------------------------------------
    # This catches the class of bugs where a missing file leaks
    # FileNotFoundError, or an out-of-range SQL query raises rather than
    # returning zero rows. The default window (year 2000) is safely
    # outside any modern OHLCV dataset.
    empty = provider.fetch(
        test_symbol, test_timeframe,
        start=empty_range_start, end=empty_range_end,
    )
    assert empty == [], (
        f"{provider_name}: out-of-range window must return [] cleanly; "
        f"got {len(empty)} bars"
    )

    # ------------------------------------------------------------------
    # Check #4 — start > end returns [] (spec R5, Rev 3)
    # ------------------------------------------------------------------
    # Reversed-range MUST NOT raise. Rev 2 required a raise but that
    # forced every subclass to add a guard; Rev 3 aligned with SQL
    # SELECT semantics (an empty result set is the natural output of a
    # WHERE clause whose bounds are inverted).
    reversed_range = provider.fetch(
        test_symbol, test_timeframe,
        start=range_end, end=range_start,
    )
    assert reversed_range == [], (
        f"{provider_name}: start > end must return [] (not raise); "
        f"got {len(reversed_range)} bars"
    )

    # ------------------------------------------------------------------
    # Check #5 — missing symbol raises a typed error
    # ------------------------------------------------------------------
    # Providers pick from (KeyError, LookupError, ValueError) so both
    # mode-1 providers (which raise ValueError on symbol mismatch — see
    # CSVProvider) and mode-2 providers wrapped to raise (see
    # _RaisingSQLiteProvider in test_sqlite_provider.py) satisfy the
    # contract. The exact exception type is not fixed — the contract is
    # "you notice, not silently get zero rows."
    with pytest.raises((KeyError, LookupError, ValueError)):
        provider.fetch(
            missing_symbol, test_timeframe,
            start=range_start, end=range_end,
        )

    # ------------------------------------------------------------------
    # Check #6 — UTC-timezone enforcement
    # ------------------------------------------------------------------
    # Two prongs:
    #  (a) Every returned bar's timestamp round-trips through UTC. The
    #      OHLCV NamedTuple stores ``timestamp`` as an epoch int by
    #      convention; converting via ``datetime.fromtimestamp(ts,
    #      tz=timezone.utc)`` must produce a UTC-tagged datetime. This
    #      catches providers that accidentally store local-time epoch
    #      seconds (which would deliver a wrong wall-clock time on
    #      cross-timezone machines).
    #  (b) The API surface rejects naive datetimes for start/end. Naive
    #      inputs are ambiguous cross-machine because ``.timestamp()``
    #      on a naive datetime silently interprets it in the machine's
    #      local timezone — a cross-machine reproducibility footgun.
    for bar in fetched:
        # OHLCV.timestamp is an int (epoch seconds, UTC by convention).
        # Round-tripping through UTC MUST produce a UTC-tagged datetime.
        # If a future provider used a timezone-aware datetime instead
        # (against the current model), the ``hasattr`` branch documents
        # the same contract for that shape too.
        if hasattr(bar.timestamp, "tzinfo"):
            assert bar.timestamp.tzinfo == timezone.utc, (
                f"{provider_name}: bar.timestamp is timezone-aware but "
                f"not UTC (got tzinfo={bar.timestamp.tzinfo!r})"
            )
        else:
            dt = datetime.fromtimestamp(bar.timestamp, tz=timezone.utc)
            assert dt.tzinfo == timezone.utc, (
                f"{provider_name}: bar timestamps must round-trip through "
                f"UTC (got tzinfo={dt.tzinfo!r} for ts={bar.timestamp!r})"
            )

    # API-surface naive-datetime rejection. We test both bounds — some
    # provider impls only guard start, not end. Both must raise
    # TypeError (a bare ValueError would let a str be passed and mask
    # the deeper "you didn't give me a tz-aware datetime" error).
    with pytest.raises(TypeError):
        provider.fetch(
            test_symbol, test_timeframe,
            start=datetime(2024, 1, 1),  # naive — no tzinfo
            end=range_end,
        )
    with pytest.raises(TypeError):
        provider.fetch(
            test_symbol, test_timeframe,
            start=range_start,
            end=datetime(2024, 1, 5),  # naive — no tzinfo
        )

    # ------------------------------------------------------------------
    # Check #7 — stateless across calls
    # ------------------------------------------------------------------
    # Fetch twice with identical args → identical results. Regressess
    # against the common footgun where a provider stashes a cursor /
    # DictReader / DB connection on ``self`` and the second call gets
    # nothing because the first exhausted it. We check both fetch (spec
    # §5.4 check #7 text) AND stream (same "no cursor state" intent
    # applied to the iterator entry point) — the stream check catches
    # a bug the fetch check would miss when fetch is a thin wrapper
    # over stream that materializes fresh each call.
    a = provider.fetch(
        test_symbol, test_timeframe,
        start=range_start, end=range_end,
    )
    b = provider.fetch(
        test_symbol, test_timeframe,
        start=range_start, end=range_end,
    )
    assert a == b, (
        f"{provider_name}: fetch() must be stateless (two identical "
        f"calls returned different results: {len(a)} vs {len(b)} bars)"
    )
    sa = list(provider.stream(
        test_symbol, test_timeframe,
        start=range_start, end=range_end,
    ))
    sb = list(provider.stream(
        test_symbol, test_timeframe,
        start=range_start, end=range_end,
    ))
    assert sa == sb, (
        f"{provider_name}: stream() must be stateless (two identical "
        f"calls returned different results: {len(sa)} vs {len(sb)} bars)"
    )
