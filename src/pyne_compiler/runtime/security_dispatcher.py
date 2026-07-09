"""D5 §4.2 runtime dispatcher — prefetches every ``request.security``
secondary series before ``ScriptRunner.run_iter()`` starts.

Fetching lazily on every bar would blow the FMP retry budget on the first
long-running strategy — a script with 3 secondaries × 5000 bars = 15 000
extra fetches, each subject to 5 retries. Prefetching once, aligning to
the primary bar grid via forward-fill, and threading a
``{context_id: DataFrame}`` map to the executor collapses that to 3
fetches total.

Priority order (D5 §4.2, verbatim):

    1. If ``data_resolver`` supplied (Python API only), call it per context.
    2. Else if the context is dynamic (symbol or timeframe resolved at
       runtime), defer to lazy per-bar fetch — log a warning and return an
       empty DataFrame from this pass so the executor's fallback path takes
       over. Documented 5-10× perf caveat (D5 §4.4).
    3. Else fetch via ``fmp_provider`` wrapped in the shared retry budget.
    4. Post-fetch: forward-fill align to the primary series' timestamps so
       PyneCore's per-bar reads never see NaN in the middle of a series.

Cache path uses :class:`SecondarySeriesCache` — key
``(symbol, timeframe, start_bar, end_bar)``. Hit rate is expected high
because real strategies reference the same 2-3 secondaries repeatedly
across backtest re-runs.

The executor threads the returned map into PyneCore's ``request.security``
substrate via a module-level global — that monkey-patch
(``_install_secondaries_hook``) is a separate bead (n6j) and is
DELIBERATELY not touched here.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from openbb_pine.compiler_errors import PineDataResolverError
from openbb_pine.runtime.fmp_provider import (
    FMPOHLCVProvider,
    FMPRequest,
    infer_asset_class,
)
from openbb_pine.runtime.fmp_retry import call_with_retry
from openbb_pine.runtime.secondary_cache import SecondarySeriesCache

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from openbb_pine.compiler.types import SecurityContext


__all__ = [
    "prefetch_security_contexts",
    "align_to_primary",
]


_log = logging.getLogger(__name__)


# --- Alignment helper ---------------------------------------------------------


def align_to_primary(
    secondary: pd.DataFrame, primary_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Forward-fill ``secondary`` onto ``primary_index``.

    Per D5 §4.2 step 4. Secondaries that resolve to a different timeframe
    than the primary (e.g. daily secondary on a 5-minute primary chart) get
    each higher-timeframe bar broadcast across the primary bars that fall
    within it, until the next higher-timeframe bar arrives — which is
    exactly Pine's ``request.security`` semantics with ``gaps=barmerge.
    gaps_off`` (the default).

    Empty ``secondary`` → returns an empty DataFrame with the primary's
    columns preserved (which is nothing, since there ARE no columns).
    """
    if secondary.empty:
        # Preserve schema shape without materialising empty rows against the
        # primary index — the executor's fallback path decides whether to
        # zero-fill or NaN-fill from here.
        return secondary.copy()
    # ``reindex(method="ffill")`` handles both alignment and forward-fill in
    # one pass. Pandas will insert NaN for primary timestamps that fall
    # BEFORE the first secondary bar (no earlier value to carry forward);
    # that matches Pine's behaviour (``na`` until the first secondary bar).
    return secondary.reindex(primary_index, method="ffill")


# --- Static-vs-dynamic marker helpers ----------------------------------------


def _context_is_dynamic(ctx: "SecurityContext") -> bool:
    """Return True if the context defers to per-bar lazy fetch.

    D5 §4.1 defines ``dynamic_symbol`` and ``dynamic_timeframe`` markers on
    ``SecurityContext``, but the Wave-1 ``SecurityContext`` dataclass
    (owned by bead ``d75``) doesn't have them yet. Until it does, this
    helper always returns False (treat every context as static and route to
    FMP). The ``getattr`` fallback means once bead ``d75`` adds the fields,
    this file starts honouring them without a change.

    TODO(d75): remove the ``getattr`` fallback once ``SecurityContext`` has
    ``dynamic_symbol`` / ``dynamic_timeframe`` fields. Follow-up bead
    ``n6j`` will also want to wire the per-bar lazy fetch path referenced
    in D5 §4.4 — this dispatcher today just logs + returns empty for
    dynamic contexts, deferring the fetch responsibility to the executor.
    """
    return bool(
        getattr(ctx, "dynamic_symbol", False)
        or getattr(ctx, "dynamic_timeframe", False)
    )


# --- Bar-window inference for cache key --------------------------------------


def _extract_window(
    primary: FMPRequest | pd.DataFrame,
) -> tuple[Any, Any]:
    """Return ``(start_bar, end_bar)`` for the cache key.

    * ``pd.DataFrame`` primary → use the first / last index values.
    * ``FMPRequest`` primary → use ``request.start`` / ``request.end``.

    Missing endpoints (empty frame, ``None`` start/end) fall back to a
    sentinel string ``"?"`` so the cache key stays deterministic.
    """
    if isinstance(primary, pd.DataFrame):
        if primary.empty:
            return ("?", "?")
        return (primary.index[0], primary.index[-1])
    # FMPRequest branch — start/end may be None (OpenBB's default window).
    start = primary.start if primary.start is not None else "?"
    end = primary.end if primary.end is not None else "?"
    return (start, end)


# --- Per-context fetch paths -------------------------------------------------


def _fetch_via_resolver(
    context_id: str,
    ctx: "SecurityContext",
    data_resolver: Callable[[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Call the user-supplied ``data_resolver(symbol, timeframe)`` and wrap
    any raised exception in :class:`PineDataResolverError` per D5 §4.3.

    The resolver receives ``symbol`` and ``timeframe`` as plain strings —
    the same shape Pine's ``request.security(symbol, timeframe, expr)``
    accepts. It MUST return a ``pd.DataFrame`` (verified by the caller so
    the error message can name the offending context).
    """
    try:
        return data_resolver(ctx.symbol, ctx.timeframe)
    except Exception as exc:  # noqa: BLE001 -- intentional broad catch to wrap
        raise PineDataResolverError(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            context_id=context_id,
        ) from exc


def _fetch_via_fmp(
    context_id: str,
    ctx: "SecurityContext",
    fmp_provider_template: FMPOHLCVProvider,
) -> pd.DataFrame:
    """Fetch the secondary via ``FMPOHLCVProvider``, reusing the primary
    provider's ``provider_used`` (``"fmp"`` / ``"fmp_cached"``) and
    ``user_settings`` so the secondary honours the same routing decision.

    Wrapped in :func:`call_with_retry` so a rate-limited or transient FMP
    failure on ONE secondary consumes the shared retry budget and surfaces
    as ``PineFMPUnreachableError`` — never a bare ``httpx`` / ``OpenBBError``.
    """
    provider_name = fmp_provider_template.provider

    def _do_fetch() -> pd.DataFrame:
        # Build a fresh FMPRequest for this secondary — the primary
        # provider's request is for a different symbol/tf.
        req = FMPRequest(
            symbol=ctx.symbol,
            interval=ctx.timeframe,
            start=None,
            end=None,
            asset_class=infer_asset_class(ctx.symbol),
        )
        provider = FMPOHLCVProvider(req, provider=provider_name)
        # Access the DataFrame directly (before OHLCV coercion) — the
        # dispatcher works in DataFrame-space so alignment / forward-fill
        # are trivial. ``_fetch`` is deliberately private on
        # FMPOHLCVProvider today; if it goes public later this call site
        # stays the same.
        return provider._fetch()  # noqa: SLF001 -- deliberate cross-module reach

    label = f"secondary:{ctx.symbol}:{ctx.timeframe}[{context_id}]"
    return call_with_retry(_do_fetch, label=label, provider=provider_name)


# --- Main entry point --------------------------------------------------------


def prefetch_security_contexts(
    contexts: dict[str, "SecurityContext"],
    primary: FMPRequest | pd.DataFrame,
    *,
    fmp_provider: FMPOHLCVProvider | None,
    data_resolver: Callable[[str, str], pd.DataFrame] | None = None,
    cache: SecondarySeriesCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch every secondary series before ``ScriptRunner.run_iter()`` starts.

    Returns ``{context_id: DataFrame}`` — the executor threads this into
    PyneCore's ``request.security`` runtime via the (n6j-owned)
    ``_install_secondaries_hook`` monkey-patch that reads from a
    module-level global on the emitted ``@pyne`` module.

    Parameters
    ----------
    contexts
        ``compiled.security_contexts`` — the map C3 populated. If ``None``
        or empty this function short-circuits to ``{}`` (no fetches, no
        cache lookups).
    primary
        The primary bar grid. Either an ``FMPRequest`` (executor's FMP
        path) or a ``pd.DataFrame`` (executor's BYO path). See the
        NotImplementedError path below for the Wave-1 caveat.
    fmp_provider
        The already-constructed primary FMPOHLCVProvider. We reuse its
        ``provider`` (``"fmp"`` / ``"fmp_cached"``) so all secondaries
        route the same way. ``None`` is legal only when every context has
        a ``data_resolver`` fallback available.
    data_resolver
        Optional per-context BYO resolver (Python API only; not exposed by
        REST — D5 §4.3). When supplied it wins over the FMP path, per D5
        priority order.
    cache
        Optional :class:`SecondarySeriesCache`. When provided, hits skip
        both the resolver and the FMP fetch. Misses populate the cache
        after the fetch completes. ``None`` disables caching.

    Priority per D5 §4.2
    --------------------
    1. Cache lookup (if ``cache`` provided).
    2. ``data_resolver`` (if supplied).
    3. Dynamic-symbol / dynamic-timeframe → empty DataFrame + warning
       (D5 §4.4 defers per-bar lazy fetch to a follow-up bead).
    4. ``fmp_provider`` with the shared retry budget.
    5. Forward-fill align to primary index.

    Raises
    ------
    PineFMPUnreachableError
        From :func:`call_with_retry` when the FMP path exhausts its budget.
    PineDataResolverError
        When a user-supplied ``data_resolver`` raises.
    NotImplementedError
        When ``primary`` is an ``FMPRequest`` (see below).

    Wave-1 caveat: ``FMPRequest`` primary
    -------------------------------------
    The Wave-1 executor call site passes a ``pd.DataFrame`` primary (it
    fetches the primary before calling us). Handling ``FMPRequest``
    directly would require this function to fetch the primary itself —
    duplicating the executor's provider construction path. We raise
    ``NotImplementedError`` for now with a clear pointer; the follow-up
    bead that touches ``executor.py`` (``n6j``) can either (a) pass a
    DataFrame after fetching or (b) extend this function's primary-fetch
    path. Marked as a TODO in-body.
    """
    # Fast path: empty / None contexts → no work, no cache calls, no
    # provider validation. Matches D5 spec ("only prefetch when there are
    # contexts to prefetch").
    if not contexts:
        return {}

    # TODO(n6j / follow-up executor bead): support ``FMPRequest`` primary.
    # For Wave 1, only DataFrame primary is required — the executor fetches
    # the primary before invoking us, so we can lean on ``primary.index``
    # for alignment.
    if not isinstance(primary, pd.DataFrame):
        raise NotImplementedError(
            "prefetch_security_contexts currently supports only a "
            "pd.DataFrame `primary` (executor pre-fetches). Follow-up "
            "bead (executor wiring) will extend to FMPRequest primaries."
        )

    primary_index: pd.DatetimeIndex = primary.index  # type: ignore[assignment]
    start_bar, end_bar = _extract_window(primary)

    result: dict[str, pd.DataFrame] = {}
    for context_id, ctx in contexts.items():
        # 1) Cache lookup — hits skip everything else.
        if cache is not None:
            cached = cache.get(ctx.symbol, ctx.timeframe, start_bar, end_bar)
            if cached is not None:
                result[context_id] = cached
                continue

        # 2) User-supplied resolver wins (Python API only).
        if data_resolver is not None:
            df = _fetch_via_resolver(context_id, ctx, data_resolver)
        # 3) Dynamic contexts defer to the executor's per-bar path.
        elif _context_is_dynamic(ctx):
            _log.warning(
                "prefetch_security_contexts: context %r has dynamic "
                "symbol/timeframe (symbol=%r tf=%r); deferring to per-bar "
                "lazy fetch — D5 §4.4 documents the 5-10× perf caveat.",
                context_id, ctx.symbol, ctx.timeframe,
            )
            # Return an empty DataFrame so the executor can distinguish
            # "no data yet" from "no context" (the key is still present).
            result[context_id] = pd.DataFrame()
            continue
        # 4) Default: FMP route with shared retry budget.
        else:
            if fmp_provider is None:
                # Can't route to FMP and no resolver — the caller wired
                # inconsistent options. Surface a clean error instead of an
                # ``AttributeError`` later.
                raise ValueError(
                    f"prefetch_security_contexts: context {context_id!r} "
                    f"({ctx.symbol!r} @ {ctx.timeframe!r}) requires FMP "
                    "but no `fmp_provider` was supplied and no "
                    "`data_resolver` fallback is available."
                )
            df = _fetch_via_fmp(context_id, ctx, fmp_provider)

        # 5) Alignment (only for freshly-fetched frames; cached entries
        # were aligned at write time).
        aligned = align_to_primary(df, primary_index)
        result[context_id] = aligned

        # Populate cache post-alignment so subsequent hits skip the
        # forward-fill work too.
        if cache is not None:
            cache.put(ctx.symbol, ctx.timeframe, start_bar, end_bar, aligned)

    return result


# --- Utility datetime coercion (mirrors fmp_provider._ts_to_utc_seconds
#     but the dispatcher operates on frames, not raw ts). Not currently
#     needed — kept as a private hook for the follow-up FMPRequest path.
def _coerce_datetime(value: Any) -> datetime | None:  # pragma: no cover
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return None
