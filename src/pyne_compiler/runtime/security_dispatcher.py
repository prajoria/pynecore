"""D5 §4.2 runtime dispatcher — prefetches every ``request.security``
secondary series before ``ScriptRunner.run_iter()`` starts.

Fetching lazily on every bar would blow the shared retry budget on the
first long-running strategy — a script with 3 secondaries × 5000 bars =
15 000 extra fetches, each subject to 5 retries. Prefetching once,
aligning to the primary bar grid via forward-fill, and threading a
``{context_id: DataFrame}`` map to the executor collapses that to 3
fetches total.

Priority order (D5 §4.2, verbatim):

    1. If ``data_resolver`` supplied (Python API only), call it per context.
    2. Else if the context is dynamic (symbol or timeframe resolved at
       runtime), defer to lazy per-bar fetch — log a warning and return an
       empty DataFrame from this pass so the executor's fallback path takes
       over. Documented 5-10× perf caveat (D5 §4.4).
    3. Else fetch via ``provider.fetch(...)`` — the caller passes in a
       concrete :class:`_DataProviderStub` (post-E1: real
       ``pynecore.providers.Provider``). Concrete-provider concerns
       (retry envelope, request builder, provider-name reuse) live in the
       caller — see :mod:`openbb_pine.runtime.executor` (post-E0.3:
       :mod:`~executor_shell`).
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

E0.2 abstraction boundary
-------------------------
The dispatcher no longer imports concrete-provider types or their retry
helpers. Callers wrap their concrete data provider (whatever it is) in a
:class:`_DataProviderStub`-conforming object and pass it via ``provider=``.
Any retry / envelope semantics (e.g. transient-error handling for a
network-backed provider) is now the caller's responsibility to install
around the stub's ``fetch``. See the E0.2 gate test in
``test_prefetch_security.py`` for the enforced banned-import list.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from openbb_pine.compiler_errors import PineDataResolverError
from openbb_pine.runtime.secondary_cache import SecondarySeriesCache

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from openbb_pine.compiler.types import SecurityContext
    from openbb_pine.runtime._data_provider_stub import _DataProviderStub


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
    the provider). The ``getattr`` fallback means once bead ``d75`` adds
    the fields, this file starts honouring them without a change.

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


def _extract_window(primary: pd.DataFrame) -> tuple[Any, Any]:
    """Return ``(start_bar, end_bar)`` for the cache key.

    An empty frame falls back to a sentinel string ``"?"`` so the cache
    key stays deterministic.

    Note: pre-E0.2 this helper also accepted a request-object primary
    (from the pre-decoupling concrete-provider path). That branch is gone
    — the dispatcher only supports DataFrame primaries now (see the
    ``NotImplementedError`` raise for non-DataFrame primary).
    """
    if primary.empty:
        return ("?", "?")
    return (primary.index[0], primary.index[-1])


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


def _fetch_via_provider(
    context_id: str,
    ctx: "SecurityContext",
    provider: "_DataProviderStub",
    *,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    """Fetch the secondary via the caller-supplied
    :class:`_DataProviderStub` (post-E1: real
    ``pynecore.providers.Provider``).

    The dispatcher no longer knows or cares whether the underlying
    concrete provider is network-backed, a BYO adapter, or something
    else — that's the caller's problem. The caller is also responsible
    for wrapping the ``fetch`` call in any retry envelope, so the
    dispatcher never surfaces provider-specific transport errors
    directly from this path — it propagates whatever the
    caller-installed wrapper raises.

    ``context_id`` and ``ctx`` are accepted for logging / future
    telemetry hooks, but are not passed to ``provider.fetch`` (whose
    signature is deliberately kept to the abstract Provider shape).
    """
    del context_id  # accepted for symmetry with _fetch_via_resolver
    return provider.fetch(ctx.symbol, ctx.timeframe, start=start, end=end)


# --- Main entry point --------------------------------------------------------


def prefetch_security_contexts(
    contexts: dict[str, "SecurityContext"],
    primary: pd.DataFrame,
    *,
    provider: "_DataProviderStub | None",
    data_resolver: Callable[[str, str], pd.DataFrame] | None = None,
    cache: SecondarySeriesCache | None = None,
    primary_start: datetime | None = None,
    primary_end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch every secondary series before ``ScriptRunner.run_iter()`` starts.

    Returns ``{context_id: DataFrame}`` — the executor threads this into
    PyneCore's ``request.security`` runtime via the (n6j-owned)
    ``_install_secondaries_hook`` monkey-patch that reads from a
    module-level global on the emitted ``@pyne`` module.

    Parameters
    ----------
    contexts
        ``compiled.security_contexts`` — the map C3 populated. If empty
        this function short-circuits to ``{}`` (no fetches, no cache
        lookups).
    primary
        The primary bar grid as a ``pd.DataFrame`` — the executor
        pre-fetches so we can lean on ``primary.index`` for alignment.
        Passing anything else raises ``NotImplementedError`` (Wave-1
        scope).
    provider
        The active :class:`_DataProviderStub` supplied by the caller.
        Post-E1 this becomes ``pynecore.providers.Provider``; for now
        the caller (executor_shell in E0.3) wraps its concrete data
        provider in a stub adapter. ``None`` is legal only when every
        context has a ``data_resolver`` fallback available.
    data_resolver
        Optional per-context BYO resolver (Python API only; not exposed
        by REST — D5 §4.3). When supplied it wins over the provider
        path, per D5 priority order.
    cache
        Optional :class:`SecondarySeriesCache`. When provided, hits skip
        both the resolver and the provider fetch. Misses populate the
        cache after the fetch completes. ``None`` disables caching.
    primary_start, primary_end
        Optional bar-window bounds forwarded to ``provider.fetch``. When
        omitted, the provider's fetch is invoked with ``start=None`` /
        ``end=None`` and the concrete provider decides its default
        window. Not used for cache-key derivation (that still comes from
        the primary DataFrame's index endpoints).

    Priority per D5 §4.2
    --------------------
    1. Cache lookup (if ``cache`` provided).
    2. ``data_resolver`` (if supplied).
    3. Dynamic-symbol / dynamic-timeframe → empty DataFrame + warning
       (D5 §4.4 defers per-bar lazy fetch to a follow-up bead).
    4. ``provider.fetch(...)`` — caller-installed provider stub.
    5. Forward-fill align to primary index.

    Raises
    ------
    PineDataResolverError
        When a user-supplied ``data_resolver`` raises.
    NotImplementedError
        When ``primary`` is not a DataFrame (Wave-1 scope marker).
    ValueError
        When a static context has neither a provider nor a resolver
        available.

    Any exception raised by ``provider.fetch(...)`` propagates
    unchanged — the caller owns retry / envelope semantics (e.g.
    wrapping a network-backed provider's ``fetch`` in a retry decorator
    that emits a typed transport error on budget exhaustion).
    """
    # Fast path: empty / None contexts → no work, no cache calls, no
    # provider validation. Matches D5 spec ("only prefetch when there are
    # contexts to prefetch").
    if not contexts:
        return {}

    # Wave-1 scope guard: a request-object primary (and any other
    # non-DataFrame) is not supported. The dispatcher requires the caller
    # to materialise the primary bar grid before calling us so we can
    # lean on ``primary.index`` for alignment.
    if not isinstance(primary, pd.DataFrame):
        raise NotImplementedError(
            "prefetch_security_contexts currently supports only a "
            "pd.DataFrame `primary` (executor pre-fetches). Non-DataFrame "
            "primaries (e.g. a request-builder object) must be "
            "materialised by the caller before invoking the dispatcher."
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
        # 4) Default: caller-installed provider stub.
        else:
            if provider is None:
                # Can't route to a provider and no resolver — the caller
                # wired inconsistent options. Surface a clean error
                # instead of an ``AttributeError`` later.
                raise ValueError(
                    f"prefetch_security_contexts: context {context_id!r} "
                    f"({ctx.symbol!r} @ {ctx.timeframe!r}) requires a "
                    "data provider but no `provider` was supplied and no "
                    "`data_resolver` fallback is available."
                )
            df = _fetch_via_provider(
                context_id,
                ctx,
                provider,
                start=primary_start,
                end=primary_end,
            )

        # 5) Alignment (only for freshly-fetched frames; cached entries
        # were aligned at write time).
        aligned = align_to_primary(df, primary_index)
        result[context_id] = aligned

        # Populate cache post-alignment so subsequent hits skip the
        # forward-fill work too.
        if cache is not None:
            cache.put(ctx.symbol, ctx.timeframe, start_bar, end_bar, aligned)

    return result


# --- Utility datetime coercion — kept as a private hook for the
#     follow-up request-object primary path (Wave-1 scope only supports
#     DataFrame primaries, per the NotImplementedError above).
def _coerce_datetime(value: Any) -> datetime | None:  # pragma: no cover
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return None
