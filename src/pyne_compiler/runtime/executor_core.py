"""Provider-agnostic core of the Pine runtime — E0.3 split (bd-9zb).

Extracted from ``executor.py`` per Pine Extraction Design §6.E0.3. This
module carries the runtime logic that MOVES to pynecore in E2 — the
compile-source materialisation, ``ScriptRunner`` wiring, per-yield
snapshot capture, alert collection, and DataFrame emission. It knows
NOTHING about specific data providers, retry envelopes, or the
attribution string. Those concerns live in the sibling
``executor_shell`` module (STAYS in openbb-fork).

Contract with the caller (executor_shell today; pynecore's own
harness post-E2):

* The caller instantiates a concrete data provider and passes it via
  the ``provider`` kwarg. The provider MUST expose ``iter_ohlcv()``
  yielding ``pynecore.types.ohlcv.OHLCV`` NamedTuples in
  time-ascending order (both openbb-fork concrete providers already
  do). It SHOULD also expose ``provider_used`` (str), ``bars_consumed``
  (int), ``symbol`` (str), ``interval`` (str | None), and
  ``asset_class`` (str | None) — executor_core reads these
  opportunistically via ``getattr`` so minimal test stubs still work.

* The caller owns any retry envelope around the provider's fetches
  (e.g. the openbb-fork wraps its network-backed provider's fetch
  method transparently). The core never handles retry classification.

* The caller wraps the returned ``(results_df, exec_ms, alerts)`` tuple
  in an envelope (``OBBject`` in openbb-fork) and attaches attribution
  metadata. The core deliberately returns raw pieces so a future
  pynecore-side ``execute`` can consume them without importing
  ``openbb_core``.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import tempfile
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from openbb_pine.compiler_errors import (
    PineExecTimeoutError,
    PineSecurityError,
)
from openbb_pine.runtime._pynecore_glue import (
    capture_alerts,
    ensure_pyne_header,
    make_default_syminfo,
    scan_for_forbidden_imports,
)
from openbb_pine.runtime.limits import DEFAULT_TIMEOUT_S, enforce_limits

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from openbb_pine.compiler.types import CompiledModule
    from pynecore.types.ohlcv import OHLCV


# --- Result emission ----------------------------------------------------------


def _collect_results(
    candles_and_plots: Iterable[tuple["OHLCV", dict[str, Any]]],
) -> pd.DataFrame:
    """Drain pre-copied ``(candle, plot_snapshot)`` pairs into a DataFrame.

    Per D2 §6.2, we deliberately concatenate at the end rather than
    streaming a Pydantic list, because the PRD §6 wall-clock budget
    (<=500 ms p50) is tighter than the per-yield-write cost.

    The caller MUST pass in **already-copied** plot dicts (see
    ``run_compiled``). PyneCore's ``run_iter()`` yields
    ``(candle, lib._plot_data)`` and then CLEARS ``lib._plot_data`` after
    the yield (``script_runner.py:811``). A post-hoc ``list(...)`` of the
    iterator captures the same (cleared) reference for every bar -- the
    snapshot must happen inside the loop body.
    """
    rows: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for candle, plot_data in candles_and_plots:
        timestamps.append(int(candle.timestamp))
        rows.append(plot_data if plot_data else {})

    # Build the DataFrame. Even if every plot dict is empty (a script with
    # no plot() calls), we still want a frame indexed by the bar timestamps
    # so downstream code can join to other surfaces. Empty plot dicts ->
    # zero columns; the DatetimeIndex carries the bar count.
    idx = pd.DatetimeIndex(
        [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps],
        name="date",
    )
    if not rows or all(not r for r in rows):
        return pd.DataFrame(index=idx)

    # Union of keys across all yielded plot dicts (a script may add a plot
    # mid-run via conditional logic, though that is unusual). DataFrame
    # construction handles missing keys -> NaN.
    return pd.DataFrame(rows, index=idx)


# --- Runtime entry point ------------------------------------------------------


def run_compiled(
    compiled: "CompiledModule",
    *,
    provider: Any,
    symbol: str | None = None,
    interval: str | None = None,
    start: datetime | None = None,  # noqa: ARG001 -- carried for shell parity
    end: datetime | None = None,  # noqa: ARG001 -- carried for shell parity
    params: dict[str, Any] | None = None,  # noqa: ARG001 -- C5 future hook
    timeout_s: int | None = None,
    asset_class: str | None = None,
) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    """Run a compiled ``@pyne`` module end-to-end using ``provider``.

    Returns the raw ``(results_df, exec_ms, alerts)`` tuple. The caller
    is responsible for wrapping this in an ``OBBject`` (or any other
    envelope) and attaching provider metadata / attribution. See the
    module docstring for the split contract.

    Parameters
    ----------
    compiled
        A ``CompiledModule`` produced by D1's codegen. ``source`` is
        materialised to a tempfile so PyneCore's ``@pyne`` AST hook can
        fire on import.
    provider
        Concrete data provider — see the module docstring for the
        expected surface (``iter_ohlcv()`` + a few optional attributes).
        The core does not validate the provider type — the shell is
        responsible for instantiating a supported provider before
        calling.
    symbol, interval, start, end
        Threaded for symmetry with the shell's public signature. ``start``
        and ``end`` are unused by the core loop today — the provider
        already scoped its data window at construction time — but are
        kept in the signature so a future pynecore ``Provider.stream``
        implementation can honour them without a signature bump.
    params
        Reserved for C5 (compiler inputs). Threaded but unused in M1.
    timeout_s
        Wall-clock cap in seconds (default ``DEFAULT_TIMEOUT_S = 30``).
    asset_class
        Optional override for the ``syminfo.asset_class`` field. When
        omitted, the core reads ``provider.asset_class`` opportunistically
        and falls back to ``"equity"``.

    Returns
    -------
    tuple
        ``(results_df, exec_ms, alerts)`` where:

        * ``results_df`` — ``pd.DataFrame`` indexed by tz-aware
          ``DatetimeIndex``, one column per ``plot()`` call.
        * ``exec_ms`` — integer milliseconds of wall-clock time the
          ``ScriptRunner.run_iter()`` loop consumed.
        * ``alerts`` — list of ``{"bar_index", "ts", "message"}`` dicts,
          one per ``alert()`` call.

    Raises
    ------
    PineSecurityError
        The compiled source references a forbidden module (T3 second
        line of defense — the compiler's T1 allowlist is the first).
    PineExecTimeoutError
        The script exceeded ``timeout_s`` wall-clock seconds.
    """
    # --- T3 second line of defense ---------------------------------------
    # See _pynecore_glue for the rationale. The user code is executed
    # through PyneCore which needs __import__; restrictor here is the AST scan.
    offenders = scan_for_forbidden_imports(compiled.source)
    if offenders:
        raise PineSecurityError(
            rule="SEC001",
            node_kind=f"forbidden imports: {sorted(set(offenders))}",
            hint=(
                "This is a T3 sandbox-violation; the compiler (T1) should "
                "have blocked it earlier."
            ),
        )

    # --- Materialise compiled source to a tempfile ----------------------
    # PyneCore's import_script(path) re-opens the file to verify the @pyne
    # magic docstring and then import_module-loads the stem. We give it a
    # tempfile whose stem will not collide with installed modules.
    source = ensure_pyne_header(compiled.source)
    sha_tag = (compiled.sha or "anon")[:12]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=f"pyne_{sha_tag}_",
        encoding="utf-8",
        delete=False,
    ) as fh:
        fh.write(source)
        script_path = Path(fh.name)

    # --- Build syminfo + lazy PyneCore imports --------------------------
    primary_symbol = symbol or getattr(provider, "symbol", "BYO")
    primary_interval = interval or getattr(provider, "interval", None) or "1D"
    si = make_default_syminfo(
        primary_symbol,
        primary_interval,
        asset_class=(asset_class or getattr(provider, "asset_class", "equity")),
    )

    # ScriptRunner needs the @pyne AST hook installed; importing pynecore
    # (which our sys.path bridge has made importable) registers it.
    from pynecore import lib as _pyne_lib  # noqa: PLC0415
    from pynecore.core.script_runner import ScriptRunner  # noqa: PLC0415

    alerts: list[dict[str, Any]] = []
    timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
    exec_started_ns: int | None = None
    exec_ended_ns: int | None = None

    def _bar_index_now() -> int:
        return int(getattr(_pyne_lib, "bar_index", 0))

    def _ts_now() -> int:
        # _time is millis in PyneCore; convert to seconds for OBB OBBject contract.
        ms = int(getattr(_pyne_lib, "_time", 0) or 0)
        return ms // 1000 if ms else 0

    try:
        # NOTE (bd-78w, SITE 2): This currently calls provider.iter_ohlcv(),
        # not the abstract _DataProviderStub interface's stream()/fetch() that
        # the Pine Extraction Design §6.E0.3 (lines 244 & 461) targets. The
        # reconciliation is deferred to Phase 2B E3 (bd-78w extended scope,
        # per PR #420 review) — once pynecore lands Provider.stream()/fetch()
        # in E1.1, this call switches to stream(symbol, timeframe, start=,
        # end=) and the currently-dead start/end kwargs on run_compiled
        # (# noqa: ARG001 -- carried for shell parity) become active.
        runner = ScriptRunner(
            script_path=script_path,
            ohlcv_iter=provider.iter_ohlcv(),
            syminfo=si,
        )

        # The alert capture + wall-clock cap wrap ``run_iter()`` together so
        # an in-flight alert that exceeds the budget still raises cleanly
        # (the alert callback finishes synchronously inside the script call).
        # Per-yield copy is REQUIRED -- PyneCore reuses lib._plot_data across
        # yields and clears it after each yield (script_runner.py:811). A
        # post-hoc copy via list(runner.run_iter()) captures the same
        # (cleared) dict for every bar. We materialise (candle, dict(plot))
        # inside the generator so the snapshot is taken BEFORE the clear.
        candles_and_plots: list[tuple[Any, dict[str, Any]]] = []
        with capture_alerts(
            alerts,
            bar_index_getter=_bar_index_now,
            timestamp_getter=_ts_now,
        ), enforce_limits(timeout_s=timeout):
            exec_started_ns = _time.perf_counter_ns()
            # Indicators yield 2-tuples; strategies yield 3-tuples. We only
            # care about (candle, plot_data) here -- the trades tuple element
            # is consumed by the strategy emitter in Phase 2.
            for tup in runner.run_iter():
                candle = tup[0]
                plot_data = tup[1]
                candles_and_plots.append((candle, dict(plot_data) if plot_data else {}))
            exec_ended_ns = _time.perf_counter_ns()
    except PineExecTimeoutError:
        # Make sure we still measure elapsed even on timeout, for debug.
        if exec_ended_ns is None:
            exec_ended_ns = _time.perf_counter_ns()
        raise
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover -- best-effort cleanup
            pass

    # --- Emit raw pieces (envelope is the shell's job) ------------------
    results_df = _collect_results(candles_and_plots)

    if exec_started_ns is None or exec_ended_ns is None:
        exec_ms = 0
    else:
        exec_ms = max(0, (exec_ended_ns - exec_started_ns) // 1_000_000)

    return results_df, int(exec_ms), alerts


__all__ = ["run_compiled", "_collect_results"]
