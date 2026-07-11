"""D5 §5.3 — ``_install_secondaries_hook`` monkey-patch context manager.

Wraps ``ScriptRunner.run_iter()`` so PyneCore's ``request.security()`` reads
from the DataFrames :func:`openbb_pine.runtime.security_dispatcher.
prefetch_security_contexts` already produced, rather than trying to resolve
the request itself (which would either fail — the vendored PyneCore's
``request.security`` deliberately raises ``RuntimeError`` because it expects
its own SecurityTransformer AST rewrite to have handled the call — or, in
future when we integrate PyneCore's transformer, spin up its heavyweight
multi-process security machinery for what we've already prefetched).

The hook is a **single-call substitute**: when the compiled script calls
``pynecore.lib.request.security(symbol, timeframe, expression)`` at bar
``N``, we:

1. Look up ``(symbol, timeframe)`` in ``security_contexts`` to find the
   matching ``context_id`` (C3 populated the map at compile time). Static
   lookup — a miss raises :class:`PineSecurityContextNotFoundError` so the
   operator sees the mismatch immediately rather than a silent nan.
2. Fetch ``secondaries[context_id]`` — a ``pd.DataFrame`` already
   forward-filled to the primary bar grid by the c1x dispatcher (D5 §4.2
   step 4). We index it via ``iloc[current_bar_index]``.
3. Coerce ``expression`` (which for M2 is one of PyneCore's ``Source``
   sentinels — ``close`` / ``volume`` / ``high`` / … — see
   ``pynecore.types.source.Source``) into a column name and return the
   scalar value.

For M2 we support only static symbol + static timeframe contexts. Dynamic
contexts (``dynamic_symbol`` / ``dynamic_timeframe`` per D5 §4.4) raise
:class:`PineSecurityContextNotFoundError(reason="dynamic_unsupported")`
so the failure mode is loud and documented.

Design decisions locked in per D5 §5.3
--------------------------------------

* **Manual save/restore** rather than ``unittest.mock.patch.object`` — the
  runtime isn't a test surface and pulling ``unittest`` into the runtime
  import graph is unnecessary weight. The ``try/finally`` shape is one
  screen and matches the ``capture_alerts`` precedent in ``_pynecore_glue``.
* **Callable bar-index accessor** rather than a snapshot int. The executor
  advances PyneCore's ``lib.bar_index`` state per yield; passing a
  ``Callable[[], int]`` lets the hook read the current bar on every call
  without the executor having to reach into our internals mid-loop
  (mirrors how ``capture_alerts`` accepts ``bar_index_getter`` /
  ``timestamp_getter``).
* **Nested install is safe (via closure capture, not module-level LIFO)**
  — every ``install_secondaries_hook(...)`` call snapshots the current
  ``pynecore.lib.request.security`` into a closure named ``original``. When
  the ``with`` block exits, ``finally`` restores exactly that captured
  value. There is NO module-level LIFO stack: nested-install correctness
  falls out of Python closures, not out of a tracker we maintain. Two
  consequences worth flagging: (a) if a third party mutates
  ``request.security`` mid-flight between two nested installs, the mid-flight
  value is what the inner ``original`` captured, so unwind restores that
  third-party value before restoring PyneCore's stub — realistic in a
  pytest fixture that patches then yields; (b) a "double install-uninstall"
  race across two threads would corrupt the module-global (see thread-safety
  note below).
* Only ``pynecore.lib.request.security`` and ``request.security_lower_tf``
  are patched. ``security_lower_tf`` (LTF intrabar reads) is out of scope
  for M2, but we still patch it — raising :class:`PineUnsupportedBuiltinError`
  with the M2-scope tracking URL — so the failure surface is a
  Pine-typed error, not a bare ``RuntimeError`` from PyneCore's stub whose
  message ("rewritten by SecurityTransformer during compilation") is
  misleading in our pipeline (we deliberately do NOT run SecurityTransformer).
* **Thread safety.** The hook mutates the module-global
  ``pynecore.lib.request.security``. Concurrent ``run_compiled()`` calls
  in the same process would trample each other's ``_hook_security``
  closures — the last install wins, and a script for symbol A could
  return values for symbol B. M2 does NOT support concurrent Pine
  execution: the executor caller MUST serialize ``run_compiled()``
  invocations (e.g. via ``asyncio.Lock`` in the FastAPI handler). We
  install a ``threading.Lock``-guarded install/uninstall so that a
  second thread attempting to install while the first is patched raises
  loudly (:class:`RuntimeError`) rather than silently corrupting. See
  D5 §5.3 for the design decision; M3+ tracks moving state into
  :mod:`contextvars` for per-request scoping.

Clean-room: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pandas as pd

from pyne_compiler.errors.base import (
    PineSecurityContextNotFoundError,
    PineUnsupportedBuiltinError,
)

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from pyne_compiler.compiler.types import SecurityContext


__all__ = [
    "install_secondaries_hook",
    "expression_column_name",
]


_log = logging.getLogger(__name__)


# --- Thread-safety guard ----------------------------------------------------
#
# The hook mutates the module-global ``pynecore.lib.request.security``. If
# two threads install concurrently, the second install would wrap the
# first's hook (fine for LIFO unwind on ONE thread), but each thread's
# closure captures its own ``secondaries`` / ``security_contexts`` /
# ``get_current_bar_index`` — so once the second install completes, ALL
# ``request.security`` calls (from either thread) resolve through the
# second thread's closure. That is silent cross-request data corruption.
#
# We use an ``RLock`` here — installing on the SAME thread nests correctly
# (the closure-capture-based LIFO in :func:`install_secondaries_hook`
# handles that). A DIFFERENT thread attempting to install while the lock is
# held raises loudly rather than corrupting: the caller (executor / REST
# handler) is expected to serialize ``run_compiled()`` invocations in M2.
# M3+ tracks moving state into :mod:`contextvars` per D5 §5.3 for true
# per-request scoping without a global lock.
_INSTALL_LOCK = threading.RLock()


# --- Expression -> column-name helper ---------------------------------------


def expression_column_name(expression: Any) -> str:
    """Return the DataFrame column name for a PyneCore ``expression`` arg.

    PyneCore's ``request.security(symbol, timeframe, expression)`` accepts
    the built-in ``Source`` sentinels ``close`` / ``volume`` / ``open`` /
    ``high`` / ``low`` / ``hl2`` / ``hlc3`` / ``ohlc4`` (see
    ``pynecore.types.source.Source``). Those objects render their name via
    ``__str__`` — we prefer :attr:`Source.name` when available so we don't
    depend on the ``__str__`` shape.

    Plain strings (``"close"``, ``"volume"``) pass through unchanged so
    tests that don't want to import PyneCore's ``Source`` type can drive
    the hook directly.

    Anything else (arbitrary expressions like ``close + open``) falls
    outside M2 scope — the c1x dispatcher already forward-fills whole
    OHLCV frames, so only the sentinel columns are addressable here.
    Callers that hand us something exotic get a ``TypeError`` with the
    offending value in the message rather than an ``AttributeError`` deep
    in the hook.

    Type-check strictness
    ---------------------
    We deliberately do NOT accept "any object with a truthy ``.name``
    string" — that would silently swallow a ``pd.Series(name="close")``
    or ``pathlib.Path("/tmp/close")`` and produce a wrong-but-plausible
    column name. Accepted expression shapes:

    * ``str`` — pass-through (the primary "cheap test doubles" path);
    * an instance of PyneCore's :class:`pynecore.types.source.Source`
      (imported lazily to avoid pulling PyneCore into this module's
      import graph for callers that never install the hook);
    * an object whose class name is ``"Source"`` or ``"_FakeSource"``
      (the test-double name used by
      :file:`tests/unit/test_security_hook_monkey_patch.py`). We match
      by class name — not just ``hasattr("name")`` — because arbitrary
      objects (Series, DataFrames, Paths, dataclasses) happen to expose
      a ``.name`` attribute.

    Anything else raises ``TypeError`` so a codegen or wiring bug
    surfaces at the boundary, not deep inside the hook.
    """
    # 1) Plain string — pass through (fastest path; used by cheap tests).
    if isinstance(expression, str):
        return expression
    # 2) PyneCore's Source sentinel — the production shape. Import lazily
    # so this module has no hard dependency on PyneCore for callers that
    # never install the hook (e.g. static-analysis-only imports).
    try:
        from pynecore.types.source import Source  # noqa: PLC0415
    except ImportError:  # pragma: no cover -- PyneCore always present in runtime
        Source = None  # type: ignore[assignment]
    if Source is not None and isinstance(expression, Source):
        name = getattr(expression, "name", None)
        if isinstance(name, str) and name:
            return name
        # Source with an empty/missing name is a codegen bug — fall through
        # to the TypeError below so the operator sees the offending value.
    # 3) Duck-typed Source-alike: match by class name to keep out
    # accidental hits like ``pd.Series(name=...)`` / ``pathlib.Path``.
    # This is the escape hatch for test doubles that don't want to
    # subclass PyneCore's ``Source``.
    cls_name = type(expression).__name__
    if cls_name in {"Source", "_FakeSource"}:
        name = getattr(expression, "name", None)
        if isinstance(name, str) and name:
            return name
    # 4) Anything else: refuse with a message that names the offender.
    raise TypeError(
        f"request.security(): unsupported expression type "
        f"{type(expression).__name__!r} (value={expression!r}); M2 hook "
        "supports built-in Source sentinels (close, volume, high, low, "
        "open, hl2, hlc3, ohlc4) and plain column-name strings."
    )


# --- Context resolution ------------------------------------------------------


def _format_available_keys(
    security_contexts: dict[str, "SecurityContext"],
) -> list[str]:
    """Normalise ``available_keys`` into the pre-formatted
    ``"ctx_id: 'SYMBOL'@'TF'"`` shape used by :class:`PineSecurityContextNotFoundError`.

    Both raise sites in this module (:func:`_lookup_context_id` and the
    ``secondaries``-missing branch of :func:`install_secondaries_hook`)
    route through this helper so downstream JSON consumers see one
    consistent shape regardless of which code path raised. The class
    docstring in :mod:`openbb_pine.errors` mandates this shape.
    """
    return [
        f"{cid}: {ctx.symbol!r}@{ctx.timeframe!r}"
        for cid, ctx in security_contexts.items()
    ]


def _context_is_dynamic(ctx: "SecurityContext") -> bool:
    """Mirror ``security_dispatcher._context_is_dynamic`` — but resolved
    at the hook boundary so a dynamic context surfaces as a runtime error
    instead of silently returning empty (the dispatcher path).

    Uses ``getattr`` fallbacks so this file stays compatible with the
    Wave-1 ``SecurityContext`` shape (bead d75) and the Wave-2 shape (bead
    y86) that adds ``dynamic_symbol`` / ``dynamic_timeframe`` fields —
    Wave-2 subagents develop concurrently, so we can't assume the fields
    exist yet.
    """
    return bool(
        getattr(ctx, "dynamic_symbol", False)
        or getattr(ctx, "dynamic_timeframe", False)
    )


def _lookup_context_id(
    symbol: str,
    timeframe: str,
    security_contexts: dict[str, "SecurityContext"],
) -> str:
    """Resolve ``(symbol, timeframe)`` → ``context_id``.

    Raises :class:`PineSecurityContextNotFoundError` with the full list of
    known ``(symbol, timeframe)`` pairs so a mismatch (e.g. ``"1D"`` vs
    ``"1d"``) is visible in one error message rather than a silent NaN.
    """
    for cid, ctx in security_contexts.items():
        if ctx.symbol == symbol and ctx.timeframe == timeframe:
            return cid
    # Miss — build a "did-you-mean"-style list for the operator.
    raise PineSecurityContextNotFoundError(
        symbol=symbol,
        timeframe=timeframe,
        reason="not_found",
        available_keys=_format_available_keys(security_contexts),
    )


# --- The hook ---------------------------------------------------------------


@contextmanager
def install_secondaries_hook(
    secondaries: dict[str, pd.DataFrame],
    security_contexts: dict[str, "SecurityContext"],
    *,
    get_current_bar_index: Callable[[], int],
) -> Iterator[None]:
    """Monkey-patch ``pynecore.lib.request.security`` for one executor pass.

    Parameters
    ----------
    secondaries
        The ``{context_id: DataFrame}`` map that
        :func:`openbb_pine.runtime.security_dispatcher.prefetch_security_contexts`
        produced. Each DataFrame is already forward-filled to the primary
        bar index (D5 §4.2 step 4); we index it with ``iloc``. A frame may
        be *empty* (``df.empty`` True) — that happens when the dispatcher
        routed a dynamic context to a deferred per-bar-lazy-fetch code
        path (see ``security_dispatcher.py`` step 3, "dynamic contexts
        defer to the executor's per-bar path"). M2 does NOT support that
        path yet; the hook detects the empty frame at read time and
        raises :class:`PineSecurityContextNotFoundError(reason=
        "dynamic_unsupported")` rather than letting ``iloc[0]`` blow up
        with a bare :class:`IndexError`.
    security_contexts
        The compiler-populated ``{context_id: SecurityContext}`` map from
        :attr:`openbb_pine.compiler.types.CompiledModule.security_contexts`.
        We iterate it once per hook call to resolve ``(symbol, timeframe)``
        → ``context_id``; the map is expected to be tiny (a handful of
        contexts) so no lookup dict is warranted.
    get_current_bar_index
        Callable returning the current primary-series bar index, called
        once per ``request.security()`` invocation. The executor threads
        this from ``pynecore.lib.bar_index`` (already stateful — see
        ``executor.run_compiled``'s ``_bar_index_now`` helper for the
        precedent alongside ``capture_alerts``). MUST be in
        ``[0, len(secondary))``; a negative value is a bug (executor
        underflow, wrap-around) and the hook raises :class:`IndexError`
        rather than silently returning ``iloc[-1]`` — that would be a
        Pine lookahead-bias defect (the last row is the *future* to the
        current bar).

    Yields
    ------
    None
        Enter the ``with`` block; ``pynecore.lib.request.security`` and
        ``pynecore.lib.request.security_lower_tf`` are patched for the
        block's lifetime and restored on exit (``try/finally`` —
        restoration runs even if the compiled script raises).

    Behaviour of the patched function
    --------------------------------

    Signature matches PyneCore's ``request.security(symbol, timeframe,
    expression, **kwargs)`` — extra kwargs (``gaps``, ``lookahead``,
    ``ignore_invalid_symbol``, …) are accepted for future compatibility
    but ignored in M2 (they were already applied by the c1x dispatcher's
    forward-fill and would need per-bar handling that D5 defers to M3+).

    ``request.security_lower_tf`` is also patched to raise
    :class:`PineUnsupportedBuiltinError` on call. Rationale: PyneCore's
    stub raises a bare ``RuntimeError`` whose message points at
    "SecurityTransformer during compilation" — misleading in our
    pipeline, which deliberately does NOT run SecurityTransformer.
    Surfacing the M2-scope decision as a Pine-typed error keeps the
    error surface uniform.

    Return type mirrors what the field-name column contains at
    ``iloc[current_bar_index]`` — typically a ``float`` for ``close`` /
    ``volume``. Tuple-valued ``expression=[close, volume]`` is out of
    M2 scope (see :func:`expression_column_name`); the compiler's
    codegen for tuple returns is a separate bead.

    Raises
    ------
    PineSecurityContextNotFoundError
        * When ``(symbol, timeframe)`` doesn't match any known context
          (``reason="not_found"``).
        * When the matched context has its ``dynamic_symbol`` /
          ``dynamic_timeframe`` flag set (``reason="dynamic_unsupported"``,
          M2 static-only).
        * When the resolved ``context_id`` has no entry in
          ``secondaries`` (dispatcher wire-up bug; ``reason="not_found"``).
        * When the resolved ``context_id`` maps to an empty DataFrame
          (dispatcher routed to deferred dynamic fetch;
          ``reason="dynamic_unsupported"``).

        The error carries ``available_keys`` in the pre-formatted
        ``"ctx_id: 'SYMBOL'@'TF'"`` shape from
        :func:`_format_available_keys` so downstream JSON consumers see
        one consistent shape.
    IndexError
        When ``get_current_bar_index()`` returns a value outside
        ``[0, len(secondary))``. The message names the bar index, the
        context id, and the frame length so the operator can attribute
        the wire-up bug (executor state-machine underflow vs. dispatcher
        forward-fill mismatch).
    KeyError
        When the column resolved from ``expression`` is not present in
        the prefetched frame (typically a codegen bug that emitted a
        non-OHLCV column name). Message names the missing column and
        lists the columns that ARE present.
    TypeError
        When ``expression`` is not a supported shape (see
        :func:`expression_column_name`). Raised at the boundary — no
        deep-in-hook ``AttributeError`` slippage.
    PineUnsupportedBuiltinError
        When the compiled script calls ``request.security_lower_tf``.
        M2 does not support LTF intrabar reads; the compiler is expected
        to reject them at type-check time, so a hit here is either a
        compiler-gate escape or a bring-your-own script that bypasses
        our type checker.
    RuntimeError
        When a second thread attempts to enter :func:`install_secondaries_hook`
        while another thread already holds the install lock — M2 does
        not support concurrent Pine execution (see module docstring).
    """
    # Local import to keep the pynecore sys.path bridge ordering intact —
    # the runtime layer imports pynecore lazily so unit tests that don't
    # touch runtime code don't pay the import cost. Matches the pattern
    # already established in ``_pynecore_glue.capture_alerts`` (see line
    # 176 of that file).
    from pynecore.lib import request as request_module  # noqa: PLC0415

    # Thread-safety guard — see module docstring. Same-thread nested
    # installs are fine (``_INSTALL_LOCK`` is an RLock); cross-thread
    # concurrent installs raise loudly instead of corrupting.
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "install_secondaries_hook: another thread is already inside "
            "an installed hook. M2 does not support concurrent Pine "
            "execution — serialize run_compiled() invocations in the "
            "caller (e.g. asyncio.Lock in the FastAPI handler). See "
            "openbb_pine.runtime.security_hook module docstring."
        )

    # Snapshot the current values BEFORE we patch — Python closure capture
    # (in the ``finally`` block below) is what unwinds nested installs in
    # LIFO order; there is NO module-level stack tracking installs. If a
    # third-party (e.g. a pytest fixture) mutated ``request.security`` /
    # ``request.security_lower_tf`` between an outer install and this one,
    # we capture that third-party value here, and ``finally`` restores it
    # on exit — realistic surprise mode noted in the module docstring.
    original_security = request_module.security
    original_security_lower_tf = request_module.security_lower_tf

    def _hook_security(
        symbol: Any,
        timeframe: Any,
        expression: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        """Bar-time lookup that replaces PyneCore's stub.

        We coerce ``symbol`` / ``timeframe`` to ``str`` explicitly so the
        hook doesn't refuse a caller that hands us ``syminfo.ticker`` (a
        Source-like object whose ``__str__`` gives the underlying string)
        even after C3 has statically resolved it. If C3 couldn't resolve
        statically we're already in the dynamic-context branch, which
        raises via ``_context_is_dynamic``.
        """
        sym_s = str(symbol)
        tf_s = str(timeframe)

        # Resolve (symbol, timeframe) → context_id. Missing → raise with
        # the full context list attached.
        context_id = _lookup_context_id(sym_s, tf_s, security_contexts)

        # M2 only supports static contexts. If the matched context is
        # dynamic, surface a documented, actionable error rather than
        # silently taking the slow path we haven't built yet.
        ctx = security_contexts[context_id]
        if _context_is_dynamic(ctx):
            raise PineSecurityContextNotFoundError(
                symbol=sym_s,
                timeframe=tf_s,
                reason="dynamic_unsupported",
                context_id=context_id,
            )

        # Look up the pre-aligned secondary frame. Missing context_id in
        # ``secondaries`` while present in ``security_contexts`` means
        # the dispatcher never ran (wire-up bug) — raise the same class
        # so the surface stays uniform. This is defensive; production
        # code paths always populate both maps together.
        if context_id not in secondaries:
            raise PineSecurityContextNotFoundError(
                symbol=sym_s,
                timeframe=tf_s,
                reason="not_found",
                context_id=context_id,
                available_keys=_format_available_keys(security_contexts),
                message=(
                    f"context_id {context_id!r} resolved from "
                    f"({sym_s!r}, {tf_s!r}) has no prefetched frame "
                    "(dispatcher may not have run)"
                ),
            )

        df = secondaries[context_id]

        # Empty-frame guard — the c1x dispatcher writes ``pd.DataFrame()``
        # for a context routed to the deferred per-bar-lazy-fetch path
        # (see security_dispatcher.py step 3 for dynamic contexts). That
        # path is not built out in M2; without this guard, ``iloc[0]``
        # would raise a bare ``IndexError`` far removed from the
        # documented failure surface. Detecting it here lets us raise the
        # documented ``dynamic_unsupported`` error with full context.
        if df.empty:
            raise PineSecurityContextNotFoundError(
                symbol=sym_s,
                timeframe=tf_s,
                reason="dynamic_unsupported",
                context_id=context_id,
                message=(
                    f"context_id {context_id!r} resolved from "
                    f"({sym_s!r}, {tf_s!r}) has an empty prefetched "
                    "frame — the dispatcher routed this context to the "
                    "deferred per-bar lazy fetch (D5 §4.4), which is "
                    "not yet supported in M2."
                ),
            )

        column = expression_column_name(expression)

        # Bar-index read. Explicit range check because ``iloc[-N]`` in
        # pandas is a valid backwards index and would silently return
        # the LAST row of the secondary — a Pine lookahead-bias defect
        # (the strategy would see future data as if it were the current
        # bar). ``iloc[N]`` past the end already raises ``IndexError``
        # but with a generic pandas message; we produce our own so the
        # operator sees the context id and the frame length in one shot.
        bar_index = get_current_bar_index()
        if bar_index < 0 or bar_index >= len(df):
            raise IndexError(
                f"request.security(): bar_index {bar_index} out of range "
                f"for prefetched frame for context {context_id!r} "
                f"(len={len(df)}); this is a wire-up bug — the "
                "dispatcher forward-fills to the primary index (D5 §4.2 "
                "step 4)."
            )
        row = df.iloc[bar_index]

        # A missing column raises KeyError with the column name and the
        # list of columns that ARE present — friendlier than a bare NaN
        # and useful for a codegen or dispatcher-alignment bug.
        if column not in row.index:
            raise KeyError(
                f"request.security(): column {column!r} not present in "
                f"prefetched frame for context {context_id!r} "
                f"(available columns: {list(row.index)})"
            )
        return row[column]

    def _hook_security_lower_tf(*_args: Any, **_kwargs: Any) -> Any:
        """M2 does not support LTF intrabar reads.

        PyneCore's stub raises a bare :class:`RuntimeError` whose message
        points at ``SecurityTransformer during compilation`` — misleading
        in our pipeline. Substitute a Pine-typed error carrying the
        M2-scope decision so the operator can file the follow-up bead
        against the right tracking label.
        """
        raise PineUnsupportedBuiltinError(
            builtin="request.security_lower_tf",
            suggested_alternative=(
                "M2 does not support LTF intrabar reads; fetch the lower "
                "timeframe as a separate security context and align it in "
                "your Pine logic."
            ),
            tracking_url=(
                "https://github.com/OpenBB-finance/OpenBBTerminal/"
                "issues?label=pine-builtin&q=security_lower_tf"
            ),
        )

    request_module.security = _hook_security  # type: ignore[assignment]
    request_module.security_lower_tf = _hook_security_lower_tf  # type: ignore[assignment]
    try:
        yield
    finally:
        # Restore. Runs even if the compiled script raises inside the
        # ``with`` block, so a script crash never leaves the runtime
        # module in a patched state that would corrupt the next call.
        # LIFO unwind is emergent from closure capture (see the snapshot
        # comment above), not from a module-level tracker.
        request_module.security = original_security  # type: ignore[assignment]
        request_module.security_lower_tf = original_security_lower_tf  # type: ignore[assignment]
        _INSTALL_LOCK.release()
