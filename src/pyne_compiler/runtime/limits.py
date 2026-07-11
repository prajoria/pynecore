"""Per-script wall-clock and memory caps for the Pine runtime.

Per D2 section 10.1 + PRD section 5.2 T2.

The runtime wraps every ``ScriptRunner.run_iter()`` call in
``enforce_limits()``:

* **POSIX** -- ``signal.SIGALRM`` fires the handler after ``timeout_s``
  wall-clock seconds; ``resource.setrlimit(RLIMIT_AS, ...)`` clamps the
  process's virtual-memory footprint to 2 GiB. The signal handler raises
  ``PineExecTimeoutError``, which propagates out of the user script.

* **Windows** -- there is no ``SIGALRM``. We fall back to a
  ``threading.Timer`` that calls ``_thread.interrupt_main()``; that
  injects ``KeyboardInterrupt`` into the main thread, which the context
  manager catches and re-raises as ``PineExecTimeoutError``. ``RLIMIT_AS``
  is unavailable; the operator README recommends container deployment
  for hard memory caps on Windows.

In all cases the alarm / timer is cleared in ``finally`` so an early
return (or an unrelated exception) cannot leave a stale handler armed.

``timeout_s=0`` is treated as "no wall-clock cap" -- useful for callers
that only want the RLIMIT_AS effect (or for unit tests that just want to
prove the context manager is a no-op when disabled).
"""

from __future__ import annotations

import signal
import sys
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Iterator

from pyne_compiler.errors.base import PineExecTimeoutError

# --- Public constants (D2 section 10.1) ---------------------------------------

DEFAULT_TIMEOUT_S: int = 30
"""Per-script wall-clock budget in seconds. Overridable via
``pine.settings.exec_timeout_s`` (P4 wiring)."""

DEFAULT_RLIMIT_AS: int = 2 * 1024 ** 3
"""Virtual-memory cap in bytes (2 GiB). POSIX only."""

MAX_BARS_PER_REQUEST: int = 5_000_000
"""Soft cap on bars consumed per request. Anything above raises
``PineDataValidationError`` at the provider layer (R7 territory)."""


_HAS_SIGALRM: bool = hasattr(signal, "SIGALRM")


# --- POSIX implementation -----------------------------------------------------


def _posix_timeout_handler(timeout_s: int):
    """Build the SIGALRM handler closure that raises PineExecTimeoutError."""

    def _handler(_signum: int, _frame: FrameType | None) -> None:  # noqa: ARG001
        raise PineExecTimeoutError(
            f"Pine script exceeded {timeout_s}s wall-clock budget"
        )

    return _handler


@contextmanager
def _posix_enforce(timeout_s: int) -> Iterator[None]:
    """POSIX implementation: SIGALRM + RLIMIT_AS."""
    import resource  # noqa: PLC0415  -- POSIX-only import kept local

    previous_handler = None
    alarm_armed = False
    if timeout_s > 0:
        previous_handler = signal.signal(
            signal.SIGALRM, _posix_timeout_handler(timeout_s)
        )
        signal.alarm(timeout_s)
        alarm_armed = True

    # Best-effort virtual-memory cap. We clamp DOWN -- if the existing soft
    # limit is already tighter, leave it alone (a parent runner may have
    # narrowed the budget further on purpose).
    rlimit_set = False
    try:
        if hasattr(resource, "RLIMIT_AS"):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # ``-1`` (RLIM_INFINITY) means "no current cap"; treat it as
            # infinity for the min() so DEFAULT_RLIMIT_AS wins.
            current = soft if soft != resource.RLIM_INFINITY else DEFAULT_RLIMIT_AS
            target = min(current, DEFAULT_RLIMIT_AS)
            # Don't tighten beyond the hard limit; if hard is RLIM_INFINITY we
            # can set whatever we want.
            if hard != resource.RLIM_INFINITY:
                target = min(target, hard)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (target, hard))
                rlimit_set = True
            except (ValueError, OSError):  # pragma: no cover -- platform quirk
                # Some sandboxes (Docker user namespaces) refuse RLIMIT_AS.
                # Don't fail the request; T2 is best-effort on those hosts.
                pass
        yield
    finally:
        if alarm_armed:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        # We intentionally do NOT restore RLIMIT_AS -- tightening a process's
        # virtual-memory cap is a one-way ratchet on most platforms.
        # See ``man 2 setrlimit`` re. hard limits.
        del rlimit_set  # silence the linter on POSIX, keep the variable for debugging


# --- Windows fallback ---------------------------------------------------------


@contextmanager
def _windows_enforce(timeout_s: int) -> Iterator[None]:
    """Windows fallback: ``threading.Timer`` + ``_thread.interrupt_main()``.

    Best-effort -- depends on the main thread periodically yielding to the
    Python interpreter (true for almost all Python code, but a CPU-bound
    C extension may delay the interrupt until it returns). RLIMIT_AS is
    not available; operators wanting hard memory caps on Windows should
    deploy under a container with ``--memory`` (PRD section 5.3).
    """
    if timeout_s <= 0:
        # No wall-clock cap requested -- pure no-op.
        yield
        return

    import _thread  # noqa: PLC0415

    fired = threading.Event()

    def _fire() -> None:
        fired.set()
        # ``interrupt_main()`` injects KeyboardInterrupt into the main
        # thread the next time it yields to the interpreter.
        _thread.interrupt_main()

    timer = threading.Timer(timeout_s, _fire)
    timer.daemon = True
    timer.start()
    try:
        yield
    except KeyboardInterrupt as exc:
        if fired.is_set():
            raise PineExecTimeoutError(
                f"Pine script exceeded {timeout_s}s wall-clock budget"
            ) from exc
        # User-triggered Ctrl-C while the timer is still pending: re-raise
        # as-is so the operator gets the usual KeyboardInterrupt experience.
        raise
    finally:
        timer.cancel()


# --- Public API ---------------------------------------------------------------


@contextmanager
def enforce_limits(timeout_s: int = DEFAULT_TIMEOUT_S) -> Iterator[None]:
    """Wall-clock + memory cap context manager.

    Raises :class:`PineExecTimeoutError` if user code inside the block
    exceeds ``timeout_s`` wall-clock seconds. On POSIX, also caps the
    process's virtual-memory footprint to ``DEFAULT_RLIMIT_AS`` (2 GiB).
    On Windows, the memory cap is best-effort -- see module docstring.

    Parameters
    ----------
    timeout_s
        Wall-clock budget in seconds. ``0`` disables the cap entirely
        (useful for tests + for cooperative pre-flight checks that just
        want the RLIMIT_AS side effect on POSIX).

    Examples
    --------
    >>> from openbb_pine.runtime.limits import enforce_limits
    >>> with enforce_limits(timeout_s=30):
    ...     run_compiled_pine_module(...)  # doctest: +SKIP
    """
    if _HAS_SIGALRM:
        with _posix_enforce(timeout_s):
            yield
    else:
        with _windows_enforce(timeout_s):
            yield


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_RLIMIT_AS",
    "MAX_BARS_PER_REQUEST",
    "enforce_limits",
]


# Re-export the platform flag so callers can branch ergonomically without
# poking at ``signal`` themselves.
if not _HAS_SIGALRM:  # pragma: no cover -- platform-conditional
    assert sys.platform.startswith("win"), (
        "No SIGALRM on a non-Windows platform -- limits.py needs a third path."
    )
