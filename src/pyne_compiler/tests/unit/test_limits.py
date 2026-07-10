"""Tests for ``openbb_pine.runtime.limits``.

D2 section 10.1 / PRD section 5.2 T2: per-script wall-clock + memory caps.
POSIX path uses ``signal.SIGALRM`` + ``resource.setrlimit``; Windows path
falls back to a ``threading.Timer`` that interrupts the main thread.

These tests pick small timeouts (``timeout_s=1``) so the suite runs fast.
POSIX-only tests are skipped on Windows via ``pytest.mark.skipif``.
"""

from __future__ import annotations

import signal
import sys
import time

import pytest

from openbb_pine.errors import PineExecTimeoutError
from openbb_pine.runtime.limits import (
    DEFAULT_RLIMIT_AS,
    DEFAULT_TIMEOUT_S,
    MAX_BARS_PER_REQUEST,
    enforce_limits,
)


_HAS_SIGALRM = hasattr(signal, "SIGALRM")


# ---------- constants ----------------------------------------------------


def test_default_constants_match_d2_spec():
    """D2 section 10.1 -- timeout 30s, 2 GiB RLIMIT_AS, 5M bar cap."""
    assert DEFAULT_TIMEOUT_S == 30
    assert DEFAULT_RLIMIT_AS == 2 * 1024 ** 3
    assert MAX_BARS_PER_REQUEST == 5_000_000


# ---------- happy path: fast function completes normally ------------------


def test_fast_function_returns_normally():
    """A function that finishes well under the budget must not raise."""
    with enforce_limits(timeout_s=5):
        x = sum(range(100))
    assert x == 4950


def test_enforce_limits_yields_no_value():
    """The contextmanager yields ``None`` -- ``with enforce_limits() as v:``
    should bind ``v`` to ``None``."""
    with enforce_limits(timeout_s=5) as v:
        assert v is None


def test_zero_timeout_disables_wall_clock():
    """``timeout_s=0`` means no SIGALRM is scheduled -- a long sleep would
    normally be a violation but with 0 we just don't enforce."""
    # Keep the sleep tiny so the test stays fast even with a zero-cap that
    # turns into a no-op.
    with enforce_limits(timeout_s=0):
        time.sleep(0.01)


# ---------- adversarial: infinite loop / sleep triggers timeout -----------


@pytest.mark.skipif(not _HAS_SIGALRM, reason="POSIX-only SIGALRM test")
def test_long_sleep_raises_pine_exec_timeout_error_posix():
    """``time.sleep(60)`` with ``timeout_s=1`` must raise PineExecTimeoutError
    inside the context manager."""
    start = time.monotonic()
    with pytest.raises(PineExecTimeoutError):
        with enforce_limits(timeout_s=1):
            time.sleep(60)
    # And the alarm should fire near 1s, not 60s.
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"timeout fired too late: {elapsed:.2f}s"


@pytest.mark.skipif(not _HAS_SIGALRM, reason="POSIX-only SIGALRM test")
def test_busy_loop_raises_pine_exec_timeout_error_posix():
    """A tight ``while True: pass`` busy loop must also be interruptible.
    On POSIX SIGALRM fires regardless of GIL state."""
    start = time.monotonic()
    with pytest.raises(PineExecTimeoutError):
        with enforce_limits(timeout_s=1):
            # Bounded to prevent runaway on broken platforms.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                pass
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"timeout fired too late: {elapsed:.2f}s"


def test_timeout_error_message_mentions_budget():
    """The raised PineExecTimeoutError message must name the wall-clock
    budget so operators understand which knob to twist."""
    if not _HAS_SIGALRM:
        pytest.skip("Windows path is best-effort; message asserted on POSIX")
    with pytest.raises(PineExecTimeoutError, match=r"1s"):
        with enforce_limits(timeout_s=1):
            time.sleep(10)


# ---------- alarm is cleared on normal exit (no leaked signal) ------------


@pytest.mark.skipif(not _HAS_SIGALRM, reason="POSIX-only SIGALRM cleanup test")
def test_alarm_cleared_on_normal_exit():
    """If user code finishes before the deadline, the SIGALRM must be
    cancelled in the ``finally`` -- otherwise a later block would receive
    a stray alarm."""
    with enforce_limits(timeout_s=10):
        x = 1 + 1
    # ``signal.alarm(0)`` returns the previous alarm; after the context manager
    # exits cleanly it should be 0.
    leftover = signal.alarm(0)
    assert leftover == 0, f"stray alarm scheduled: {leftover}s"
    assert x == 2


@pytest.mark.skipif(not _HAS_SIGALRM, reason="POSIX-only SIGALRM cleanup test")
def test_alarm_cleared_on_exception():
    """An exception inside the block must still clear the alarm."""
    with pytest.raises(RuntimeError):
        with enforce_limits(timeout_s=10):
            raise RuntimeError("boom")
    leftover = signal.alarm(0)
    assert leftover == 0


# ---------- nested usage --------------------------------------------------


@pytest.mark.skipif(not _HAS_SIGALRM, reason="POSIX-only SIGALRM nest test")
def test_nested_enforce_limits_inner_wins():
    """A nested ``with enforce_limits`` should not crash the outer block --
    the inner replaces the alarm, the outer's finally clears it. This test
    just asserts no spurious exception when both blocks finish fast."""
    with enforce_limits(timeout_s=10):
        with enforce_limits(timeout_s=10):
            time.sleep(0.01)
    # Alarm cleared after both blocks.
    assert signal.alarm(0) == 0


# ---------- Windows-path smoke test --------------------------------------


@pytest.mark.skipif(_HAS_SIGALRM, reason="Windows-only timer fallback test")
def test_windows_timer_raises_on_long_sleep():
    """The Windows fallback uses ``threading.Timer`` + ``_thread.interrupt_main``
    to interrupt the main thread. The contextmanager re-raises that as
    PineExecTimeoutError."""
    with pytest.raises(PineExecTimeoutError):
        with enforce_limits(timeout_s=1):
            time.sleep(10)


def test_windows_fast_function_completes_without_timer_firing():
    """Even on the Windows fallback path a fast function must complete cleanly."""
    with enforce_limits(timeout_s=2):
        result = sum(range(1000))
    assert result == 499_500


# ---------- the module surface is importable on both platforms ------------


def test_module_imports_on_any_platform():
    """Regression: a missing-import or top-level NameError on Windows would
    silently strand the runtime. Re-import the module here to be sure."""
    import importlib

    import openbb_pine.runtime.limits as mod

    importlib.reload(mod)
    assert hasattr(mod, "enforce_limits")
    assert hasattr(mod, "DEFAULT_TIMEOUT_S")
    # Ensure we're not accidentally on a stub when running on Windows.
    if sys.platform == "win32":
        assert not _HAS_SIGALRM
