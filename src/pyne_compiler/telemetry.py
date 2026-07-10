"""D1 §3.5 + PRD §9.4 failure-mode telemetry.

Post-E0.4 shape (see plan Task E0.4 and design doc §6.E0.4):

* :class:`TelemetrySink` — the ``Protocol`` the compiler programs
  against. Post-extraction (E2) this Protocol moves to
  ``pyne_compiler.telemetry``; until then it lives here so a single
  ``from openbb_pine.telemetry import TelemetrySink`` gated by
  ``TYPE_CHECKING`` inside the compiler stays valid.
* :class:`OpenBBTelemetrySink` — the concrete implementation the
  openbb-fork routers (``run_router.py``, later ``strategies_router.py``)
  instantiate and pass to ``compile_pine(telemetry=...)``. Keeps
  per-instance counters (dict[str, int]) — matches the pre-E0.4
  module-level counter behavior on a per-sink basis.
* Module-level ``record_unsupported_*`` / ``get_unsupported_*_counts``
  / ``reset_metrics`` helpers — kept as a thin delegation to a
  module-global :class:`OpenBBTelemetrySink` for one release. This is
  back-compat for tests / operators that call these free functions
  directly (e.g. ``tests/unit/test_error_model.py`` TestTelemetryCounters
  block). The delegation costs one attribute lookup per call and lets us
  remove the free functions in a follow-up bead after every caller
  migrates.

.. warning::

   The module-global :data:`_DEFAULT_SINK` (and by extension the free
   ``record_unsupported_*`` writers) is **compiler-decoupled post-E0.4**.
   Nothing in the compiler or the router path writes to it in production
   any more — the compiler always goes through the injected
   :class:`TelemetrySink`, and the router owns a fresh per-request
   :class:`OpenBBTelemetrySink` that it surfaces on
   ``OBBject.extra["pine_telemetry"]``. Tests that want to observe
   compiler-side telemetry MUST inject a sink via
   ``compile_pine(telemetry=<sink>)`` and read counts off THAT sink;
   calling ``reset_metrics()`` and then inspecting the module-global
   counters after a compile will silently see zeros. The
   ``TestTelemetryCounters`` block in ``tests/unit/test_error_model.py``
   still passes because it calls ``record_unsupported_*`` directly — it
   does NOT prove the compiler wrote through. See
   ``tests/unit/test_telemetry_injection.py`` for the injected-sink
   pattern.

Two counter dimensions per sink:

* ``pine_unsupported_builtin_total{name}`` — incremented every time the
  compiler raises :class:`~openbb_pine.errors.PineUnsupportedBuiltinError`
  for a Pine builtin name. Powers the PRD §3.4 L0.5 wild-corpus
  coverage-shortfall attribution: when a corpus script fails to compile,
  the operator can see WHICH builtins were missing without re-running the
  compiler on the whole corpus.
* ``pine_unsupported_feature_total{name}`` — same shape but for
  :class:`~openbb_pine.errors.PineUnsupportedFeatureError`, keyed by the
  ``PF###`` feature code (or the feature label if no code).

Integration: every ``raise PineUnsupportedBuiltinError(name)`` /
``raise PineUnsupportedFeatureError(...)`` site in the compiler calls the
appropriate ``record_*`` method on the injected sink BEFORE raising. When
no sink is injected (``telemetry=None`` — the compiler default), the
guard short-circuits and no counter fires; the raise still happens.

# TODO(P4): wire the OpenBBTelemetrySink counters into real
# ``prometheus_client.Counter`` instances when the observability bead
# lands. Until then, ``sink.get_*_counts()`` doubles as a test hook and
# a debug surface the ``openbb-pine doctor`` CLI can print.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# --- Protocol the compiler programs against ---------------------------------


@runtime_checkable
class TelemetrySink(Protocol):
    """The compiler's telemetry-emission contract (D1 §3.5).

    ``@runtime_checkable`` here catches the "did you forget to define
    these methods at all" mistake at runtime; behavioural conformance
    (sink actually records when the compiler calls it) is verified
    end-to-end by ``tests/unit/test_telemetry_injection.py``.

    Post-E2 this Protocol lives in ``pyne_compiler.telemetry``. During
    E0.4 it lives here; compiler modules import it via ``TYPE_CHECKING``
    so a rename during E2 is a one-line change per module.
    """

    def record_unsupported_feature(self, name: str) -> None:
        """Increment ``pine_unsupported_feature_total{name}`` on the sink.

        ``name`` should be the ``PF###`` code (preferred) or the feature
        label when no code was assigned. Called BEFORE
        ``raise PineUnsupportedFeatureError(...)`` so the counter is
        accurate even when an outer ``except`` swallows the exception.
        """
        ...

    def record_unsupported_builtin(self, name: str) -> None:
        """Increment ``pine_unsupported_builtin_total{name}`` on the sink.

        ``name`` is the Pine builtin qualified name (e.g. ``"ta.ichimoku"``).
        Called BEFORE ``raise PineUnsupportedBuiltinError(name)`` — see
        the ``_raise_unsupported_builtin`` collect-then-raise pattern in
        :mod:`openbb_pine.compiler.type_checker`.
        """
        ...


# --- Concrete implementation openbb-fork routers instantiate ----------------


class OpenBBTelemetrySink:
    """The openbb-fork's :class:`TelemetrySink` implementation.

    Keeps in-process counters on the instance — one sink per router call
    site (per D3 §4.6, the router builds a fresh sink for each request so
    per-request unsupported-feature counts stay isolated).

    Two-way readable: after :func:`compile_pine` returns, the router calls
    :meth:`get_unsupported_feature_counts` / :meth:`get_unsupported_builtin_counts`
    to surface the counts in the response envelope's ``warnings`` slot.
    :meth:`reset` is intentionally provided for test isolation but is a
    no-op in production (routers use a fresh sink per call).
    """

    def __init__(self) -> None:
        self._feature_counts: dict[str, int] = {}
        self._builtin_counts: dict[str, int] = {}

    # -- record (called by the compiler) -------------------------------------

    def record_unsupported_feature(self, name: str) -> None:
        self._feature_counts[name] = self._feature_counts.get(name, 0) + 1

    def record_unsupported_builtin(self, name: str) -> None:
        self._builtin_counts[name] = self._builtin_counts.get(name, 0) + 1

    # -- read (called by the router after compile_pine returns) --------------

    def get_unsupported_feature_counts(self) -> dict[str, int]:
        """Return a copy of the feature-unsupported counter map.

        A copy (not the live reference) so callers can mutate their view
        without corrupting sink state.
        """
        return dict(self._feature_counts)

    def get_unsupported_builtin_counts(self) -> dict[str, int]:
        """Return a copy of the builtin-unsupported counter map."""
        return dict(self._builtin_counts)

    # -- reset (test isolation) ----------------------------------------------

    def reset(self) -> None:
        """Zero both counter maps.

        Exposed for test isolation and for operators who want to clear
        state between runs of the doctor CLI.
        """
        self._feature_counts.clear()
        self._builtin_counts.clear()


# --- Module-level back-compat shim ------------------------------------------
#
# Pre-E0.4 the compiler recorded to module-level dicts via free functions.
# Tests and operator tooling that call those free functions directly (see
# ``tests/unit/test_error_model.py`` TestTelemetryCounters) keep working
# because we route them through a module-global OpenBBTelemetrySink.
#
# Post-E0.4 the compiler NEVER calls these free functions — it goes through
# the injected sink. The compiler and the module-global sink are therefore
# fully decoupled; the module-global exists ONLY for back-compat with
# out-of-compiler callers.
#
# WARNING: Do NOT add new callers of ``record_unsupported_*`` (the writers)
# outside of tests that are explicitly documenting the back-compat surface.
# The free readers stay useful for that surface only; production observers
# should read counts off a per-request :class:`OpenBBTelemetrySink` via
# ``compile_pine(telemetry=<sink>)`` and either surface them on
# ``OBBject.extra["pine_telemetry"]`` (router path) or accept a sink
# directly (test path). A follow-up bead will delete the free writers
# once ``tests/unit/test_error_model.py`` migrates to the injected form.


_DEFAULT_SINK = OpenBBTelemetrySink()  # noqa: E305 — deliberate module global; see WARNING above


def record_unsupported_builtin(name: str) -> None:
    """Increment the module-global sink's builtin counter.

    Back-compat shim — the compiler no longer calls this. Kept for tests
    and operator tooling that hit the free-function surface directly.
    """
    _DEFAULT_SINK.record_unsupported_builtin(name)


def record_unsupported_feature(name: str) -> None:
    """Increment the module-global sink's feature counter.

    Back-compat shim — the compiler no longer calls this.
    """
    _DEFAULT_SINK.record_unsupported_feature(name)


def get_unsupported_builtin_counts() -> dict[str, int]:
    """Return a copy of the module-global sink's builtin counter map."""
    return _DEFAULT_SINK.get_unsupported_builtin_counts()


def get_unsupported_feature_counts() -> dict[str, int]:
    """Return a copy of the module-global sink's feature counter map."""
    return _DEFAULT_SINK.get_unsupported_feature_counts()


def reset_metrics() -> None:
    """Zero both counter maps on the module-global sink.

    Exposed for test isolation. Does NOT reset per-request sinks used by
    the routers — those are freshly instantiated per call.
    """
    _DEFAULT_SINK.reset()


__all__ = [
    "TelemetrySink",
    "OpenBBTelemetrySink",
    "record_unsupported_builtin",
    "record_unsupported_feature",
    "get_unsupported_builtin_counts",
    "get_unsupported_feature_counts",
    "reset_metrics",
]
