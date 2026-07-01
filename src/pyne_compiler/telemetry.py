"""D1 §3.5 + PRD §9.4 failure-mode telemetry counters for openbb-pine.

Lightweight in-process counters until a real Prometheus wire lands (the P4
observability bead). Same shape as :mod:`openbb_pine.runtime.fmp_retry`'s
``_fmp_unreachable_counters`` — dict[str, int] with a :func:`reset_metrics`
helper for test isolation.

Two counters:

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
appropriate ``record_*`` function BEFORE raising. Because counters are the
easy part and forgetting one silently degrades the L0.5 attribution
metric, the AST-walking enforcement test in
``tests/unit/test_error_model.py`` verifies the raise-sites still resolve
their codes; call-site coverage is verified by the integration test in
the same file.

# TODO(P4): wire these into real ``prometheus_client.Counter`` instances
# when the observability bead lands. Until then, ``get_*_counts()`` doubles
# as a test hook and a debug surface the ``openbb-pine doctor`` CLI can
# print.
"""

from __future__ import annotations


# --- Storage -----------------------------------------------------------------

_unsupported_builtin_counters: dict[str, int] = {}
"""``pine_unsupported_builtin_total{name=...}`` stand-in — keyed by the Pine
builtin qualified name (e.g. ``"ta.ichimoku"``)."""

_unsupported_feature_counters: dict[str, int] = {}
"""``pine_unsupported_feature_total{name=...}`` stand-in — keyed by the
``PF###`` feature code (or the feature label if no code)."""


# --- Recorders ---------------------------------------------------------------


def record_unsupported_builtin(name: str) -> None:
    """Increment the ``pine_unsupported_builtin_total{name}`` counter.

    Call this immediately BEFORE ``raise PineUnsupportedBuiltinError(name)``
    so the counter is accurate even when the raise unwinds through a
    ``try`` that swallows the exception.
    """
    _unsupported_builtin_counters[name] = (
        _unsupported_builtin_counters.get(name, 0) + 1
    )


def record_unsupported_feature(name: str) -> None:
    """Increment the ``pine_unsupported_feature_total{name}`` counter.

    ``name`` should be the ``PF###`` code (preferred) or the feature label
    when no code was assigned.
    """
    _unsupported_feature_counters[name] = (
        _unsupported_feature_counters.get(name, 0) + 1
    )


# --- Accessors ---------------------------------------------------------------


def get_unsupported_builtin_counts() -> dict[str, int]:
    """Return a copy of the builtin-unsupported counter map.

    A copy (not the live reference) so callers can mutate their view without
    corrupting counter state. Matches the sunk-cost invariant on
    ``runtime.fmp_retry._fmp_unreachable_counters``.
    """
    return dict(_unsupported_builtin_counters)


def get_unsupported_feature_counts() -> dict[str, int]:
    """Return a copy of the feature-unsupported counter map."""
    return dict(_unsupported_feature_counters)


def reset_metrics() -> None:
    """Zero both counter maps. Exposed for test isolation and for operators
    who want to clear state between runs of the doctor CLI."""
    _unsupported_builtin_counters.clear()
    _unsupported_feature_counters.clear()


__all__ = [
    "record_unsupported_builtin",
    "record_unsupported_feature",
    "get_unsupported_builtin_counts",
    "get_unsupported_feature_counts",
    "reset_metrics",
]
