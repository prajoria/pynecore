"""E0.3 gate: the pre-split ``executor`` module remains a working
deprecation shim (bd-9zb).

Covers three concerns:

1. Re-exports the pre-split public surface (``ProviderOrData``,
   ``run_compiled``) plus the ``_resolve_data_source`` helper that at
   least one internal caller (the shell's own test module) imports.
2. The re-exports are the same object as the shell's originals (``is``
   identity) — so patch/mock semantics against ``executor.<name>`` still
   route to the shell's implementation via reference, not a copy.
3. Importing the module emits a :class:`DeprecationWarning` so callers
   are nudged to migrate to ``executor_shell`` / ``executor_core``.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def _reload_executor():
    """Reload the executor shim so the module-level DeprecationWarning
    fires INSIDE the test's ``catch_warnings`` block. Otherwise the
    first import (during collection) already consumed it."""
    for name in list(sys.modules):
        if name == "openbb_pine.runtime.executor":
            del sys.modules[name]
    return importlib.import_module("openbb_pine.runtime.executor")


def test_shim_reexports_run_compiled():
    from openbb_pine.runtime import executor, executor_shell

    assert executor.run_compiled is executor_shell.run_compiled


def test_shim_reexports_provider_or_data():
    from openbb_pine.runtime import executor, executor_shell

    assert executor.ProviderOrData is executor_shell.ProviderOrData


def test_shim_reexports_resolve_data_source():
    """Internal helper import path also works via the shim."""
    from openbb_pine.runtime import executor, executor_shell

    assert executor._resolve_data_source is executor_shell._resolve_data_source


def test_shim_reexports_collect_results():
    """The pre-split ``_collect_results`` helper (now in executor_core)
    is still importable via the shim so callers doing white-box tests
    against ``executor._collect_results`` keep working across the
    one-release window."""
    from openbb_pine.runtime import executor, executor_core

    assert executor._collect_results is executor_core._collect_results


def test_shim_emits_deprecation_warning_on_import():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reload_executor()

    dep_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert dep_warnings, (
        "Expected DeprecationWarning when importing "
        "openbb_pine.runtime.executor (post-E0.3 shim). Got: "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    message = str(dep_warnings[0].message)
    # Message must name the successor modules so the migration hint is
    # actionable without a doc lookup.
    assert "executor_shell" in message
    assert "executor_core" in message


def test_shim_dunder_all_matches_pre_split_public_surface():
    """The shim exposes the same ``__all__`` the pre-split module did
    (``ProviderOrData`` + ``run_compiled``). Internal helpers like
    ``_resolve_data_source`` are re-exported for import but are not
    part of the wildcard-imported public surface — that was true of the
    pre-split module too (its ``__all__`` did not list private names)."""
    from openbb_pine.runtime import executor

    assert set(executor.__all__) == {"ProviderOrData", "run_compiled"}
