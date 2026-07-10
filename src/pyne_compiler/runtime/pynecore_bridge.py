"""Sys.path bridge for the submodule-vendored pynecore deployment.

When openbb-fork uses ``third_party/pynecore/`` as a git submodule (not a
pip-installed package), pynecore is not on sys.path by default. This
module prepends ``<submodule>/src`` so ``import pynecore`` resolves.

Idempotent: safe to call multiple times.
No-op when pynecore is already importable (the happy path once pynecore
is pip-installed alongside pyne_compiler).

Post-E2 this file moves to ``src/pyne_compiler/runtime/pynecore_bridge.py``.
The E3 refactor makes ``openbb_pine.__init__`` delegate here instead of
owning the sys.path insertion directly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# Upper bound on the parent-walk in :func:`_submodule_src_dir`. Chosen to
# cover the intended repo layout (runtime/pynecore_bridge.py -> runtime/ ->
# openbb_pine/ -> pine/ -> extensions/ -> openbb_platform/ -> repo-root,
# i.e. depth 6) with a small margin. An unbounded walk would keep climbing
# to the filesystem root and could silently bind to an unrelated
# ``third_party/pynecore/src`` living in some ancestor directory (a
# ``pip install --target ~/vendor/`` layout, a wrapper monorepo, an
# unrelated colocated checkout). Six hops is enough for every layout we
# support and stops well before /.
_PARENT_WALK_MAX_DEPTH = 6


def is_pynecore_installed() -> bool:
    """Return True when ``import pynecore`` would succeed without our path manipulation."""
    return importlib.util.find_spec("pynecore") is not None


def _submodule_src_dir() -> Path:
    """Return the path to ``third_party/pynecore/src`` relative to this file's repo root.

    Walks at most :data:`_PARENT_WALK_MAX_DEPTH` parents of this file looking
    for a ``third_party/pynecore/src`` directory that contains a real
    ``pynecore/__init__.py`` — the presence check guards against binding to
    an unrelated repo's ``third_party/pynecore/src`` in an ancestor
    directory (see the module comment on ``_PARENT_WALK_MAX_DEPTH``).

    Raises
    ------
    ImportError
        When no plausible pynecore submodule tree is found within the
        bounded walk. The message covers both PRD §4.4 install paths
        (``pip install pynesys-pynecore`` and
        ``git submodule update --init --recursive third_party/pynecore``)
        so users hit an actionable hint regardless of which one they picked.
    """
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:_PARENT_WALK_MAX_DEPTH]:
        candidate = parent / "third_party" / "pynecore" / "src"
        # Both checks are required: the dir may exist as an empty stub
        # (submodule not initialized) or belong to an unrelated project;
        # the __init__.py presence pins us to a real pynecore package tree.
        if candidate.is_dir() and (candidate / "pynecore" / "__init__.py").is_file():
            return candidate
    raise ImportError(
        "openbb-pine: PyneCore is neither installed (`pip install "
        "pynesys-pynecore`) nor present as a git submodule at "
        f"{here.parents[min(_PARENT_WALK_MAX_DEPTH - 1, len(here.parents) - 1)]}"
        "/third_party/pynecore/src. Initialize the submodule with "
        "`git submodule update --init --recursive third_party/pynecore` "
        "or install the PyPI package. See PRD section 4.4."
    )


def install_pynecore_path() -> None:
    """Prepend the pynecore submodule's ``src/`` to ``sys.path`` if pynecore isn't installed.

    Idempotent: repeated calls have no effect after the first successful insert.
    No-op when pynecore is already importable.

    Raises
    ------
    ImportError
        When pynecore is not importable AND no submodule tree can be
        located within the bounded parent-walk. See :func:`_submodule_src_dir`
        for the message shape — it preserves the actionable hint from the
        pre-refactor ``_install_pynecore_path`` so callers using
        ``except ImportError:`` around ``import openbb_pine`` keep working.
    """
    if is_pynecore_installed():
        return
    src_dir = str(_submodule_src_dir())
    if src_dir in sys.path:
        return  # already inserted, don't duplicate
    sys.path.insert(0, src_dir)


__all__ = ["install_pynecore_path", "is_pynecore_installed"]
