"""E0.5 gate: pynecore_bridge module exists with install_pynecore_path +
is_pynecore_installed, and is idempotent (safe to call multiple times).

The dev venv has ``pynesys-pynecore`` pip-installed, so the natural
``install_pynecore_path()`` call is a no-op there. The tests below either
skip cleanly when a precondition doesn't hold (``pytest.skip``) or force
the vendored branch via ``monkeypatch`` — silent-pass ("body skipped,
report says PASSED") is treated as a coverage regression and avoided.
"""

from __future__ import annotations

import sys

import pytest


def test_bridge_module_exports_expected_functions() -> None:
    from openbb_pine.runtime import pynecore_bridge
    assert callable(pynecore_bridge.install_pynecore_path)
    assert callable(pynecore_bridge.is_pynecore_installed)


def test_bridge_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls insert the vendored src/ dir at most once.

    Forces the mutation branch via monkeypatch so the dedup guard
    (``if src_dir in sys.path: return``) is actually exercised even in a
    dev venv where ``pynesys-pynecore`` is pip-installed. Without the
    monkeypatch this test collapses into ``0 <= 1`` and a future refactor
    that breaks dedup would silently pass CI.
    """
    from openbb_pine.runtime import pynecore_bridge

    monkeypatch.setattr(pynecore_bridge, "is_pynecore_installed", lambda: False)
    before = list(sys.path)
    # Simulate a clean pre-bridge state: the bridge fires at openbb_pine
    # import time (in openbb_pine/__init__.py), so by the time this test
    # runs src_dir is already in sys.path. Remove it so the assertion
    # tests the actual prepend behavior, not a no-op from the dedup guard.
    src_dir = str(pynecore_bridge._submodule_src_dir())
    sys.path[:] = [p for p in before if p != src_dir]
    baseline = list(sys.path)
    try:
        pynecore_bridge.install_pynecore_path()
        pynecore_bridge.install_pynecore_path()
        pynecore_bridge.install_pynecore_path()

        # Exactly one — this is the dedup assertion, meaningful because we
        # forced the mutation branch (monkeypatch) AND ensured the initial
        # sys.path didn't already contain src_dir.
        assert sys.path.count(src_dir) == 1
        # Prepended at index 0 — pynecore-shadowing search order matters.
        # Now safe to assert directly because we ensured a clean baseline
        # (no pytest-injected src_dir competing for position 0).
        assert sys.path[0] == src_dir, (
            f"expected src_dir {src_dir!r} at sys.path[0], "
            f"found {sys.path[0]!r} (bridge failed to prepend)"
        )
        # Nothing else got inserted.
        extra = [p for p in sys.path if p not in baseline and p != src_dir]
        assert extra == [], f"bridge inserted unexpected paths: {extra}"
    finally:
        # Don't leak the insert to sibling tests. Restore ORIGINAL before,
        # not baseline (which had src_dir removed for the test).
        sys.path[:] = before


def test_bridge_is_noop_when_pynecore_already_installed() -> None:
    """When pynecore is already importable, install_pynecore_path() is a no-op.

    Skips explicitly (rather than silently passing) if the precondition
    ``is_pynecore_installed()`` does not hold — an honest coverage report
    is worth more than a green tick.
    """
    from openbb_pine.runtime import pynecore_bridge

    if not pynecore_bridge.is_pynecore_installed():
        pytest.skip("pynecore is not installed in this environment")

    before = list(sys.path)
    pynecore_bridge.install_pynecore_path()
    assert sys.path == before, (
        "bridge should be a no-op when pynecore is already installed"
    )


def test_submodule_src_dir_walk_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Guard against unbounded parent-walk silently binding to a stray tree.

    Constructs a fake filesystem where a ``third_party/pynecore/src/pynecore``
    package lives *beyond* the bounded walk (at depth 7 from the fake file)
    and asserts that ``_submodule_src_dir`` refuses to bind to it. Without
    the bound the old implementation would climb past six parents and
    silently attach an unrelated pynecore tree.
    """
    from openbb_pine.runtime import pynecore_bridge

    # Build a directory ladder deep enough that the plausible-pynecore tree
    # sits at depth > _PARENT_WALK_MAX_DEPTH from ``fake_file``.
    depth = pynecore_bridge._PARENT_WALK_MAX_DEPTH + 1
    root = tmp_path
    for i in range(depth):
        root = root / f"d{i}"
    root.mkdir(parents=True)
    fake_file = root / "pynecore_bridge.py"
    fake_file.write_text("# stub\n", encoding="utf-8")

    # Place the plausible tree at the outermost ancestor (tmp_path itself),
    # which is depth+1 parents above fake_file — past the bound.
    stray = tmp_path / "third_party" / "pynecore" / "src" / "pynecore"
    stray.mkdir(parents=True)
    (stray / "__init__.py").write_text("# stray\n", encoding="utf-8")

    # Monkeypatch Path(__file__) inside the module by replacing the module's
    # __file__ and calling _submodule_src_dir — but the function reads
    # __file__ at call time, so patch it via the module attribute.
    monkeypatch.setattr(pynecore_bridge, "__file__", str(fake_file))
    with pytest.raises(ImportError, match="pynesys-pynecore"):
        pynecore_bridge._submodule_src_dir()


def test_submodule_src_dir_rejects_empty_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A ``third_party/pynecore/src`` dir without ``pynecore/__init__.py`` is not accepted.

    Guards against binding to an uninitialized submodule stub or an
    unrelated project's ``third_party/pynecore/src`` directory that
    happens to sit in an ancestor path.
    """
    from openbb_pine.runtime import pynecore_bridge

    # File at depth 1 below tmp_path; stub at depth 0 (within the bound).
    root = tmp_path / "d0"
    root.mkdir()
    fake_file = root / "pynecore_bridge.py"
    fake_file.write_text("# stub\n", encoding="utf-8")

    stub = tmp_path / "third_party" / "pynecore" / "src"
    stub.mkdir(parents=True)  # dir exists but no pynecore/__init__.py inside

    monkeypatch.setattr(pynecore_bridge, "__file__", str(fake_file))
    with pytest.raises(ImportError, match="pynesys-pynecore"):
        pynecore_bridge._submodule_src_dir()
