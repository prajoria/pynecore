"""Placeholder-package installability tests.

Verifies that the ``pyne_compiler`` skeleton is discoverable by setuptools
and coexists with ``pynecore`` in the same distribution
(``pynesys-pynecore``). Deleted / replaced by the real compiler test suite
after the E2 ``git filter-repo`` step.

Function names use pynecore's project convention
(``python_functions = __test_*__`` in ``pytest.ini``) so they are picked up
by the existing test runner without any config change.
"""


def __test_import_pyne_compiler__():
    """``import pyne_compiler`` succeeds from an installed distribution."""
    import pyne_compiler  # noqa: F401 — import is the assertion


def __test_version_string__():
    """``pyne_compiler.__version__`` is a non-empty ``str``."""
    import pyne_compiler

    assert isinstance(pyne_compiler.__version__, str)
    assert pyne_compiler.__version__ != ""


def __test_pynecore_also_importable__():
    """``pyne_compiler`` and ``pynecore`` coexist in the same install."""
    import pynecore  # noqa: F401 — import is the assertion
    import pyne_compiler  # noqa: F401 — import is the assertion
