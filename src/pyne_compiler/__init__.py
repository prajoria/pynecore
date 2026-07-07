"""Placeholder package for the Pine-to-Python clean-room compiler + runtime.

This is an empty skeleton created ahead of the E2 ``git filter-repo`` step
that will import the real compiler + runtime sources from
``prajoria/openbb-fork`` (task epic ``OpenBBTechnical-rbf``). Once E2 lands,
this file is replaced with the real package init and ``__version__`` gets
promoted to a real release number.

Nothing in this module imports ``pynecore`` — it is deliberately empty so
setuptools can discover it alongside ``src/pynecore/`` without dragging in
any compiler internals that do not exist yet.
"""

__version__ = "0.1.0-pre-extraction"
