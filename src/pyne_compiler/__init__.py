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

#: Package version. PEP 440-clean so ``packaging.version.Version(__version__)``
#: succeeds — design §6.E3 step 7 embeds this string in ``CompiledModule.
#: compiler_version`` and the compile-cache-key hash, and downstream code
#: that parses it must not raise ``InvalidVersion``. The ``+pre.extraction``
#: local-version segment (dots, not dashes) preserves the human-readable
#: intent while satisfying PEP 440. Promoted to a real release number at E2.
__version__ = "0.1.0+pre.extraction"
