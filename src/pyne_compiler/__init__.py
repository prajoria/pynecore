"""Pine-to-Python clean-room compiler + runtime.

Merged in via E2.3 from the extraction scratch clone
(``pre-openbb-extraction-2026-07-09`` rollback anchor). See design
§6.E3 step 7 for how ``__version__`` is embedded into
``CompiledModule.compiler_version`` and the compile-cache-key hash.
"""

#: Package version. PEP 440-clean so ``packaging.version.Version(__version__)``
#: succeeds — parsed downstream in the compile-cache key.
__version__ = "0.1.0"
