"""Restricted exec namespace for user-compiled ``@pyne`` modules.

Per D2 section 10.1 + PRD section 5.2 T3.

Defense-in-depth with D1's T1 codegen allowlist: T1 prevents the compiler
from *emitting* dangerous imports / calls; this module prevents the compiled
module from *executing* dangerous builtins should T1 ever leak. Both have
to fail for an escape.

``_ALLOWED_BUILTINS`` is a deliberate frozenset of arithmetic / iteration /
type-introspection primitives. Everything outside that set -- in particular
``__import__``, ``open``, ``exec``, ``eval``, ``compile``, ``input``,
``globals``, ``locals``, ``vars``, ``setattr``, ``delattr``, ``dir``,
``exit``, ``quit``, ``help``, ``breakpoint``, ``getattr``, ``hasattr``,
``memoryview``, ``bytes``, ``bytearray`` -- is unreachable from user code
because it does not exist in the namespace's ``__builtins__`` dict at all.
"""

from __future__ import annotations

import builtins
from typing import Any

# Arithmetic / iteration / type-introspection only. NO I/O, NO eval, NO import.
# Explicitly absent (and tested as such in ``tests/unit/test_restricted.py``):
#   __import__, open, exec, eval, compile, input,
#   globals, locals, vars, setattr, delattr, dir,
#   exit, quit, help, breakpoint,
#   getattr, hasattr, id,
#   memoryview, bytes, bytearray.
_ALLOWED_BUILTINS: frozenset[str] = frozenset({
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "int", "isinstance", "issubclass", "len",
    "list", "map", "max", "min", "object", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
})


def build_restricted_namespace(compiled_module_path: str) -> dict[str, Any]:
    """Build a fresh exec namespace for one compiled ``@pyne`` module.

    The caller owns the returned dict and may stash module-level globals in
    it after the ``exec``. The dict is NOT shared between calls -- each
    invocation returns a new mapping so one script cannot leak state into
    another via the namespace.

    The installed ``__builtins__`` is a plain dict (not the
    ``builtins`` module) containing only the names in ``_ALLOWED_BUILTINS``.
    Python's ``exec()`` honors whatever ``__builtins__`` it finds in the
    globals dict, so a name lookup for e.g. ``open`` raises ``NameError``,
    and ``import os`` raises ``ImportError`` (because ``__import__`` is
    not resolvable).

    Parameters
    ----------
    compiled_module_path
        Filesystem path to the compiled module file. Stored as ``__file__``
        in the namespace so tracebacks show the user a meaningful location.
    """
    restricted_builtins: dict[str, Any] = {
        name: getattr(builtins, name) for name in _ALLOWED_BUILTINS
    }
    namespace: dict[str, Any] = {
        "__name__": "pine_user_script",
        "__file__": compiled_module_path,
        "__builtins__": restricted_builtins,
    }
    return namespace


__all__ = ["_ALLOWED_BUILTINS", "build_restricted_namespace"]
