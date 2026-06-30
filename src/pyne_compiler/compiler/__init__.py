"""The openbb-pine compiler — D1 territory.

This package houses the lexer (C1), parser/IR builder (C2), type checker (C3),
IR dataclasses (C4), codegen (C5), compile cache (C6), and v5→v6 auto-migration
shim (C7).

Public surfaces
---------------

* :func:`compile_pine` — high-level facade: ``str source -> CompiledModule``.
  Detects the ``//@version=`` pragma, applies :func:`migrate_v5_to_v6` if
  the source is v5, tokenises, parses, type-checks (C3), then emits Python
  source via codegen (C5). Most callers (REST router D3 §4.6, CLI
  ``openbb-pine run``, ``obb.pine.run``) want this. Returns the C5
  :class:`CompiledModule` carrying ``source``, ``sha``, ``builtins_used``,
  ``security_contexts``, ``cache_status``.
* :func:`compile_pine_to_program` — same pipeline as :func:`compile_pine`
  but stops one stage earlier (after C3's type checker) and returns the
  typed IR :class:`Program`. Intended for tests + tooling that need the
  IR without emitting Python (migration log + diagnostic inspection).
* :func:`tokenize`, :func:`parse`, :func:`migrate_v5_to_v6`,
  :func:`detect_pine_version`, :func:`emit` — re-exported low-level
  surfaces for tests + advanced usage. The compile pipeline is layered
  so each stage stays independently testable.

D1 §1.5 says the compile pipeline is ``source → lexer → migration? →
parser → type-checker → codegen → cache``. The shim slot lives between
the lexer's pragma detection and the parser's grammar dispatch — but
because the v5→v6 rewrites are source-level (PRD §3.2 mentions only
syntactic rewrites), we apply the shim BEFORE the lexer so the lexer
sees a v6 source. The lexer is dialect-agnostic anyway; running it
before vs. after the rewrites is observably identical.

Higher-level entry points (REST endpoints, OBBject construction) live in
``openbb_pine.pine_router``; this package stays pure compiler.

Cache (C6) — out of scope for the C5 bead
-----------------------------------------

:func:`compile_pine` always sets ``CompiledModule.cache_status="bypass"``
and a placeholder blake2b hash. C6 (separate bead) will:

1. Compute the real cache key per D1 §6 (blake2b over source ‖ params ‖
   compiler_version ‖ pine_version).
2. Lookup the cache before re-running the pipeline; set
   ``cache_status="hit"`` and return the cached module on hit.
3. Write the emitted source through to the cache on miss; set
   ``cache_status="miss"``.

The seam is the public ``CompiledModule`` contract; C6 wraps without
changing the facade signature.
"""

from __future__ import annotations

from openbb_pine import __version__
from openbb_pine.compiler import ir
from openbb_pine.compiler.codegen import emit
from openbb_pine.compiler.lexer import Token, tokenize
from openbb_pine.compiler.parser import parse
from openbb_pine.compiler.type_checker import check
from openbb_pine.compiler.types import CompiledModule
from openbb_pine.compiler.v5_migration import (
    V5Rewrite,
    V5_REWRITES,
    detect_pine_version,
    migrate_v5_to_v6,
)

__all__ = [
    "Token",
    "tokenize",
    "parse",
    "check",
    "emit",
    "compile_pine",
    "compile_pine_to_program",
    "CompiledModule",
    "V5Rewrite",
    "V5_REWRITES",
    "detect_pine_version",
    "migrate_v5_to_v6",
]


def _detect_and_migrate(source: str, target_version: int) -> tuple[str, int]:
    """Shared detect-and-migrate prelude for both facades.

    Returns ``(possibly_migrated_source, final_version)``. Raises
    :class:`PineUnsupportedFeatureError` for unsupported version pairs.
    """
    if target_version not in (5, 6):
        from openbb_pine.errors import PineUnsupportedFeatureError

        raise PineUnsupportedFeatureError(
            message=(
                f"PF002 compile_pine: target_version={target_version!r} is "
                "not supported. The compiler targets Pine v6; v5 sources "
                "are migrated to v6 transparently. See PRD §11 for the "
                "forward-compatibility plan."
            )
        )

    detected = detect_pine_version(source)

    # Apply migration when source < target. v6→v5 down-migration is not
    # supported (PRD §3.3 non-goal); the call still passes through so the
    # parser surfaces the inevitable v5-grammar miss with a normal
    # PineSyntaxError if it ever happens.
    if detected == 5 and target_version == 6:
        source, _rewrites_log = migrate_v5_to_v6(source)
        # The log is dropped here — the REST `/pine/compile` endpoint
        # (D3 §4.6) re-runs migrate_v5_to_v6 directly when it wants to
        # surface the log to the user. Keeping compile_pine's return
        # type pure (CompiledModule, not (CompiledModule, log)) preserves
        # the contract CLI + obb.pine.run share with the facade.
    elif detected != target_version:
        # Defensive: detect_pine_version already raises on PF001/PF002;
        # if we reach here it's the supported-pair-mismatch case
        # (e.g. detected=6 but target=5, which we don't support).
        from openbb_pine.errors import PineUnsupportedFeatureError

        raise PineUnsupportedFeatureError(
            message=(
                f"PF002 compile_pine: source is v{detected}, requested "
                f"target_version={target_version}. The only supported "
                "migration direction is v5→v6."
            )
        )

    return source, target_version


def compile_pine_to_program(
    source: str, *, target_version: int = 6, type_check: bool = True
) -> "ir.Program":
    """Parse [+ type-check] ``source``, returning the IR :class:`Program`.

    Lower-level facade than :func:`compile_pine`: runs the pipeline through
    C3 (type checker) and stops, returning the IR for callers that want to
    inspect / introspect / re-emit. Used by:

    * The C7 migration tests (which care about the post-migration IR
      structure, not the emitted Python).
    * Tooling that needs the IR shape (e.g. a future ``openbb pine ast``
      sub-command, a debugger, a doc-generator).

    The :func:`compile_pine` facade composes this with C5 emit to produce
    the :class:`CompiledModule` end consumers (REST, CLI, OBBject) actually
    receive.

    Parameters
    ----------
    source
        Pine v5 or v6 source text.
    target_version
        5 or 6. Defaults to 6. v5 sources are migrated transparently.
    type_check
        When True (default), runs C3 over the parsed program and returns
        the typed IR. When False, stops at the parser's raw IR — useful
        for tests / tooling that compare IR shape and don't want to
        re-litigate a (known-incomplete) C3 forward-declaration limitation.
        The ``False`` path is intentionally narrow: production callers
        should use :func:`compile_pine` which always type-checks.

    Raises
    ------
    PineUnsupportedFeatureError
        Source uses an unsupported version (PF001 / PF002) or a v5
        construct no rewrite handles (PF003).
    PineSyntaxError
        Source fails to tokenise or parse under the target grammar.
    PineTypeError, PineUnsupportedBuiltinError
        C3 type-checker rejections (only when ``type_check=True``).
    """
    migrated, version = _detect_and_migrate(source, target_version)
    tokens = tokenize(migrated)
    program = parse(tokens, pine_version=version)
    if not type_check:
        return program
    type_check_result = check(program, pine_version=version)
    return type_check_result.program


def compile_pine(source: str, *, target_version: int = 6) -> CompiledModule:
    """Compile a Pine source string into a :class:`CompiledModule` (D1 §1.5).

    The high-level facade. Hides four concerns from callers:

    1. **Version detection.** Reads ``//@version=N`` via
       :func:`detect_pine_version`. Defaults to v6 when no pragma is
       found (per PRD §2.1 + TradingView editor's behavior).
    2. **v5→v6 migration.** When ``detected != target_version`` and the
       transition is the supported v5→v6, runs :func:`migrate_v5_to_v6`
       transparently. Other ``(detected, target)`` pairs raise — there
       is no v6→v5 down-migration and the v4-or-earlier branch is a PRD
       §3.3 non-goal.
    3. **Tokenize + parse + type-check.** Lex / parse / C3.
    4. **Codegen (C5).** Walks the typed IR, builds an ``ast.Module``,
       runs the closed-allowlist gate (D1 §3.3), unparses to the
       canonical Python source, wraps in a :class:`CompiledModule`.

    Returns
    -------
    CompiledModule
        Carrying ``source`` (the @pyne-headed Python text PyneCore will
        import-and-run), ``sha`` (placeholder until C6 lands the real
        cache key), ``builtins_used`` (the C3-populated set),
        ``security_contexts`` (None in Phase 1),
        ``cache_status="bypass"`` (C6's seam).

    Raises
    ------
    PineUnsupportedFeatureError
        Source uses an unsupported version (PF001 / PF002), a v5 construct
        no rewrite handles (PF003), a typed decl in body (PF002), a
        ``strategy()`` directive (PF010 — deferred to M2), or a
        ``library()`` directive (PF011).
    PineSyntaxError
        Source fails to tokenise or parse under the target grammar.
    PineTypeError, PineUnsupportedBuiltinError
        C3 type-checker rejections.
    PineCodegenError
        ALWAYS a compiler bug per D1 §3.5 — fires if the allowlist gate
        catches a node / module / name the emitter shouldn't have produced.

    Notes
    -----
    The ``target_version`` kwarg is accepted but currently only 5 / 6 are
    valid. The parameter exists so the call-site contract is forward-
    compatible with the v6→v7 migration when TradingView ships a v7
    (PRD §11 "v7 reopens this").
    """
    migrated, version = _detect_and_migrate(source, target_version)
    tokens = tokenize(migrated)
    program = parse(tokens, pine_version=version)
    type_check_result = check(program, pine_version=version)
    compiled = emit(
        type_check_result.program,
        builtins_used=type_check_result.builtins_used,
        security_contexts=type_check_result.security_contexts,
        pine_version=version,
        compiler_version=__version__,
    )
    return compiled
