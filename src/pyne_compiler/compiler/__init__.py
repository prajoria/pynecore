"""The openbb-pine compiler — D1 territory.

This package houses the lexer (C1), parser/IR builder (C2), type checker (C3),
IR dataclasses (C4), codegen (C5), compile cache (C6), and v5→v6 auto-migration
shim (C7).

Public surfaces
---------------

* :func:`compile_pine` — high-level facade: ``str source -> CompiledModule``.
  Detects the ``//@version=`` pragma, applies :func:`migrate_v5_to_v6` if
  the source is v5, tokenises, parses, type-checks (C3), then emits Python
  source via codegen (C5). Wraps the pipeline with the C6 compile cache —
  cache probe → hit? return; else run pipeline → cache_write → return.
  Most callers (REST router D3 §4.6, CLI ``openbb-pine run``,
  ``obb.pine.run``) want this. Returns the C5 :class:`CompiledModule`
  carrying ``source``, ``sha`` (the C6 cache key), ``builtins_used``,
  ``security_contexts``, ``cache_status`` (``"hit"`` / ``"miss"`` /
  ``"bypass"``).
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

Cache (C6 — bead 0e9.5.6) integration
-------------------------------------

The facade wraps the compile pipeline with a content-addressed cache
under ``~/.openbb/pine_cache/`` (see :mod:`compile_cache`):

1. Compute the cache key from the RAW source (before migration) so that
   a v5 script and the equivalent hand-migrated v6 do NOT share a slot —
   they emit different Python.
2. Probe the cache: on hit, return the cached :class:`CompiledModule`
   with ``cache_status="hit"`` (skip the entire compile pipeline).
3. On miss, run lex → migrate? → parse → check → emit. Overwrite the
   :func:`emit`-computed placeholder sha with the real cache key and set
   ``cache_status="miss"``. Write through the cache.
4. When ``use_cache=False``, skip the probe and the write. The returned
   module carries ``cache_status="bypass"``. This is what the router
   uses when a caller explicitly opts out.

The cache key is a 256-bit BLAKE2b hash of ``source ‖ params ‖
compiler_version ‖ pine_version`` (D1 §6.1). Corruption / stale entries
degrade to a miss, never break compilation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyne_compiler import __version__
from pyne_compiler.compiler import ir
from pyne_compiler.compiler.codegen import emit
from pyne_compiler.compiler.compile_cache import (
    DEFAULT_CACHE_DIR,
    cache_read,
    cache_write,
    make_cache_key,
)
from pyne_compiler.compiler.lexer import Token, tokenize
from pyne_compiler.compiler.parser import parse
from pyne_compiler.compiler.type_checker import check
from pyne_compiler.compiler.types import CompiledModule
from pyne_compiler.compiler.v5_migration import (
    V5Rewrite,
    V5_REWRITES,
    detect_pine_version,
    migrate_v5_to_v6,
)

if TYPE_CHECKING:  # pragma: no cover — imports for typing only
    # Post-9bh: import target is pyne_compiler.telemetry (extraction
    # complete). Kept behind TYPE_CHECKING so the compiler never actually
    # imports telemetry at runtime — the E0.4 injection contract (see
    # plan Task E0.4, Step 5 and design doc §6.E0.4; the telemetry
    # module docstring recaps it).
    from pyne_compiler.telemetry import TelemetrySink

__all__ = [
    "Token",
    "tokenize",
    "parse",
    "check",
    "emit",
    "compile_pine",
    "compile_pine_to_program",
    "CompiledModule",
    "DEFAULT_CACHE_DIR",
    "V5Rewrite",
    "V5_REWRITES",
    "detect_pine_version",
    "migrate_v5_to_v6",
]


def _detect_and_migrate(
    source: str,
    target_version: int,
    *,
    telemetry: "TelemetrySink | None" = None,
) -> tuple[str, int]:
    """Shared detect-and-migrate prelude for both facades.

    Returns ``(possibly_migrated_source, final_version)``. Raises
    :class:`PineUnsupportedFeatureError` for unsupported version pairs.
    Threads the injected ``telemetry`` sink (E0.4) into
    :func:`detect_pine_version` and :func:`migrate_v5_to_v6` so PF001 /
    PF002 / PF003 raise paths record on the caller's sink.
    """
    if target_version not in (5, 6):
        from pyne_compiler.errors.base import PineUnsupportedFeatureError

        if telemetry is not None:
            telemetry.record_unsupported_feature("PF002")
        raise PineUnsupportedFeatureError(
            message=(
                f"PF002 compile_pine: target_version={target_version!r} is "
                "not supported. The compiler targets Pine v6; v5 sources "
                "are migrated to v6 transparently. See PRD §11 for the "
                "forward-compatibility plan."
            )
        )

    detected = detect_pine_version(source, telemetry=telemetry)

    # Apply migration when source < target. v6→v5 down-migration is not
    # supported (PRD §3.3 non-goal); the call still passes through so the
    # parser surfaces the inevitable v5-grammar miss with a normal
    # PineSyntaxError if it ever happens.
    if detected == 5 and target_version == 6:
        source, _rewrites_log = migrate_v5_to_v6(source, telemetry=telemetry)
        # The log is dropped here — the REST `/pine/compile` endpoint
        # (D3 §4.6) re-runs migrate_v5_to_v6 directly when it wants to
        # surface the log to the user. Keeping compile_pine's return
        # type pure (CompiledModule, not (CompiledModule, log)) preserves
        # the contract CLI + obb.pine.run share with the facade.
    elif detected != target_version:
        # Defensive: detect_pine_version already raises on PF001/PF002;
        # if we reach here it's the supported-pair-mismatch case
        # (e.g. detected=6 but target=5, which we don't support).
        from pyne_compiler.errors.base import PineUnsupportedFeatureError

        if telemetry is not None:
            telemetry.record_unsupported_feature("PF002")
        raise PineUnsupportedFeatureError(
            message=(
                f"PF002 compile_pine: source is v{detected}, requested "
                f"target_version={target_version}. The only supported "
                "migration direction is v5→v6."
            )
        )

    return source, target_version


def compile_pine_to_program(
    source: str,
    *,
    target_version: int = 6,
    type_check: bool = True,
    telemetry: "TelemetrySink | None" = None,
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
    telemetry
        Optional :class:`TelemetrySink` (E0.4). Threaded through the
        detect-and-migrate prelude and (when ``type_check=True``) into
        C3 so every unsupported-* raise records on the caller's sink
        before propagating. ``None`` = no sink; raises still fire.

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
    migrated, version = _detect_and_migrate(
        source, target_version, telemetry=telemetry
    )
    tokens = tokenize(migrated)
    program = parse(tokens, pine_version=version)
    if not type_check:
        return program
    type_check_result = check(program, pine_version=version, telemetry=telemetry)
    return type_check_result.program


def compile_pine(
    source: str,
    *,
    target_version: int = 6,
    params: dict[str, Any] | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    telemetry: "TelemetrySink | None" = None,
) -> CompiledModule:
    """Compile a Pine source string into a :class:`CompiledModule` (D1 §1.5).

    The high-level facade. Hides five concerns from callers:

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
    5. **Cache (C6).** Content-addressed lookup under
       ``~/.openbb/pine_cache/``. On hit, skips the entire compile
       pipeline and returns the cached module with ``cache_status="hit"``.
       On miss, runs the pipeline and writes the result through with
       ``cache_status="miss"``. When ``use_cache=False``, both probe and
       write are skipped; the returned module carries
       ``cache_status="bypass"``.

    Parameters
    ----------
    source
        Pine v5 or v6 source text.
    target_version
        5 or 6. Defaults to 6. v5 sources are migrated transparently.
    params
        Optional user-supplied parameter dict, carved into the cache key
        so different runs of the same source with different params get
        distinct cache slots. Phase-1 callers pass ``None``; higher
        surfaces (D3's REST router) may thread through actual params.
    use_cache
        When True (default), enable the C6 compile cache. When False,
        skip both the probe and the write — useful for CI runs that
        want a deterministic cold path.
    cache_dir
        Optional override for the on-disk cache directory. When None
        (default), uses :data:`DEFAULT_CACHE_DIR`
        (``~/.openbb/pine_cache/``). Tests + operator overrides pass a
        tmp_path here.
    telemetry
        Optional :class:`TelemetrySink` (E0.4) — the openbb-fork router
        instantiates an :class:`~openbb_pine.telemetry.OpenBBTelemetrySink`
        per request and passes it here. The sink is threaded through
        every stage that raises ``PineUnsupported*Error``
        (:func:`detect_pine_version`, :func:`migrate_v5_to_v6`,
        :func:`check`, :func:`emit`) so per-request counts are captured
        on the sink and can be surfaced in the router's response
        envelope. ``None`` = telemetry disabled; the raises still fire
        but no counter increments. Cache hits skip the entire pipeline
        so no counter fires either — this matches the pre-E0.4 behavior
        (a cached script was already validated on the compile that
        populated the cache slot).

    Returns
    -------
    CompiledModule
        Carrying ``source`` (the @pyne-headed Python text PyneCore will
        import-and-run), ``sha`` (the C6 cache key when ``use_cache=True``,
        otherwise the codegen placeholder), ``builtins_used`` (the
        C3-populated set), ``security_contexts`` (None in Phase 1),
        ``cache_status`` (``"hit"`` / ``"miss"`` / ``"bypass"``).

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

    Cache key uses the RAW source (before migration) so that v5 and v6
    forms of the same script don't share a cache slot — they'd emit
    different Python. If a caller compiled the v5 form first, then later
    compiled the equivalent v6 form, they get distinct cached entries.
    """
    # C6 cache probe — first, so hits skip the entire pipeline.
    key: str | None = None
    if use_cache:
        # Compute the key from the RAW source (before migration). D1 §6.1.
        cd = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        key = make_cache_key(
            source,
            params=params,
            compiler_version=__version__,
            pine_version=target_version,
        )
        hit = cache_read(key, cache_dir=cd)
        if hit is not None:
            return hit  # cache_status already "hit"

    # Cache miss (or bypass) — full pipeline.
    migrated, version = _detect_and_migrate(
        source, target_version, telemetry=telemetry
    )
    tokens = tokenize(migrated)
    program = parse(tokens, pine_version=version)
    type_check_result = check(program, pine_version=version, telemetry=telemetry)
    compiled = emit(
        type_check_result.program,
        builtins_used=type_check_result.builtins_used,
        security_contexts=type_check_result.security_contexts,
        pine_version=version,
        compiler_version=__version__,
        telemetry=telemetry,
    )

    # Overwrite the emit()-produced placeholder sha + cache_status with the
    # real cache-key / miss status. Use ``dataclasses.replace`` since
    # ``CompiledModule`` is ``frozen=True``. When use_cache=False the sha
    # stays as emit's placeholder and cache_status stays "bypass" — the
    # facade's contract is that the sha you receive is always the value
    # the cache would use if it were writing that entry (either the real
    # key, or the placeholder when the cache is off).
    if use_cache:
        assert key is not None  # narrowing for type checker
        compiled = replace(compiled, sha=key, cache_status="miss")
        cache_write(compiled, cache_dir=cd)
    # When use_cache=False, keep emit()'s placeholder shape (sha="" +
    # cache_status="bypass") — caller opted out of the cache contract.

    return compiled
