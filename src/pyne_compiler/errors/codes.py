"""D1 §5.2 error-code namespace — three-letter prefix + three-digit code.

**Stable across releases; append-only, never renumbered.** Adding a new code
means adding a row to :data:`ERROR_CODES` and (usually) documenting it in
``docs/designs/openbb-pine/error-codes.md``.

Prefix ownership:

| Prefix   | Class                          | Range     | Owner              |
|----------|--------------------------------|-----------|--------------------|
| ``PS``   | PineSyntaxError                | PS001+    | Lexer / parser     |
| ``PT``   | PineTypeError                  | PT001-008 | Type checker (D1 §4.4) |
| ``PU``   | PineUnsupportedBuiltinError    | PU001+    | Stdlib bridge      |
| ``PF``   | PineUnsupportedFeatureError    | PF001+    | Feature owner      |
| ``CG``   | PineCodegenError               | CG001-006 | Codegen (D1 §3.2)  |
| ``IC``   | PineInternalCompilerError      | IC001+    | Compiler           |
| ``CACHE``| PineCacheError                 | CACHE001+ | Cache              |
| ``RT``   | PineRuntimeError               | RT001+    | Runtime            |
| ``SEC``  | PineSecurityError              | SEC001+   | Security           |
| ``PROV`` | PineProviderError family       | PROV001+  | Data provider      |

Each code is a single :class:`ErrorCodeSpec` entry so :func:`grep` finds it
in one line. When you add a new code, run
``pytest tests/unit/test_error_model.py::TestErrorCodeEnforcement`` — the
AST-walking test scans every ``raise Pine*Error(rule=...)`` in the codebase
and verifies the literal is registered here. This converts the "typo in
error code" bug class into an immediate CI failure.
"""

from __future__ import annotations

from dataclasses import dataclass

# NOTE (bead OpenBBTechnical-3cf E0.1 review):
# Post-split, `compiler_errors` has no top-level import of `error_codes`
# (verified 2026-07-07) so a top-of-file import here is safe. The prior
# deferred import inside `assert_code_registered` guarded against a
# speculative cycle that never materialized. If a future refactor makes
# `compiler_errors` import from `error_codes`, revert this to a lazy
# import inside the function to break the cycle.
from pyne_compiler.errors.base import PineInternalCompilerError


@dataclass(frozen=True, slots=True)
class ErrorCodeSpec:
    """Metadata for a single error code — mirrors what
    ``docs/designs/openbb-pine/error-codes.md`` documents user-facing."""

    code: str
    class_name: str
    short_description: str
    detailed_description: str
    since_version: str
    tracking_label: str | None = None
    """GitHub label filter that surfaces this code's issues."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Rows kept single-lineish so ``grep`` on ``"CG002"`` etc. finds one hit.
# Long descriptions folded across a trailing implicit string concatenation.
#

ERROR_CODES: dict[str, ErrorCodeSpec] = {
    # --- Syntax (PS###) ---------------------------------------------------
    "PS001": ErrorCodeSpec(
        code="PS001",
        class_name="PineSyntaxError",
        short_description="Unexpected token",
        detailed_description=(
            "The lexer or parser hit a token that no grammar production "
            "accepts at the current position. Typically caused by typos, "
            "unbalanced parens, or missing operators between operands. "
            "The message includes a hint sourced from lark's "
            "match_examples() when the failure matches a known pattern."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS002": ErrorCodeSpec(
        code="PS002",
        class_name="PineSyntaxError",
        short_description="Malformed version pragma",
        detailed_description=(
            "The ``//@version=N`` pragma at line 1 is missing the version "
            "number or has non-digit characters after the ``=`` sign."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS003": ErrorCodeSpec(
        code="PS003",
        class_name="PineSyntaxError",
        short_description="Mismatched indentation",
        detailed_description=(
            "A dedent doesn't line up with any previously-seen indent "
            "level. Pine uses Python-style significant indentation; "
            "mixed tabs + spaces or an unexpected outdent trigger this."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS004": ErrorCodeSpec(
        code="PS004",
        class_name="PineSyntaxError",
        short_description="Unexpected character",
        detailed_description=(
            "The lexer hit a character not part of any valid token — "
            "typically stray unicode punctuation, or an ASCII character "
            "outside the alphabet Pine accepts at the current position."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS005": ErrorCodeSpec(
        code="PS005",
        class_name="PineSyntaxError",
        short_description="Unterminated string literal",
        detailed_description=(
            "A string literal (`\"...\"` or `'...'`) did not reach its "
            "closing quote before EOF or end-of-line."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS006": ErrorCodeSpec(
        code="PS006",
        class_name="PineSyntaxError",
        short_description="Unsupported pine_version",
        detailed_description=(
            "``compile_pine(pine_version=N)`` was called with an N outside "
            "the supported {5, 6} set. v5 is auto-migrated to v6 by C7."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),
    "PS007": ErrorCodeSpec(
        code="PS007",
        class_name="PineSyntaxError",
        short_description="Missing grammar file",
        detailed_description=(
            "The lark grammar file for the requested Pine version is not "
            "present in ``compiler/grammar/``. Package integrity issue."
        ),
        since_version="0.1.0",
        tracking_label="pine-syntax",
    ),

    # --- Type checker (PT001-PT008 reserved by D1 §4.4) -------------------
    "PT001": ErrorCodeSpec(
        code="PT001",
        class_name="PineTypeError",
        short_description="Function parameter qualifier mismatch",
        detailed_description=(
            "A function parameter declared ``simple<T>`` cannot receive a "
            "``series<T>`` argument (e.g. ``ta.sma(close, length)`` where "
            "``length`` is ``simple<int>``, not ``series<int>``). Pine's "
            "qualifier lattice: ``const → input → simple → series`` — "
            "promotion is one-way and cannot be reversed."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT002": ErrorCodeSpec(
        code="PT002",
        class_name="PineTypeError",
        short_description="`var` initializer must be simple",
        detailed_description=(
            "``var x = e`` requires ``e: simple<T>`` (Pine spec: ``var`` "
            "initializers run once at bar 0)."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT003": ErrorCodeSpec(
        code="PT003",
        class_name="PineTypeError",
        short_description="`if`/`while` condition must be boolean",
        detailed_description=(
            "``if cond`` requires ``cond: series<bool>`` or "
            "``simple<bool>``; implicit truthiness on ``int``/``float`` is "
            "rejected."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT004": ErrorCodeSpec(
        code="PT004",
        class_name="PineTypeError",
        short_description="History access requires series<T> value + integer offset",
        detailed_description=(
            "``x[n]`` history access requires ``x: series<T>`` and ``n`` "
            "must be ``simple<int>`` or ``const<int>``."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT005": ErrorCodeSpec(
        code="PT005",
        class_name="PineTypeError",
        short_description="`:=` reassign qualifier violation",
        detailed_description=(
            "``:=`` reassign target must already be declared, and the RHS "
            "qualifier must be ≤ LHS qualifier per the lattice."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT006": ErrorCodeSpec(
        code="PT006",
        class_name="PineTypeError",
        short_description="`na` propagation",
        detailed_description=(
            "Any non-``na`` op with an ``na`` operand yields ``na`` of "
            "the inferred type (``NaT.unify(T) → T``)."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT007": ErrorCodeSpec(
        code="PT007",
        class_name="PineTypeError",
        short_description="v6 type argument mismatch",
        detailed_description=(
            "v6 type arguments (e.g. ``array.new<float>(0)``) must satisfy "
            "the builtin's signature; mismatch names both expected and "
            "received. Reserved for the v6 type-argument checker."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),
    "PT008": ErrorCodeSpec(
        code="PT008",
        class_name="PineTypeError",
        short_description="UDT field not found",
        detailed_description=(
            "``obj.field`` access requires ``field`` to exist on the UDT; "
            "suggestions via Levenshtein distance."
        ),
        since_version="0.1.0",
        tracking_label="pine-type",
    ),

    # --- Unsupported features (PF###) -------------------------------------
    "PF001": ErrorCodeSpec(
        code="PF001",
        class_name="PineUnsupportedFeatureError",
        short_description="Pine v1-v4 not supported",
        detailed_description=(
            "The migration shim covers v5→v6 only; v1-v4 is an explicit "
            "PRD §3.3 non-goal. See the tracking issue to request a "
            "specific script be re-evaluated."
        ),
        since_version="0.1.0",
        tracking_label="pine-v5-migration",
    ),
    "PF002": ErrorCodeSpec(
        code="PF002",
        class_name="PineUnsupportedFeatureError",
        short_description="Typed decl in body / future-version pragma",
        detailed_description=(
            "Two distinct cases share this code: (a) the C3 type checker "
            "rejects typed variable declarations in function bodies "
            "(they land later), (b) the compiler refuses to speculate on "
            "a Pine version above v6."
        ),
        since_version="0.1.0",
        tracking_label="pine-feature",
    ),
    "PF003": ErrorCodeSpec(
        code="PF003",
        class_name="PineUnsupportedFeatureError",
        short_description="v5 construct not migrated",
        detailed_description=(
            "The v5 → v6 regex rewrite in ``V5_REWRITES`` did not match "
            "this call (likely a multi-line or deeply-nested form). File "
            "a rewrite request with a minimal repro."
        ),
        since_version="0.1.0",
        tracking_label="pine-v5-migration",
    ),
    "PF010": ErrorCodeSpec(
        code="PF010",
        class_name="PineUnsupportedFeatureError",
        short_description="strategy() decl — deferred to M2",
        detailed_description=(
            "Strategy compilation lands at M2 per PRD §3.2 (bead 0e9.5.6). "
            "Indicator scripts compile today; strategy scripts are "
            "scaffolded but fast-fail here."
        ),
        since_version="0.1.0",
        tracking_label="pine-feature",
    ),
    "PF011": ErrorCodeSpec(
        code="PF011",
        class_name="PineUnsupportedFeatureError",
        short_description="library() decl — deferred (no Phase-1 use-case)",
        detailed_description=(
            "``library()`` scripts define reusable functions for import by "
            "other Pine scripts. No Phase-1 use-case; deferred until "
            "someone files a request."
        ),
        since_version="0.1.0",
        tracking_label="pine-feature",
    ),

    # --- Codegen (CG###) --------------------------------------------------
    "CG001": ErrorCodeSpec(
        code="CG001",
        class_name="PineCodegenError",
        short_description="Disallowed AST node type",
        detailed_description=(
            "Codegen emitted an ``ast.*`` node outside "
            "``NODE_TYPE_ALLOWLIST`` per D1 §3.2. Always a compiler bug; "
            "a fresh CG001 in production is a P0."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),
    "CG002": ErrorCodeSpec(
        code="CG002",
        class_name="PineCodegenError",
        short_description="Disallowed ImportFrom module",
        detailed_description=(
            "Codegen emitted an ``ast.ImportFrom`` whose ``.module`` is "
            "outside ``MODULE_ALLOWLIST``. All ``ast.Import`` nodes are "
            "rejected as a category. Always a compiler bug."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),
    "CG003": ErrorCodeSpec(
        code="CG003",
        class_name="PineCodegenError",
        short_description="Disallowed top-level free name",
        detailed_description=(
            "Codegen emitted a top-level ``ast.Name`` outside "
            "``GLOBAL_NAME_ALLOWLIST`` per D1 §3.2. Always a compiler bug."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),
    "CG004": ErrorCodeSpec(
        code="CG004",
        class_name="PineCodegenError",
        short_description="Unhandled IR statement kind",
        detailed_description=(
            "``_visit_stmt`` hit an ``ir.*`` statement node the visitor "
            "does not know how to lower. Compiler-invariant; means a new "
            "IR node was added without a corresponding visit method."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),
    "CG005": ErrorCodeSpec(
        code="CG005",
        class_name="PineCodegenError",
        short_description="Unhandled assignment-target kind",
        detailed_description=(
            "``_visit_assign_target`` was given a target that is not one "
            "of Name / Attribute / Subscript / TupleExpr."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),
    "CG006": ErrorCodeSpec(
        code="CG006",
        class_name="PineCodegenError",
        short_description="Unhandled IR expression kind",
        detailed_description=(
            "``_visit_expr`` hit an ``ir.*`` expression node the visitor "
            "does not know how to lower."
        ),
        since_version="0.1.0",
        tracking_label="pine-codegen",
    ),

    # --- Internal compiler (IC###) ----------------------------------------
    "IC001": ErrorCodeSpec(
        code="IC001",
        class_name="PineInternalCompilerError",
        short_description="Unregistered error code raised",
        detailed_description=(
            "A ``raise Pine*Error(rule=...)`` used a code that is not "
            "present in :data:`ERROR_CODES`. The AST-walking enforcement "
            "test catches this at CI time; this class exists so a runtime "
            "invocation of ``assert_code_registered`` also raises "
            "structurally when the code is missing."
        ),
        since_version="0.1.0",
        tracking_label="pine-internal",
    ),

    # --- Security (SEC###) ------------------------------------------------
    "SEC001": ErrorCodeSpec(
        code="SEC001",
        class_name="PineSecurityError",
        short_description="Forbidden import (T3 sandbox violation)",
        detailed_description=(
            "The runtime's ``scan_for_forbidden_imports`` (T3 second line "
            "of defense) found an import of a disallowed module in the "
            "compiled source. Ordinarily impossible because the T1 "
            "compiler-side allowlist gate should have blocked it; this "
            "code firing in production means either the allowlist has a "
            "hole or the compiled source was tampered with."
        ),
        since_version="0.1.0",
        tracking_label="pine-security",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup(code: str) -> ErrorCodeSpec | None:
    """Return the :class:`ErrorCodeSpec` for ``code``, or None if unknown."""
    return ERROR_CODES.get(code)


def assert_code_registered(code: str) -> None:
    """Raise :class:`PineInternalCompilerError` (IC001) when ``code`` is not
    in :data:`ERROR_CODES`.

    Callers may want to guard their own raise sites with this at import time
    to fail fast rather than at the first-user-hits-the-code-path. The
    AST-walking enforcement test covers the same ground for the whole
    codebase at CI time.
    """
    if code not in ERROR_CODES:
        raise PineInternalCompilerError(
            rule="IC001",
            invariant=(
                f"error code {code!r} raised but not registered in "
                f"openbb_pine.error_codes.ERROR_CODES"
            ),
            hint=(
                "Add an ErrorCodeSpec entry to ERROR_CODES for this code, "
                "or fix the typo at the raise site."
            ),
        )


__all__ = ["ErrorCodeSpec", "ERROR_CODES", "lookup", "assert_code_registered"]
