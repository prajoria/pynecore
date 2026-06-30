"""The openbb-pine compiler — D1 territory.

This package houses the lexer (C1), parser/IR builder (C2), type checker (C3),
IR dataclasses (C4), codegen (C5), compile cache (C6), and v5→v6 auto-migration
shim (C7).

Public surfaces
---------------

* :func:`compile_pine` — high-level facade: ``str source -> IR Program``.
  Detects the ``//@version=`` pragma, applies :func:`migrate_v5_to_v6` if
  the source is v5, then tokenises and parses under the target grammar.
  Most callers (REST router D3 §4.6, CLI ``openbb-pine run``,
  ``obb.pine.run``) want this.
* :func:`tokenize`, :func:`parse`, :func:`migrate_v5_to_v6`,
  :func:`detect_pine_version` — re-exported low-level surfaces for tests
  + advanced usage. The compile pipeline is layered so each stage stays
  independently testable.

D1 §1.5 says the compile pipeline is ``source → lexer → migration? →
parser → type-checker → codegen → cache``. The shim slot lives between
the lexer's pragma detection and the parser's grammar dispatch — but
because the v5→v6 rewrites are source-level (PRD §3.2 mentions only
syntactic rewrites), we apply the shim BEFORE the lexer so the lexer
sees a v6 source. The lexer is dialect-agnostic anyway; running it
before vs. after the rewrites is observably identical.

Higher-level entry points (REST endpoints, OBBject construction) live in
``openbb_pine.pine_router``; this package stays pure compiler.
"""

from __future__ import annotations

from openbb_pine.compiler.lexer import Token, tokenize
from openbb_pine.compiler.parser import parse
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
    "compile_pine",
    "V5Rewrite",
    "V5_REWRITES",
    "detect_pine_version",
    "migrate_v5_to_v6",
]


def compile_pine(source: str, *, target_version: int = 6) -> "ir.Program":  # noqa: F821
    """Parse a Pine source string into an IR Program (D1 §1.5).

    The high-level compile-pipeline facade. Hides three concerns from
    callers:

    1. **Version detection.** Reads ``//@version=N`` via
       :func:`detect_pine_version`. Defaults to v6 when no pragma is
       found (per PRD §2.1 + TradingView editor's behavior).
    2. **v5→v6 migration.** When ``detected != target_version`` and the
       transition is the supported v5→v6, runs :func:`migrate_v5_to_v6`
       transparently. Other ``(detected, target)`` pairs raise — there
       is no v6→v5 down-migration and the v4-or-earlier branch is a PRD
       §3.3 non-goal.
    3. **Tokenize + parse under the target grammar.** Returns the IR
       :class:`Program` node the type-checker (C3) and codegen (C5)
       consume.

    Returns
    -------
    Program
        The IR root, with ``version=target_version`` after migration.

    Raises
    ------
    PineUnsupportedFeatureError
        Source uses an unsupported version (PF001 / PF002) or a v5
        construct no rewrite handles (PF003).
    PineSyntaxError
        Source fails to tokenise or parse under the target grammar.

    Notes
    -----
    The ``target_version`` kwarg is accepted but currently only 6 is
    valid. The parameter exists so the call-site contract is forward-
    compatible with the v6→v7 migration when TradingView ships a v7
    (PRD §11 "v7 reopens this").
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
        # type pure (Program, not (Program, log)) preserves the contract
        # CLI + obb.pine.run share with the facade.
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

    tokens = tokenize(source)
    return parse(tokens, pine_version=target_version)
