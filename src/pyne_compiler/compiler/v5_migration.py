"""Pine v5 → v6 auto-migration shim (D1 §1.5, PRD §3.2 Phase 1, §8.1 M1 gate (h)).

A user submits a Pine v5 source; we rewrite the small set of v5-only
constructs into their v6 equivalents and let the v6 parser/grammar do the
rest. This keeps the parser's grammar work mono-versioned — the v5 grammar
file (``pine_v5.lark``) stays a placeholder that imports the v6 grammar
verbatim, and the v5→v6 rewrites live here as source-level transforms.

Why source-level and not IR-level
---------------------------------

* The diffs between Pine v5 and v6 are syntactic / surface-renames, not
  semantic — ``study()`` → ``indicator()``, ``transp=`` stripped from
  ``plot()``, ``iff(c,a,b)`` → ``(c ? a : b)``, ``tickerid()`` →
  ``ticker.new()``, ``security()`` → ``request.security()``. Each fits in
  a regex applied to the source string.
* Source-level rewrites are diffable and explainable: the
  ``rewrites_log`` returned from :func:`migrate_v5_to_v6` lists every
  applied transform, which both the cookbook (PRD Phase 4) and the
  M1-gate test surface as user-visible reassurance.
* An IR-level migrator would force ``parse()`` to ingest v5-flavoured
  source first — but the v5 grammar isn't carved out (placeholder per
  Wave 2A). Going source-level dodges the chicken-and-egg.

Coverage honesty
----------------

This is the **most-common 5-10 rewrites** that the wild-corpus
fingerprint scan (PRD §3.4) flagged on a Phase 0 sweep — not a complete
v5→v6 catalog. The long tail (esoteric ``input()`` overloads, deprecated
``study.*`` namespace) is marked ``# TODO(P3)`` and surfaces as
:class:`openbb_pine.errors.PineUnsupportedFeatureError` (code ``PF003``)
with a tracking URL pointing to the GitHub label for filing new rewrite
requests. The honest commitment per the bead brief: M1 is "your average
RSI / Bollinger / SuperTrend v5 script compiles unedited," not
"every script TradingView ever shipped."

Clean-room note
---------------

The rewrite catalog is derived from TradingView's public **Pine Script
v6 migration guide** (the v5→v6 changelog section of the Pine reference
manual — *behavioral spec*, allowed per PRD §2.5). We did not view
TradingView's compiler source or PyneComp's source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyne_compiler.errors.base import PineUnsupportedFeatureError

if TYPE_CHECKING:  # pragma: no cover — imports for typing only
    # Post-9bh: telemetry now lives at pyne_compiler.telemetry (extraction
    # complete). Kept behind TYPE_CHECKING so the compiler never actually
    # imports telemetry at runtime — the E0.4 injection contract.
    from pyne_compiler.telemetry import TelemetrySink

__all__ = [
    "V5Rewrite",
    "V5_REWRITES",
    "detect_pine_version",
    "migrate_v5_to_v6",
]


# ---------------------------------------------------------------------------
# V5Rewrite — one source-level transform.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V5Rewrite:
    """One unit of v5→v6 source-level transformation (PRD §3.2).

    Parameters
    ----------
    name:
        Short human-readable label, surfaced verbatim in the rewrites log
        returned by :func:`migrate_v5_to_v6`. Convention: ``"<v5> →
        <v6>"`` so a log line reads as a one-shot explanation.
    pattern:
        Compiled :class:`re.Pattern` matched against the post-pragma
        source body. We keep ``str`` accepted at construction for ergonomic
        in-place edits; :meth:`__post_init__` compiles strings into
        patterns so the apply-loop sees only :class:`re.Pattern`.
    replacement:
        Pattern-substitution string. Supports :mod:`re` backrefs (``\\1``
        etc.) so a rewrite can preserve captured groups.
    description:
        One-line rationale; reference to the TradingView v5→v6 manual
        heading the rewrite implements. Surfaced in cookbook diff output
        and in any "why was my script rewritten" diagnostic.
    bidi_safe:
        True when applying the rewrite twice produces the same result
        (i.e. the v6 output does not itself match ``pattern``). Idempotent
        rewrites are checked explicitly in
        ``tests/unit/test_v5_migration.py`` to catch the next contributor
        who adds a rewrite whose output is its own input.
    """

    name: str
    pattern: re.Pattern[str]
    replacement: str
    description: str
    bidi_safe: bool = True

    def apply(self, source: str) -> tuple[str, int]:
        """Return ``(new_source, num_substitutions_made)``.

        Wraps :meth:`re.Pattern.subn` so the apply-loop in
        :func:`migrate_v5_to_v6` can decide whether to log this rewrite
        based on whether it actually fired (``num > 0``).
        """
        return self.pattern.subn(self.replacement, source)


def _re(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Wrapper for :func:`re.compile` so the V5_REWRITES table reads
    declaratively (no inline ``re.compile`` clutter).
    """
    return re.compile(pattern, flags)


# ---------------------------------------------------------------------------
# V5_REWRITES — the catalog (PRD §3.2 + Phase-1 brief).
# ---------------------------------------------------------------------------


# Order matters for two reasons:
#
# 1. `study(` → `indicator(` MUST run before any rewrite that targets
#    indicator-specific kwargs, because v6 callers will only see those
#    kwargs after the directive is renamed.
# 2. `tickerid()` → `ticker.new()` MUST run before any rewrite that
#    substitutes namespaces involving `ticker.*`, otherwise the rewrites
#    would cascade in ways that break idempotence.
#
# Every entry is `bidi_safe=True` by construction — see the per-rewrite
# note. The idempotence invariant is asserted in
# ``tests/unit/test_v5_migration.py::TestIdempotence``.
V5_REWRITES: tuple[V5Rewrite, ...] = (
    # ---- (1) study() → indicator() ---------------------------------------
    # The canonical v5→v6 rename. v5 top-level declaration was `study(...)`;
    # v6 renamed for clarity (also covers the `study.title`/`study.shorttitle`
    # naming that the deprecated study.* namespace exposed — but the namespace
    # itself is rewritten separately, see below).
    V5Rewrite(
        name="study() → indicator()",
        # `\b` so we don't accidentally match `mystudy(`; `\s*\(` so optional
        # whitespace between `study` and `(` is tolerated.
        pattern=_re(r"\bstudy\s*\("),
        replacement="indicator(",
        description=(
            "TV v5→v6 manual: top-level `study(...)` directive was renamed "
            "to `indicator(...)` in v6 for naming clarity (consistent with "
            "`strategy(...)`, `library(...)`)."
        ),
    ),
    # ---- (2) transp= argument stripping ---------------------------------
    # v5 `plot(..., transp=N)` controlled opacity. In v6 the `transp=`
    # argument was removed; opacity now flows through `color=color.new(c, N)`
    # at the color-construction site. Stripping the kwarg yields a v6 script
    # that runs (with a slight visual differential — fully-opaque output).
    # The cookbook will document the manual color.new(...) upgrade for users
    # who need the exact opacity behavior.
    V5Rewrite(
        name="strip transp= arg",
        # Match `transp = N` or `transp=N`, optionally preceded by a comma
        # plus whitespace (the common form inside a call). Drop the whole
        # argument including its leading comma so the remaining call is
        # syntactically clean.
        pattern=_re(r",\s*transp\s*=\s*\d+(?:\.\d+)?"),
        replacement="",
        description=(
            "TV v5→v6 manual: the `transp=` kwarg on `plot()` / "
            "`hline()` / etc. was removed in v6. Opacity now flows "
            "through `color=color.new(base, transp)` at the color "
            "construction site. We strip the kwarg; users who need "
            "exact opacity must migrate via `color.new(...)` manually "
            "(cookbook)."
        ),
    ),
    # ---- (3) iff(cond, a, b) → (cond ? a : b) ---------------------------
    # v5's `iff()` helper was deprecated in v6; the spec now mandates the
    # ternary operator. The rewrite preserves the exact arguments.
    #
    # The inner groups use `[^,()]+` rather than a balanced parser because
    # Pine doesn't allow nested calls to fit inside an `iff()` very often,
    # and where they do, the regex degenerates harmlessly (the call is
    # rewritten only if all three args are simple expressions; otherwise
    # the unrewritten `iff(...)` triggers an PF003 unsupported-feature
    # error and the user gets the cookbook nudge to inline manually).
    V5Rewrite(
        name="iff(c,a,b) → (c ? a : b)",
        pattern=_re(r"\biff\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        replacement=r"(\1 ? \2 : \3)",
        description=(
            "TV v5→v6 manual: the `iff(cond, then, else)` helper was "
            "deprecated. v6 mandates the ternary operator `cond ? then : "
            "else`. Three-arg-only rewrite; nested `iff()` calls require "
            "manual inlining (cookbook)."
        ),
    ),
    # ---- (4) tickerid(...) → ticker.new(...) ----------------------------
    # v5 had a bare `tickerid()` function; v6 moved it under the `ticker`
    # namespace as `ticker.new()`. The signature is otherwise identical so
    # the replacement is a name-only rename.
    V5Rewrite(
        name="tickerid() → ticker.new()",
        pattern=_re(r"\btickerid\s*\("),
        replacement="ticker.new(",
        description=(
            "TV v5→v6 manual: `tickerid(prefix, ticker, session, "
            "adjustment)` moved into the `ticker` namespace as "
            "`ticker.new(...)` for consistency with the rest of the "
            "namespace migration."
        ),
    ),
    # ---- (5) security(...) → request.security(...) ----------------------
    # v5 exposed `security()` as a top-level builtin; v6 namespaces it under
    # `request.*`. PRD §3.2 Phase 2 fully implements `request.security` for
    # cross-symbol calls; the rewrite is unconditional so v5 scripts pass
    # through Phase 1 parse even if execution still raises (the runtime
    # error is the right place to nudge users to upgrade to a fully Phase-2
    # build).
    V5Rewrite(
        name="security() → request.security()",
        # `\b(?<!\.)` keeps us from rewriting `something.security(` —
        # which is already namespaced and should be left alone.
        pattern=_re(r"(?<![\w.])security\s*\("),
        replacement="request.security(",
        description=(
            "TV v5→v6 manual: top-level `security()` was moved into the "
            "`request` namespace as `request.security()`. The signature is "
            "otherwise identical."
        ),
    ),
    # TODO(P3): the long-tail rewrites we have NOT shipped here. Each
    # surfaces as PineUnsupportedFeatureError(PF003) so users can file the
    # rewrite request against the tracking_url. Candidates surfaced by the
    # PRD §3.4 wild-corpus fingerprint scan:
    #
    #   * `nz(x)` / `nz(x, y)` — Pine v6 kept it, no rewrite needed.
    #   * `valuewhen()` / `barssince()` — kept under `ta.*` in v6; the
    #     leading-`ta.` migration is unstructured and varies by call site
    #     enough that a wide regex would over-rewrite. P3.
    #   * `study(scale=…)` — v5 `scale=scale.right` had a v6 equivalent
    #     under `scale=` on `indicator(...)` but kwarg names diverge.
    #     Needs argument-key rewrite, not just function rename. P3.
    #   * `input(...)` v5 single-call form → `input.int()` / `input.float()`
    #     / `input.bool()` etc. — needs type-inference from the default
    #     argument to pick the right v6 typed-input builtin. P3.
    #   * `na(x)` keyword form differences — kept identical in v6 for
    #     positional, but v6 added stricter named-arg handling. P3 if it
    #     bites a wild-corpus script.
    #   * `bar_index` semantics — v5 and v6 agree on the value but v6
    #     hardened the type to `series int`; pure rename, no rewrite
    #     needed at the source level (the type checker C3 will reject any
    #     v5 script that conflates `bar_index` with a `simple int`).
    #
    # The honest position: P3 expands this catalog driven by telemetry
    # from `pine_unsupported_feature_total` (the metric is wired in M1
    # when this module ships; P3 is when the wild-corpus volume of
    # PF003s makes the long tail worth the maintenance cost).
)


# ---------------------------------------------------------------------------
# detect_pine_version — read the `//@version=N` pragma; default to 6.
# ---------------------------------------------------------------------------


_PRAGMA_PATTERN: re.Pattern[str] = re.compile(
    r"^//@version=(?P<v>\d+)\s*$",
    re.MULTILINE,
)


def detect_pine_version(source: str, *, telemetry: TelemetrySink | None = None) -> int:
    """Return 4, 5, or 6 based on the ``//@version=N`` pragma (D1 §1.2).

    Defaults to 6 when no pragma is present (consistent with the lexer's
    treatment of unprefixed source — TradingView's editor inserts the
    pragma automatically, so unprefixed source is more likely to be a
    paste-of-a-snippet than a true legacy v4 script).

    Raises :class:`PineUnsupportedFeatureError` for v1-v4 (out of scope per
    PRD §3.3) and for v7+ (we don't speculate ahead). The error carries
    code ``PF001`` (legacy) or ``PF002`` (future) for REST routing.

    ``telemetry`` (E0.4): optional sink. When provided, the PF001/PF002
    raise paths call ``telemetry.record_unsupported_feature(code)`` on
    the sink BEFORE raising so an outer ``except`` cannot swallow the
    telemetry signal. When ``None`` (default), no sink is touched — the
    caller has opted out of telemetry collection.

    Only the first column-0 pragma is considered. Subsequent ``//@version=``
    occurrences are treated as regular comments — matching the lexer's
    column-0-only routing.
    """
    # Find the very first occurrence anchored at column 0. The
    # ``re.MULTILINE`` flag makes ``^`` match line-starts; we then verify
    # the match started at byte 0 OR right after a newline so we don't
    # accept an indented pragma (which a user shouldn't write but, were
    # we to, would mis-route grammar).
    m = _PRAGMA_PATTERN.search(source)
    if m is None:
        return 6
    version = int(m.group("v"))
    if version in (5, 6):
        return version
    if version < 5:
        if telemetry is not None:
            telemetry.record_unsupported_feature("PF001")
        raise PineUnsupportedFeatureError(
            message=(
                f"PF001 Pine v{version} is not supported. The migration shim "
                f"covers v5→v6 only; v1-v4 is an explicit PRD §3.3 non-goal. "
                f"See {PineUnsupportedFeatureError.tracking_url} to request a "
                f"specific v{version} script be re-evaluated."
            )
        )
    if telemetry is not None:
        telemetry.record_unsupported_feature("PF002")
    raise PineUnsupportedFeatureError(
        message=(
            f"PF002 Pine v{version} pragma found, but the compiler targets "
            f"v6. We don't speculate ahead — when TradingView ships v{version}, "
            f"file a tracking issue at {PineUnsupportedFeatureError.tracking_url}."
        )
    )


# ---------------------------------------------------------------------------
# migrate_v5_to_v6 — apply every rewrite, return new source + log.
# ---------------------------------------------------------------------------


# Pragma line: `//@version=5` (column 0, optional trailing whitespace).
# We rewrite the pragma so the parser dispatches the v6 grammar.
_V5_PRAGMA_REWRITE: re.Pattern[str] = re.compile(r"(?m)^//@version=5\s*$")


# A short list of v5 tokens that signal "this rewrite isn't covered yet."
# We scan the post-rewrite source for any of these and raise PF003 if any
# remain — the rationale is that if a v5 script still mentions `study(` or
# `iff(` after the rewrite loop ran, one of our rewrites failed (or the
# user wrote a malformed call our regex couldn't match), and silently
# letting it through would yield a confusing parse error downstream.
#
# Each entry is a (regex, human-readable feature name) pair. The first
# match wins — error surfaces the human name.
#
# The sentinel scan is run against a comment-stripped copy of the source
# (see ``_strip_line_comments``) so a v5-themed user comment like
# ``// iff(c, a, b) helper`` doesn't trigger a false PF003.
_UNHANDLED_V5_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # If `study(` survived the rewrite loop, our pattern didn't fire — likely
    # because the call spans lines in a way the simple regex missed.
    (re.compile(r"\bstudy\s*\("), "study()"),
    # Similarly for `iff(`. The three-arg simple-expr regex is conservative;
    # nested-call iff() falls through and surfaces here.
    (re.compile(r"\biff\s*\("), "iff()"),
    # `tickerid(` should be rewritten; if it isn't, complain.
    (re.compile(r"\btickerid\s*\("), "tickerid()"),
)


# Strip `// ...` line comments from source. The regex is conservative:
# it does NOT recognise `/* ... */` block comments (Pine doesn't have
# them) and it preserves strings — well-formed Pine has no `//` inside
# string literals once the lexer normalises the token stream, but at
# this layer we're operating on raw source. A naive replace is safe
# here because the sentinel scan tolerates false negatives (a missed
# unhandled construct surfaces as a parse error a few microseconds
# later — degraded UX, not wrong behavior).
_LINE_COMMENT: re.Pattern[str] = re.compile(r"//[^\n]*")


def _strip_line_comments(source: str) -> str:
    """Return ``source`` with `// ...` comments replaced by empty strings.

    Used by :func:`migrate_v5_to_v6` to scrub user comments before the
    PF003 sentinel scan. The scan is checking for "unhandled v5
    construct in actual code," not "v5 syntax mentioned in a doc
    comment."
    """
    return _LINE_COMMENT.sub("", source)


# ---------------------------------------------------------------------------
# #592: strategy.*(when=…) → strategy.*(…)  — v5→v6 order-placement rewrite
# ---------------------------------------------------------------------------

# The seven Pine v5 order-placement functions that v6 stripped `when=` from.
# Kept explicit rather than a wildcard so a future v6 addition of a strategy.*
# call that legitimately takes a `when=` (unlikely, but the shim wouldn't know)
# doesn't get chewed by accident.
_STRATEGY_WHEN_STRIP_FUNCTIONS: tuple[str, ...] = (
    "strategy.entry",
    "strategy.order",
    "strategy.exit",
    "strategy.close_all",  # BEFORE strategy.close so the regex prefers the longer match
    "strategy.close",
    "strategy.cancel_all",  # BEFORE strategy.cancel likewise
    "strategy.cancel",
)

# One matcher against every strategy.* call opener. Iteration below walks each
# match, finds the balanced paren pair, and rewrites the arglist in-place.
# Ordered longest-first so `strategy.close_all(` matches before `strategy.close(`
# would in an alt-group.
_STRATEGY_CALL_OPENER: re.Pattern[str] = re.compile(
    r"\bstrategy\.(?:entry|order|exit|close_all|close|cancel_all|cancel)\s*\("
)


def _find_matching_paren(text: str, open_paren_idx: int) -> int | None:
    """Return the index of the ``)`` that closes the ``(`` at
    ``open_paren_idx``, or ``None`` if no balanced closer exists
    (unterminated, or nested-string edge-case that would trip a naive
    scanner).

    Skips over single- and double-quoted string literals so a ``)`` inside
    a string is not mistaken for the closer. Does NOT handle triple-quoted
    strings — Pine has none. Does NOT handle escape sequences other than
    ``\\"`` and ``\\'``; Pine strings are simple.

    Silent failure (``None``) is intentional: if the call site is
    unbalanced, we leave it unchanged and let the parser produce a
    grammar error at its usual site rather than corrupt the source.
    """
    assert text[open_paren_idx] == "(", "must start at an opening paren"
    depth = 0
    i = open_paren_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ("'", '"'):
            # Walk past the string literal.
            quote = ch
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    i += 2  # skip escaped char
                    continue
                i += 1
            if i >= n:
                return None  # unterminated string
            i += 1  # skip the closing quote
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _split_top_level_args(arglist: str) -> list[str] | None:
    """Split a Pine call's argument list on top-level commas.

    ``arglist`` is the text BETWEEN the outer parens (no surrounding
    whitespace stripping). Returns ``None`` if the arglist is malformed
    (unbalanced parens or unterminated string) — caller should leave the
    call unchanged in that case.

    Skips commas inside strings and inside nested parens/brackets so a
    call like ``strategy.exit("tp", when=math.max(a, b), profit=10)`` is
    split as three args, not five.
    """
    args: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(arglist)
    while i < n:
        ch = arglist[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n and arglist[i] != quote:
                if arglist[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            if i >= n:
                return None
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif ch == "," and depth == 0:
            args.append(arglist[start:i])
            start = i + 1
        i += 1
    if depth != 0:
        return None
    args.append(arglist[start:])
    return args


def _strip_when_arg_from_strategy_calls(source: str) -> tuple[str, int]:
    """Remove the ``when=`` kwarg from every ``strategy.*`` call in
    ``source`` (see :data:`_STRATEGY_WHEN_STRIP_FUNCTIONS`).

    Returns ``(new_source, count)`` where ``count`` is the number of
    call sites where a ``when=`` was actually stripped. A call site with
    no ``when=`` kwarg is not counted.

    Semantics:
      * Simple arg forms are stripped in place — the remaining arglist is
        rejoined with commas and the call's opening/closing parens are
        preserved verbatim.
      * A call whose argument list is malformed (unbalanced parens or
        unterminated string) is left unchanged; the parser produces the
        real error a few passes later.
      * A ``when=`` occurrence inside a nested call's arg list (e.g. a
        v5 user wrapped ``strategy.exit`` around another builtin that
        happens to take a ``when=`` kwarg — unusual but not impossible)
        is NOT rewritten by this pass; only the outer ``strategy.*``
        arglist is walked.

    Idempotent: after a strip pass, the call has no ``when=`` kwarg, so a
    second pass over the same source strips zero and returns the input
    unchanged. Assertion coverage in
    ``tests/unit/test_v5_migration.py::TestIdempotence``.
    """
    # Walk left→right, rewriting as we go. Because rewrites shorten the
    # source, we track the offset delta relative to the original scan and
    # keep the match indices valid against the original string.
    out_parts: list[str] = []
    cursor = 0
    strip_count = 0

    for opener in _STRATEGY_CALL_OPENER.finditer(source):
        # Copy verbatim from the last cursor up to (and including) the
        # ``(`` — the opener match ends at index `opener.end() - 1` for
        # the paren, so `opener.end()` is the first arg-list char.
        open_paren_idx = opener.end() - 1
        close_paren_idx = _find_matching_paren(source, open_paren_idx)
        if close_paren_idx is None:
            # Malformed call — leave it alone.
            continue

        arglist = source[open_paren_idx + 1 : close_paren_idx]
        parts = _split_top_level_args(arglist)
        if parts is None:
            continue

        # Filter out any arg that is a `when=` kwarg. `when` must be the
        # LHS of an `=`, not the RHS of something like `x = when_flag`.
        # Match ``\s* when \s* = ...``.
        _WHEN_KWARG = re.compile(r"^\s*when\s*=")
        kept = [p for p in parts if not _WHEN_KWARG.match(p)]
        if len(kept) == len(parts):
            # No `when=` found in this call — nothing to strip.
            continue

        strip_count += len(parts) - len(kept)
        # Reconstruct the arglist. If a `when=` was the ONLY arg
        # (e.g. `strategy.close_all(when=cond)` → `strategy.close_all()`),
        # `kept` is []; the rejoin naturally produces "".
        new_arglist = ",".join(kept)

        # Emit source[cursor : open_paren_idx+1] + new_arglist + ")"
        out_parts.append(source[cursor : open_paren_idx + 1])
        out_parts.append(new_arglist)
        out_parts.append(")")
        cursor = close_paren_idx + 1

    # Tail.
    out_parts.append(source[cursor:])
    return "".join(out_parts), strip_count


def migrate_v5_to_v6(
    source: str, *, telemetry: TelemetrySink | None = None
) -> tuple[str, list[str]]:
    """Apply every entry in :data:`V5_REWRITES` to a v5 source string.

    Returns ``(new_source, applied_rewrites_log)`` where the log lists
    one entry per applied rewrite in source order, in the shape::

        "study() → indicator() (3 substitution(s))"

    The new source also has its ``//@version=5`` line rewritten to
    ``//@version=6`` so the parser dispatches the v6 grammar.

    Raises :class:`PineUnsupportedFeatureError` (code ``PF003``) when the
    post-rewrite source still contains a v5-only construct that none of
    :data:`V5_REWRITES` handled. The error names the offending construct
    and points at the tracking URL where users can file a rewrite request.

    ``telemetry`` (E0.4): optional sink. When provided, the PF003 raise
    path calls ``telemetry.record_unsupported_feature("PF003")`` on the
    sink BEFORE raising. When ``None`` (default), no sink is touched.

    Idempotent for all ``bidi_safe=True`` rewrites (every shipped rewrite
    is). Calling ``migrate_v5_to_v6(migrate_v5_to_v6(src)[0])`` returns
    the same source as a single call — asserted in
    ``tests/unit/test_v5_migration.py::TestIdempotence``.
    """
    if not isinstance(source, str):
        raise TypeError(
            f"migrate_v5_to_v6: source must be str, not {type(source).__name__}"
        )

    log: list[str] = []
    out = source

    # Apply each rewrite in declaration order. Order matters for cascading
    # rewrites (see V5_REWRITES top-of-table comment).
    for rewrite in V5_REWRITES:
        out, n = rewrite.apply(out)
        if n > 0:
            log.append(f"{rewrite.name} ({n} substitution(s))")

    # #592 — strip `when=` from every strategy.* order-placement call. Lives
    # outside V5_REWRITES because it needs balanced-paren + string-literal
    # awareness that plain re.subn can't safely deliver. Runs AFTER the
    # regex pass so it sees any earlier normalisations (none today, but
    # the ordering is stable for future rewrites that touch strategy.*).
    out, when_strip_n = _strip_when_arg_from_strategy_calls(out)
    if when_strip_n > 0:
        log.append(
            f"strategy.*(when=…) → strategy.*(…) ({when_strip_n} substitution(s))"
        )

    # Last step: rewrite the pragma. Doing this AFTER the rewrites means a
    # malformed pragma rewrite won't leave us with a v6 grammar pointed at
    # un-migrated v5 source.
    out, pragma_n = _V5_PRAGMA_REWRITE.subn("//@version=6", out)
    if pragma_n > 0:
        log.append(f"//@version=5 → //@version=6 ({pragma_n} substitution(s))")

    # Sentinel pass: any unhandled v5 construct surfaces as PF003 here,
    # not as a parse error downstream — gives the user a meaningful
    # message + tracking URL. Run against a comment-stripped copy so
    # user-written `// iff()` style doc comments don't trigger.
    scan_target = _strip_line_comments(out)
    for unhandled_pat, feature in _UNHANDLED_V5_PATTERNS:
        if unhandled_pat.search(scan_target):
            if telemetry is not None:
                telemetry.record_unsupported_feature("PF003")
            raise PineUnsupportedFeatureError(
                message=(
                    f"PF003 v5 {feature} not yet migrated to v6: the regex "
                    f"rewrite in V5_REWRITES did not match this call (likely "
                    f"a multi-line or deeply-nested form). File a rewrite "
                    f"request at {PineUnsupportedFeatureError.tracking_url} "
                    f"with a minimal repro."
                )
            )

    return out, log
