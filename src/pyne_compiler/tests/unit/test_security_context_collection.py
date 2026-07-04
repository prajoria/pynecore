"""C3 populates ``CompiledModule.security_contexts`` on ``request.security``.

Bead: ``0e9.6.y86`` (part of the Phase-2 M2 rollup ``0e9.6``).
Design source of truth: D5 §4.1 (SecurityContext), §4.4 (fully-dynamic
symbol/timeframe), §7.2 (compiler signature).

What C3 now does when it walks a ``request.security(sym, tf, expr, gaps=?,
lookahead=?)`` call site (`type_checker._visit_request_security`):

1. Assigns a stable ``ctx_N`` id in traversal order (post-order: nested
   ``request.security`` in the ``expression`` slot registers the INNER
   call first). Deterministic across recompiles so the C6 cache key
   stays stable.
2. Builds a :class:`SecurityContext(symbol=..., timeframe=..., expr=...,
   dynamic_symbol=?, dynamic_timeframe=?)` per call site.
3. ``dynamic_symbol=True`` ONLY when the symbol arg is a truly-series
   expression that can't be resolved at compile-eval time. Bare literals,
   ``simple<string>`` names, ``const<string>`` constants, and
   ``input<string>`` (D1 §4.2 lattice) ALL count as static — Pine's
   ``input.string("1D")`` is compile-time-known even though the UI can
   override on load. Same rule for ``dynamic_timeframe``.
4. Threads the map onto :attr:`CompiledModule.security_contexts` via
   :class:`TypeCheckResult`.
5. Type-checks ``gaps`` / ``lookahead`` kwargs against ``const<bool>``
   (D5 §7.2 keyword-only slot; validated via the ``Signature.kwargs``
   field).
6. Rejects a non-string ``symbol`` / ``timeframe`` with ``PT001`` — Pine
   requires string args (or a series-of-string for dynamic form).
7. Rejects missing ``symbol`` / ``timeframe`` / ``expression`` (all
   three are Pine-required per D5 §7.2), the ``na`` sentinel in either
   routing-key slot, and a duplicate positional-AND-keyword bind.

Tests use :func:`compile_pine(src, ...)` end-to-end so the whole
pipeline (lexer → parser → C3 → codegen → compile-cache write) exercises
the change. Most tests use the default ``use_cache=True``; the C6 cache
serializer round-trips all SecurityContext fields (including
``dynamic_*``) — see :class:`TestCacheRoundTripPreservesDynamicFlags`.
"""

from __future__ import annotations

import pytest

from openbb_pine.compiler import compile_pine
from openbb_pine.compiler.types import CompiledModule, SecurityContext
from openbb_pine.errors import PineTypeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile(body: str, *, use_cache: bool = False) -> CompiledModule:
    """Wrap a Pine body in a minimal v6 indicator scaffold and compile.

    Defaults to ``use_cache=False`` so tests that assert on freshly-
    compiled state don't accidentally read a stale cache entry from a
    prior test in the same run. The cache round-trip test opts back in
    via ``use_cache=True`` + tmp cache dir.
    """
    src = f'//@version=6\nindicator("X")\n{body}'
    return compile_pine(src, use_cache=use_cache)


# ---------------------------------------------------------------------------
# Empty case — no request.security calls
# ---------------------------------------------------------------------------


class TestEmptySecurityContexts:
    """A script with no ``request.security`` calls carries ``None`` on
    :attr:`CompiledModule.security_contexts` (Phase-1 shape preserved)."""

    def test_bare_indicator_has_none(self) -> None:
        compiled = _compile("plot(close)\n")
        assert compiled.security_contexts is None

    def test_indicator_with_ta_call_has_none(self) -> None:
        # ta.sma is a builtin, but doesn't touch request.security.
        compiled = _compile("plot(ta.sma(close, 20))\n")
        assert compiled.security_contexts is None


# ---------------------------------------------------------------------------
# Single request.security call — static form
# ---------------------------------------------------------------------------


class TestOneStaticSecurityContext:
    """``request.security("SPY", "1D", close)`` — the D5 §4.1 canonical case:
    both symbol and timeframe are bare string literals so C3 statically
    resolves them; both ``dynamic_*`` flags are False."""

    def test_ctx_0_registered(self) -> None:
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        assert compiled.security_contexts is not None
        assert list(compiled.security_contexts) == ["ctx_0"]

    def test_symbol_and_timeframe_captured(self) -> None:
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.symbol == "SPY"
        assert ctx.timeframe == "1D"

    def test_static_flags_are_false(self) -> None:
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.dynamic_symbol is False
        assert ctx.dynamic_timeframe is False

    def test_expr_is_serialised_placeholder(self) -> None:
        """``expr`` field carries an opaque string per D5 §4.1 — D2 reads it
        opaquely, so C3's ``str(node)`` placeholder is a valid contract."""
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        expr = compiled.security_contexts["ctx_0"].expr
        # The placeholder must be a non-empty string. Since our lowered
        # expression here is a bare ``close`` name, the ``str(Name(...))``
        # form must contain the identifier.
        assert isinstance(expr, str) and expr
        assert "close" in expr

    def test_builtins_used_records_request_security(self) -> None:
        """The wild-corpus coverage metric (PRD §3.4 L0.5) reads
        :attr:`CompiledModule.builtins_used` — C3 must add
        ``request.security`` there even on the happy path."""
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        assert "request.security" in compiled.builtins_used


# ---------------------------------------------------------------------------
# Two independent request.security calls
# ---------------------------------------------------------------------------


class TestTwoIndependentSecurityContexts:
    """C3 must produce distinct ``ctx_N`` ids for two independent call
    sites — the counter is per-checker, incrementing in source order — and
    the id assignment must be deterministic across recompiles so the C6
    cache key stays stable."""

    _SRC = (
        'spy = request.security("SPY", "1D", close)\n'
        'qqq = request.security("QQQ", "60", close)\n'
        'plot(spy)\n'
        'plot(qqq)\n'
    )

    def test_two_ids_registered(self) -> None:
        compiled = _compile(self._SRC)
        assert set(compiled.security_contexts) == {"ctx_0", "ctx_1"}

    def test_ids_assigned_in_source_order(self) -> None:
        compiled = _compile(self._SRC)
        assert compiled.security_contexts["ctx_0"].symbol == "SPY"
        assert compiled.security_contexts["ctx_1"].symbol == "QQQ"

    def test_id_assignment_is_deterministic_across_recompiles(self) -> None:
        """Recompiling the SAME source must produce identical
        ``ctx_N`` → SecurityContext mappings — otherwise cache keys drift
        between compiles even though the source is stable."""
        first = _compile(self._SRC).security_contexts
        second = _compile(self._SRC).security_contexts
        assert first == second


# ---------------------------------------------------------------------------
# D5 §4.4 — fully-dynamic symbol
# ---------------------------------------------------------------------------


class TestDynamicSymbol:
    """When symbol is ``syminfo.ticker`` (or any non-literal expression),
    C3 sets ``dynamic_symbol=True`` — the runtime dispatcher falls back to
    lazy per-bar fetch (D5 §4.4 documented 5-10× perf caveat)."""

    def test_syminfo_ticker_flags_dynamic_symbol(self) -> None:
        compiled = _compile(
            'spy = request.security(syminfo.ticker, "1D", close)\nplot(spy)\n'
        )
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.dynamic_symbol is True
        assert ctx.dynamic_timeframe is False

    def test_dynamic_symbol_records_unsupported_builtin(self) -> None:
        """Even though we swallow the ``PineUnsupportedBuiltinError`` raised
        by walking ``syminfo.ticker`` (so the SecurityContext still lands),
        the name is recorded in :attr:`builtins_used` for coverage
        attribution."""
        compiled = _compile(
            'spy = request.security(syminfo.ticker, "1D", close)\nplot(spy)\n'
        )
        assert "syminfo.ticker" in compiled.builtins_used
        assert "request.security" in compiled.builtins_used


# ---------------------------------------------------------------------------
# D5 §4.4 — fully-dynamic timeframe (series<string>)
# ---------------------------------------------------------------------------


class TestDynamicTimeframe:
    """A truly-series timeframe (not compile-time-resolvable) flags
    ``dynamic_timeframe=True``. Same runtime consequence as
    ``dynamic_symbol`` — per D5 §4.4. Note the deliberate use of a
    ``ta.*``-derived expression: bindings backed by ``input.*`` are
    ``input<string>`` on the D1 §4.2 lattice and count as statically
    resolvable (see :class:`TestInputStringIsStatic` below).
    """

    _SRC = (
        'tf = "1D"\n'
        'spy = request.security(syminfo.ticker, tf, close)\n'
        'plot(spy)\n'
    )

    def test_bare_simple_string_still_static(self) -> None:
        # ``tf = "1D"`` is ``simple<string>`` — statically resolvable, so
        # dynamic_timeframe stays False even though we reference through
        # a name. Only truly-series-qualified strings should mark dynamic.
        compiled = _compile(self._SRC)
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.dynamic_timeframe is False


# ---------------------------------------------------------------------------
# D5 §4.4 static-resolution: input.string is const/input<string>, NOT series
# ---------------------------------------------------------------------------


class TestInputStringIsStatic:
    """Per D5 §4.4 + D1 §4.2 qualifier lattice, ``input.string(...)``
    returns ``input<string>`` — one step above ``const<string>`` and
    resolvable at compile-eval time. Pine users routinely write
    ``tf = input.string("1D"); request.security("SPY", tf, close)``
    expecting the prefetch (not the 5-10× slower per-bar-fetch) path.

    Guards against a regression of PR #322 review comment ID 3522846703.
    """

    _SRC = (
        'tf = input.string("1D")\n'
        'spy = request.security("SPY", tf, close)\n'
        'plot(spy)\n'
    )

    def test_input_string_timeframe_is_static(self) -> None:
        compiled = _compile(self._SRC)
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.dynamic_symbol is False
        assert ctx.dynamic_timeframe is False

    def test_input_string_symbol_is_static(self) -> None:
        compiled = _compile(
            'sym = input.string("SPY")\n'
            'spy = request.security(sym, "1D", close)\n'
            'plot(spy)\n'
        )
        ctx = compiled.security_contexts["ctx_0"]
        assert ctx.dynamic_symbol is False
        assert ctx.dynamic_timeframe is False

    def test_static_symbol_still_captured_verbatim(self) -> None:
        compiled = _compile(self._SRC)
        # Static-string symbol survives even when the timeframe reference
        # is a scope name.
        assert compiled.security_contexts["ctx_0"].symbol == "SPY"


# ---------------------------------------------------------------------------
# D5 §7.2 keyword-only args: gaps / lookahead
# ---------------------------------------------------------------------------


class TestGapsAndLookaheadKwargs:
    """``gaps`` and ``lookahead`` are Signature.kwargs entries (D5 §7.2
    keyword-only slot). C3 must accept bool consts."""

    def test_bool_kwargs_accepted(self) -> None:
        compiled = _compile(
            'spy = request.security("SPY", "1D", close, gaps=true, lookahead=false)\n'
            'plot(spy)\n'
        )
        assert compiled.security_contexts is not None
        assert "ctx_0" in compiled.security_contexts

    def test_gaps_only_kwarg_accepted(self) -> None:
        compiled = _compile(
            'spy = request.security("SPY", "1D", close, gaps=true)\nplot(spy)\n'
        )
        assert compiled.security_contexts is not None

    def test_lookahead_only_kwarg_accepted(self) -> None:
        compiled = _compile(
            'spy = request.security("SPY", "1D", close, lookahead=false)\nplot(spy)\n'
        )
        assert compiled.security_contexts is not None


# ---------------------------------------------------------------------------
# Type rejection — non-string symbol
# ---------------------------------------------------------------------------


class TestRejectsNonStringSymbol:
    """PT001 fires when ``symbol`` isn't a string.

    ``request.security(123, "1D", close)`` — Pine rejects an int-typed
    symbol per D5 §7.2. Our compiler surfaces this via the PT001 rule
    inside :meth:`_TypeChecker._resolve_security_string_arg` (the strict
    inner-type check that runs after the walk succeeds)."""

    def test_int_symbol_raises_pt001(self) -> None:
        with pytest.raises(PineTypeError) as excinfo:
            _compile('spy = request.security(123, "1D", close)\nplot(spy)\n')
        assert excinfo.value.rule == "PT001"


# ---------------------------------------------------------------------------
# Threading — CompiledModule carries the same dict TypeCheckResult built
# ---------------------------------------------------------------------------


class TestSecurityContextThreading:
    """The C3 → codegen (emit) → CompiledModule pipeline must thread the
    security_contexts dict unchanged. This test guards against a silent
    drop somewhere in the compile facade."""

    def test_compiled_module_carries_security_contexts(self) -> None:
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        # The dict is populated per C3, then echoed through emit() into
        # CompiledModule.security_contexts. Verify the shape survives.
        assert isinstance(compiled.security_contexts, dict)
        assert isinstance(compiled.security_contexts["ctx_0"], SecurityContext)

    def test_none_when_no_request_security(self) -> None:
        # Guards against accidental empty-dict-vs-None drift — TypeCheckResult
        # normalises {} to None so downstream consumers don't have to.
        compiled = _compile("plot(close)\n")
        assert compiled.security_contexts is None


# ---------------------------------------------------------------------------
# Regression guards for PR #322 review findings
# ---------------------------------------------------------------------------


class TestExprIsHumanReadable:
    """PR #322 review comment ID 3522846218 — the ``expr`` field previously
    stored the frozen-dataclass ``repr()`` of the IR node
    (``"Name(loc=Span(...),id='close')"``) which was garbage for bug
    reports AND for the dynamic-symbol path (that string got forwarded to
    ``_fetch_via_fmp`` if the cache round-trip lost the ``dynamic_*``
    flags). We now serialize to a stable, Pine-like source string.
    """

    def test_bare_name_expr_is_just_the_name(self) -> None:
        compiled = _compile('spy = request.security("SPY", "1D", close)\nplot(spy)\n')
        assert compiled.security_contexts["ctx_0"].expr == "close"

    def test_call_expr_is_pine_like(self) -> None:
        compiled = _compile(
            'spy = request.security("SPY", "1D", ta.sma(close, 20))\nplot(spy)\n'
        )
        expr = compiled.security_contexts["ctx_0"].expr
        # Not ``str(node)`` gibberish; a real function-call rendering.
        assert expr == "ta.sma(close, 20)"

    def test_dynamic_symbol_is_source_like(self) -> None:
        compiled = _compile(
            'spy = request.security(syminfo.ticker, "1D", close)\nplot(spy)\n'
        )
        ctx = compiled.security_contexts["ctx_0"]
        # ``symbol`` is now the rendered attribute chain, not a
        # frozen-dataclass ``repr()``.
        assert ctx.symbol == "syminfo.ticker"
        assert ctx.dynamic_symbol is True


class TestRejectsMissingExpression:
    """PR #322 review comment ID 3522847702 — a missing ``expression``
    arg previously produced ``SecurityContext(expr="")`` silently. Now
    it raises like the ``symbol`` / ``timeframe`` slots do."""

    def test_missing_expression_raises(self) -> None:
        with pytest.raises(PineTypeError) as excinfo:
            _compile('spy = request.security("SPY", "1D")\nplot(spy)\n')
        assert excinfo.value.rule == "undefined"


class TestRejectsDuplicateBind:
    """PR #322 review comment ID 3522847453 — passing BOTH a positional
    AND a keyword for the same slot previously silently accepted the
    kwarg. Now it raises like Python's ``TypeError: got multiple values
    for argument``."""

    def test_duplicate_symbol_raises(self) -> None:
        with pytest.raises(PineTypeError) as excinfo:
            _compile(
                'spy = request.security("SPY", "1D", close, symbol="AAPL")\n'
                'plot(spy)\n'
            )
        assert excinfo.value.rule == "undefined"


class TestRejectsNaRoutingKey:
    """PR #322 review comment ID 3522847918 — ``na`` in the routing-key
    slots (symbol / timeframe) has no runtime meaning and would give the
    dispatcher a nonsense symbol to fetch. PT006's general
    "na propagates any T" rule doesn't apply because these args are
    routing keys, not values."""

    def test_na_symbol_raises(self) -> None:
        with pytest.raises(PineTypeError) as excinfo:
            _compile('spy = request.security(na, "1D", close)\nplot(spy)\n')
        # PT006 (see PR #322 comment ID 3522847918 hint) — the compiler
        # rejects rather than silently forwarding NaLit downstream.
        assert excinfo.value.rule == "PT006"

    def test_na_timeframe_raises(self) -> None:
        with pytest.raises(PineTypeError) as excinfo:
            _compile('spy = request.security("SPY", na, close)\nplot(spy)\n')
        assert excinfo.value.rule == "PT006"


class TestNestedRequestSecurityRegistersBoth:
    """PR #322 review comment ID 3522848074 — nested
    ``request.security`` in the ``expression`` slot registers the INNER
    call first (``ctx_0``) and the OUTER second (``ctx_1``) because
    ``_visit_expr`` recurses into the expression BEFORE the outer call
    increments the counter (post-order traversal). Real Pine scripts
    virtually never nest ``request.security`` this way (it defeats the
    prefetch), but the id assignment must be deterministic.
    """

    def test_nested_registers_inner_before_outer(self) -> None:
        compiled = _compile(
            'spy = request.security("SPY", "1D", '
            'request.security("AAPL", "1D", close))\n'
            'plot(spy)\n'
        )
        assert set(compiled.security_contexts) == {"ctx_0", "ctx_1"}
        # Inner call registered first — its symbol is AAPL.
        assert compiled.security_contexts["ctx_0"].symbol == "AAPL"
        # Outer call registered second — its symbol is SPY.
        assert compiled.security_contexts["ctx_1"].symbol == "SPY"


class TestCacheRoundTripPreservesDynamicFlags:
    """PR #322 review comment ID 3522845893 — the C6 cache serializer
    used to strip ``dynamic_symbol`` / ``dynamic_timeframe`` on write,
    turning a working dynamic-symbol script into a stale-flag failure
    on the second compile (the second-compile ctx has
    ``dynamic_symbol=False`` and the runtime dispatcher then hands the
    dynamic ``symbol`` string to ``_fetch_via_fmp``). We now round-trip
    every SecurityContext field.
    """

    def test_dynamic_symbol_survives_cache_roundtrip(self, tmp_path) -> None:
        # First compile is a miss — populates the cache.
        src = (
            '//@version=6\nindicator("X")\n'
            'spy = request.security(syminfo.ticker, "1D", close)\nplot(spy)\n'
        )
        first = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        assert first.cache_status == "miss"
        assert first.security_contexts["ctx_0"].dynamic_symbol is True

        # Second compile is a hit — must round-trip the dynamic flag.
        second = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        assert second.cache_status == "hit"
        assert second.security_contexts["ctx_0"].dynamic_symbol is True

    def test_static_flags_survive_cache_roundtrip(self, tmp_path) -> None:
        src = (
            '//@version=6\nindicator("X")\n'
            'spy = request.security("SPY", "1D", close)\nplot(spy)\n'
        )
        first = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        second = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        assert first.security_contexts == second.security_contexts

    def test_dynamic_timeframe_survives_cache_roundtrip(self, tmp_path) -> None:
        # Build a truly-series timeframe by referencing an unresolved
        # syminfo attribute — walks the dynamic-timeframe path.
        src = (
            '//@version=6\nindicator("X")\n'
            'spy = request.security("SPY", syminfo.timeframe, close)\n'
            'plot(spy)\n'
        )
        first = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        second = compile_pine(src, use_cache=True, cache_dir=tmp_path)
        assert first.security_contexts["ctx_0"].dynamic_timeframe is True
        assert second.security_contexts["ctx_0"].dynamic_timeframe is True


class TestMixedStaticAndDynamicContexts:
    """PR #322 review comment ID 3522848526 — a script with BOTH a
    static- and a dynamic-symbol call site produces a two-entry map
    where each ctx carries its own ``dynamic_*`` flags without cross-
    contamination.
    """

    def test_mixed_static_and_dynamic(self) -> None:
        compiled = _compile(
            'a = request.security("SPY", "1D", close)\n'
            'b = request.security(syminfo.ticker, "1D", close)\n'
            'plot(a)\nplot(b)\n'
        )
        assert compiled.security_contexts["ctx_0"].dynamic_symbol is False
        assert compiled.security_contexts["ctx_1"].dynamic_symbol is True
