"""Codegen for Pine ``request.security(sym, tf, expr)`` (bd-god / D5 §7.3).

The runtime hook :func:`openbb_pine.runtime.security_hook.install_secondaries_hook`
patches ``pynecore.lib.request.security`` directly, so the emitted call-site
stays as ``request.security(...)`` and reads through the patch at bar time.
What codegen must add is a module-level ``__security_contexts__`` dict
listing every ``(symbol, timeframe, expr)`` context C3 collected — the
prefetch dispatcher (bd-c1x) reads this via
:attr:`CompiledModule.security_contexts` at pre-run time; the module-level
constant makes the same metadata visible to any consumer inspecting the
compiled module (e.g. PyneCore's ``ScriptRunner`` also scans for
``__security_contexts__`` — see ``script_runner.py`` L405).

Clean-room: I have not viewed TradingView or PyneComp source code.
"""
from __future__ import annotations

from pyne_compiler.compiler import compile_pine


def test_request_security_emits_module_level_contexts_constant():
    src = (
        "//@version=6\n"
        'indicator("s")\n'
        'htf_close = request.security(syminfo.tickerid, "D", close)\n'
        "plot(htf_close)\n"
    )
    result = compile_pine(src)
    # Call-site preserved — runtime hook patches request.security itself.
    assert "request.security" in result.source
    # Module-level constant emitted so downstream consumers can inspect
    # what contexts the script needs prefetched.
    assert "__security_contexts__" in result.source
    # CompiledModule carries the same map (already populated by C3).
    assert result.security_contexts is not None
    assert "ctx_0" in result.security_contexts


def test_request_security_lists_every_distinct_context():
    src = (
        "//@version=6\n"
        'indicator("s")\n'
        'a = request.security(syminfo.tickerid, "D", close)\n'
        'b = request.security("BTCUSD", "60", volume)\n'
        "plot(a)\n"
    )
    result = compile_pine(src)
    assert "__security_contexts__" in result.source
    assert result.security_contexts is not None
    assert len(result.security_contexts) == 2
    # Both context ids appear in the emitted source.
    assert "ctx_0" in result.source
    assert "ctx_1" in result.source


def test_no_request_security_no_contexts_constant():
    src = (
        "//@version=6\n"
        'indicator("s")\n'
        "plot(close)\n"
    )
    result = compile_pine(src)
    # No request.security → don't emit the module constant at all.
    assert "__security_contexts__" not in result.source
    # And security_contexts is either None or empty.
    assert not result.security_contexts


def test_security_contexts_constant_carries_symbol_timeframe_expr():
    src = (
        "//@version=6\n"
        'indicator("s")\n'
        'x = request.security("AAPL", "1D", close)\n'
        "plot(x)\n"
    )
    result = compile_pine(src)
    src_text = result.source
    # Fields for the operator + prefetcher to introspect.
    assert "'symbol'" in src_text or '"symbol"' in src_text
    assert "'timeframe'" in src_text or '"timeframe"' in src_text
    assert "'expr'" in src_text or '"expr"' in src_text
    assert "'AAPL'" in src_text or '"AAPL"' in src_text
    assert "'1D'" in src_text or '"1D"' in src_text


def test_emitted_module_is_still_compilable_python():
    """__security_contexts__ emission must not break ast.unparse/compile."""
    import ast as _ast
    src = (
        "//@version=6\n"
        'indicator("s")\n'
        'x = request.security(syminfo.tickerid, "D", close)\n'
        "plot(x)\n"
    )
    result = compile_pine(src)
    # If the allowlist gate had rejected __security_contexts__ we'd have
    # raised inside compile_pine; getting here means it passed. Belt-and-
    # braces: the produced source must also compile as a standalone module.
    _ast.parse(result.source)
    compile(result.source, "<emitted>", "exec")
