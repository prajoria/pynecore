"""Codegen for Pine strategy(...) → @script.strategy decorator + strategy.* forwarders (bd-aeh).

Design ref: D5 §7.3 (Codegen C5) + §5.1 (script_type field).

Mirrors the existing @script.indicator emission shape. Compile-time only —
these tests verify that emitted Python compiles and contains the expected
tokens; runtime execution is out-of-scope (strategy runtime lives in
pynecore.lib.strategy and is exercised by that package's own tests).
"""
from __future__ import annotations

from pyne_compiler.compiler import compile_pine


def test_strategy_directive_emits_script_strategy_decorator():
    src = (
        "//@version=6\n"
        'strategy("MyStrat", overlay=true, initial_capital=10000)\n'
        "plot(close)\n"
    )
    result = compile_pine(src)
    assert result.script_type == "strategy"
    assert "@script.strategy(" in result.source
    assert "'MyStrat'" in result.source or '"MyStrat"' in result.source
    assert "overlay=True" in result.source
    assert "initial_capital=10000" in result.source


def test_strategy_entry_call_compiles():
    src = (
        "//@version=6\n"
        'strategy("s")\n'
        "if ta.crossover(close, ta.sma(close, 10))\n"
        '    strategy.entry("long", strategy.long, qty=1)\n'
    )
    result = compile_pine(src)
    assert "strategy.entry(" in result.source
    assert "strategy.long" in result.source


def test_strategy_exit_close_forwarders_compile():
    calls = (
        'strategy.exit("x", "long")',
        'strategy.close("long")',
        "strategy.close_all()",
        'strategy.cancel("id")',
        "strategy.cancel_all()",
    )
    for call in calls:
        src = f'//@version=6\nstrategy("s")\n{call}\n'
        r = compile_pine(src)
        head = call.split("(", 1)[0]
        assert head in r.source, f"missing {head} in emitted source"


def test_indicator_still_uses_script_indicator():
    src = '//@version=6\nindicator("i")\nplot(close)\n'
    r = compile_pine(src)
    assert r.script_type == "indicator"
    assert "@script.indicator(" in r.source
    assert "@script.strategy(" not in r.source
