"""strategy.position_size read (bd-9cae).

Pine v6 read-only variable exposing current position size:
  0 = flat
  positive = long (contracts held)
  negative = short
"""
from __future__ import annotations

from pyne_compiler.compiler import compile_pine


def test_position_size_compiles():
    """The bd-ph0 bb_squeeze scenario: gate exit on being in position."""
    src = '''//@version=6
strategy("s", overlay=true)
if ta.crossover(close, ta.sma(close, 20))
    strategy.entry("L", strategy.long)
if strategy.position_size > 0 and close < ta.sma(close, 20)
    strategy.close("L")
plot(close)
'''
    r = compile_pine(src, use_cache=False)
    assert r.script_type == 'strategy'
    assert 'position_size' in r.source


def test_position_size_in_conditional():
    src = '//@version=6\nstrategy("s")\nif strategy.position_size != 0\n    strategy.close_all()\nplot(close)\n'
    r = compile_pine(src, use_cache=False)
    assert r.script_type == 'strategy'


def test_position_size_arithmetic():
    """position_size supports numeric operations."""
    src = '''//@version=6
strategy("s")
size = strategy.position_size
if math.abs(size) > 0.5
    strategy.close_all()
plot(close)
'''
    r = compile_pine(src, use_cache=False)
    assert r.script_type == 'strategy'


def test_position_size_signature_present():
    """Registry check: strategy.position_size has a Signature with IMPLEMENTED notes."""
    from pyne_compiler.compiler.builtin_signatures import lookup
    sig = lookup('strategy.position_size')
    assert sig is not None, "strategy.position_size missing from BUILTIN_SIGNATURES"
    assert sig.notes == "IMPLEMENTED", f"expected IMPLEMENTED, got notes={sig.notes!r}"
