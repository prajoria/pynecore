"""Pine v6 ``var TYPE name = init`` typed declarations (bd-27v7).

``var`` bindings are persistent: initialized on the FIRST bar only;
subsequent bars retain the previous value unless explicitly reassigned
via ``:=``. The typed form (`var TYPE name = init`) is REQUIRED when the
initializer is ``na`` (which lacks an inferable type) and is optional
otherwise.

Codegen strategy: emit ``name: Persistent[T] = value`` and rely on
PyneCore's PersistentTransformer to route the binding through the
per-scope state vector. For ``na`` initializers, we emit ``NA(T)`` so
the runtime binding starts as a typed missing value.
"""
from __future__ import annotations

from pyne_compiler.compiler import compile_pine


def _compile(src: str):
    return compile_pine(src, use_cache=False)


def test_var_float_na_compiles():
    """The exact failure from bd-ph0: ``var float trail_stop = na``."""
    src = (
        '//@version=6\n'
        'indicator("t", overlay=true)\n'
        'var float trail_stop = na\n'
        'if bar_index == 5\n'
        '    trail_stop := close\n'
        'plot(trail_stop)\n'
    )
    r = _compile(src)
    assert r.script_type == 'indicator'
    assert 'trail_stop: Persistent[float]' in r.source
    assert 'NA(float)' in r.source
    assert 'from pynecore.types import' in r.source


def test_var_int_zero_compiles():
    src = (
        '//@version=6\n'
        'indicator("t")\n'
        'var int counter = 0\n'
        'counter := counter + 1\n'
        'plot(counter)\n'
    )
    r = _compile(src)
    assert r.script_type == 'indicator'
    assert 'counter: Persistent[int] = 0' in r.source


def test_var_bool_false_compiles():
    src = (
        '//@version=6\n'
        'indicator("t")\n'
        'var bool armed = false\n'
        'if close > open\n'
        '    armed := true\n'
        'plot(armed ? 1 : 0)\n'
    )
    r = _compile(src)
    assert r.script_type == 'indicator'
    assert 'armed: Persistent[bool]' in r.source


def test_var_untyped_still_works():
    """Backward compat: untyped ``var`` with inferable initializer."""
    src = '//@version=6\nindicator("t")\nvar x = 0.0\nx := close\nplot(x)\n'
    r = _compile(src)
    assert r.script_type == 'indicator'
    # Untyped var still emits a plain assignment (no Persistent annotation).
    assert 'Persistent[' not in r.source


def test_var_in_strategy_body_compiles():
    """The bd-ph0 breakout_atr_trail scenario in strategy context (trimmed
    to strategy.* builtins already supported in Phase-1)."""
    src = (
        '//@version=6\n'
        'strategy("s", overlay=true)\n'
        'var float trail_stop = na\n'
        'atr = ta.atr(14)\n'
        'if ta.crossover(close, ta.highest(high, 20)[1])\n'
        '    strategy.entry("L", strategy.long)\n'
        '    trail_stop := close - atr * 2.0\n'
        'plot(trail_stop)\n'
    )
    r = _compile(src)
    assert r.script_type == 'strategy'
    assert 'trail_stop: Persistent[float]' in r.source
