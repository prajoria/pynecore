"""Grammar coverage for the ``strategy`` surface — Phase 2 M2, bead pxc.

Design reference: ``docs/designs/openbb-pine/D5-strategy-engine.md`` §7.1.

Bead **pxc** ("Grammar — strategy(...) top-level decl + strategy.entry/exit/
close/close_all/cancel/cancel_all + strategy.long/short attribute access")
tightens grammar coverage around the ``strategy`` namespace.

Two related fixtures the C2 parser must accept — the top-level directive
was already there before this bead, but every *value-position* use of
``strategy.*`` was rejected because the token adapter reclassifies the
bareword ``strategy`` → ``KW_STRATEGY`` unconditionally, and the grammar
only permitted ``KW_STRATEGY`` as the leading terminal of the
``strategy_directive`` rule. The bead's grammar edit admits ``KW_STRATEGY``
as a ``primary`` expression (yielding ``ir.Name(id="strategy")``) so the
postfix chain composes normally.

What this file guarantees
-------------------------

1. The top-level ``strategy("Title", ...)`` directive parses with a
   realistic kwarg surface (including nested ``strategy.percent_of_equity``,
   ``strategy.commission.percent`` attribute expressions used AS kwarg
   values), producing a ``ScriptDirective(kind="strategy", ...)``.
2. ``strategy.entry / exit / close / close_all / cancel / cancel_all`` all
   parse as ``CallExpr(func=Attribute(value=Name("strategy"), attr="..."))``
   in body position.
3. ``strategy.long`` and ``strategy.short`` parse as bare ``Attribute`` nodes
   suitable for use as expression values (assigned to a variable, passed
   as a positional or keyword argument to another call).

Codegen for ``strategy(...)`` is intentionally deferred to a later bead
(``PineUnsupportedFeatureError`` PF010 at ``codegen.py`` line ~423). This
file exercises the pipeline through parse + type-check ONLY via
``compile_pine_to_program``, never through the emitter — the parser and
the type checker are the surface bead pxc owns.
"""

from __future__ import annotations

import pytest

from openbb_pine.compiler import compile_pine_to_program, ir
from openbb_pine.compiler.lexer import tokenize
from openbb_pine.compiler.parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_strategy_v6(body: str = "") -> ir.Program:
    """Parse a minimal v6 strategy scaffold with an optional body suffix.

    ``type_check=False`` deliberately — the parser is what bead pxc owns,
    and the C3 checker still fails soft on some strategy builtins (a
    separate future bead). Skipping C3 lets us assert grammar coverage in
    isolation.
    """
    src = f'//@version=6\nstrategy("Test", overlay=true)\n{body}'
    return compile_pine_to_program(src, type_check=False)


# ---------------------------------------------------------------------------
# 1. Top-level strategy directive — bare + realistic kwargs
# ---------------------------------------------------------------------------


class TestStrategyDirective:
    def test_bare_directive(self) -> None:
        """``strategy("Test", overlay=true)`` parses to a
        ``ScriptDirective(kind="strategy", title="Test", overlay=True)``.
        """
        prog = compile_pine_to_program(
            '//@version=6\nstrategy("Test", overlay=true)\n',
            type_check=False,
        )
        d = prog.directive
        assert isinstance(d, ir.ScriptDirective)
        assert d.kind == "strategy"
        assert d.title == "Test"
        assert d.overlay is True

    def test_rich_directive_with_nested_strategy_kwargs(self) -> None:
        """Realistic Pine strategy directive with many kwargs including
        nested ``strategy.percent_of_equity`` and ``strategy.commission.percent``
        attribute expressions in value position. This is the case that was
        broken pre-pxc: ``strategy=KW_STRATEGY`` inside a kwarg value.
        """
        src = (
            '//@version=6\n'
            'strategy("Test", overlay=true, initial_capital=100000, '
            'default_qty_type=strategy.percent_of_equity, default_qty_value=100, '
            'commission_type=strategy.commission.percent, commission_value=0.1)\n'
        )
        prog = compile_pine_to_program(src, type_check=False)
        d = prog.directive
        assert isinstance(d, ir.ScriptDirective)
        assert d.kind == "strategy"
        assert d.title == "Test"
        assert d.overlay is True

        # Verify the nested strategy.percent_of_equity kwarg value parses as
        # an Attribute of a Name("strategy") — this is the surface the bead
        # unlocks. Look up by name because ``arguments`` includes the
        # positional title too, and we don't care about ordering here.
        by_name = {a.name: a for a in d.arguments if a.name is not None}
        assert "default_qty_type" in by_name
        val = by_name["default_qty_type"].value
        assert isinstance(val, ir.Attribute)
        assert val.attr == "percent_of_equity"
        assert isinstance(val.value, ir.Name)
        assert val.value.id == "strategy"

        # And commission_type=strategy.commission.percent — a two-level chain.
        assert "commission_type" in by_name
        cval = by_name["commission_type"].value
        assert isinstance(cval, ir.Attribute)
        assert cval.attr == "percent"
        # Inner: strategy.commission (Attribute over Name("strategy"))
        assert isinstance(cval.value, ir.Attribute)
        assert cval.value.attr == "commission"
        assert isinstance(cval.value.value, ir.Name)
        assert cval.value.value.id == "strategy"


# ---------------------------------------------------------------------------
# 2. strategy.entry / exit / close / close_all / cancel / cancel_all
# ---------------------------------------------------------------------------


class TestStrategyBodyCalls:
    """Body-position ``strategy.<method>(...)`` calls must parse as
    ``CallExpr(func=Attribute(Name("strategy"), attr="<method>"))``.

    Each case exercises the ``postfix: primary (attribute_tail | call_tail)*``
    chain hanging off a ``KW_STRATEGY`` primary. We assert IR shape (not
    just "no exception") so a future accidental grammar rewrite can't
    silently regress ``strategy.foo`` to some other node kind.
    """

    def _strategy_call(self, prog: ir.Program) -> ir.CallExpr:
        """Return the single ExprStmt-wrapped CallExpr from the body."""
        assert len(prog.body) == 1
        stmt = prog.body[0]
        assert isinstance(stmt, ir.ExprStmt)
        assert isinstance(stmt.expr, ir.CallExpr)
        return stmt.expr

    def _assert_strategy_attr(self, call: ir.CallExpr, attr: str) -> None:
        """Assert ``call.func`` is ``Attribute(Name("strategy"), attr=...)``."""
        assert isinstance(call.func, ir.Attribute)
        assert call.func.attr == attr
        assert isinstance(call.func.value, ir.Name)
        assert call.func.value.id == "strategy"

    def test_strategy_entry(self) -> None:
        """``strategy.entry("long", strategy.long, qty=1)`` — three-arg call
        where the second positional is itself a ``strategy.*`` attribute
        access. This is the canonical shape from PyneCore docs and PineV6
        strategy tutorials.
        """
        prog = _parse_strategy_v6('strategy.entry("long", strategy.long, qty=1)\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "entry")
        # Three arguments: positional "long", positional strategy.long, kwarg qty=1
        assert len(call.args) == 3
        assert call.args[0].name is None
        assert isinstance(call.args[0].value, ir.StrLit)
        assert call.args[0].value.value == "long"
        # Second positional is Attribute(Name("strategy"), attr="long")
        assert call.args[1].name is None
        assert isinstance(call.args[1].value, ir.Attribute)
        assert call.args[1].value.attr == "long"
        assert isinstance(call.args[1].value.value, ir.Name)
        assert call.args[1].value.value.id == "strategy"
        # Third is kwarg qty=1
        assert call.args[2].name == "qty"
        assert isinstance(call.args[2].value, ir.IntLit)
        assert call.args[2].value.value == 1

    def test_strategy_exit(self) -> None:
        """``strategy.exit("exit", from_entry="long", stop=90)`` — mix of
        positional string + string kwarg + numeric kwarg.
        """
        prog = _parse_strategy_v6('strategy.exit("exit", from_entry="long", stop=90)\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "exit")
        assert len(call.args) == 3
        by_name = {a.name: a for a in call.args}
        assert None in by_name  # positional
        assert isinstance(by_name[None].value, ir.StrLit)
        assert by_name[None].value.value == "exit"
        assert "from_entry" in by_name
        assert isinstance(by_name["from_entry"].value, ir.StrLit)
        assert by_name["from_entry"].value.value == "long"
        assert "stop" in by_name
        assert isinstance(by_name["stop"].value, ir.IntLit)
        assert by_name["stop"].value.value == 90

    def test_strategy_close(self) -> None:
        prog = _parse_strategy_v6('strategy.close("long")\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "close")
        assert len(call.args) == 1
        assert isinstance(call.args[0].value, ir.StrLit)
        assert call.args[0].value.value == "long"

    def test_strategy_close_all(self) -> None:
        prog = _parse_strategy_v6('strategy.close_all()\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "close_all")
        assert len(call.args) == 0

    def test_strategy_cancel(self) -> None:
        prog = _parse_strategy_v6('strategy.cancel("id")\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "cancel")
        assert len(call.args) == 1
        assert isinstance(call.args[0].value, ir.StrLit)
        assert call.args[0].value.value == "id"

    def test_strategy_cancel_all(self) -> None:
        prog = _parse_strategy_v6('strategy.cancel_all()\n')
        call = self._strategy_call(prog)
        self._assert_strategy_attr(call, "cancel_all")
        assert len(call.args) == 0


# ---------------------------------------------------------------------------
# 3. strategy.long / strategy.short — attribute-access as expression value
# ---------------------------------------------------------------------------


class TestStrategyDirectionValues:
    """``strategy.long`` and ``strategy.short`` must parse as bare
    ``Attribute(Name("strategy"), attr="...")`` in value position — as the
    RHS of an assignment, as a positional call argument, or as a kwarg
    value.

    In real Pine these resolve at runtime to the ``direction`` enum values.
    At parse time they are just attribute-chain expressions.
    """

    @pytest.mark.parametrize("direction", ["long", "short"])
    def test_strategy_direction_as_assign_rhs(self, direction: str) -> None:
        """``d = strategy.long`` / ``d = strategy.short``."""
        prog = _parse_strategy_v6(f'd = strategy.{direction}\n')
        assert len(prog.body) == 1
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        assert isinstance(a.target, ir.Name) and a.target.id == "d"
        val = a.value
        assert isinstance(val, ir.Attribute)
        assert val.attr == direction
        assert isinstance(val.value, ir.Name)
        assert val.value.id == "strategy"

    @pytest.mark.parametrize("direction", ["long", "short"])
    def test_strategy_direction_as_call_arg(self, direction: str) -> None:
        """``strategy.entry("id", direction=strategy.long)`` — kwarg value.

        Distinct from the positional case in :meth:`test_strategy_entry`
        because kwarg parsing hits ``kwarg: NAME ASSIGN expression`` which
        is a different reduction than a bare positional expression.
        """
        prog = _parse_strategy_v6(
            f'strategy.entry("id", direction=strategy.{direction})\n'
        )
        assert len(prog.body) == 1
        stmt = prog.body[0]
        assert isinstance(stmt, ir.ExprStmt)
        call = stmt.expr
        assert isinstance(call, ir.CallExpr)
        # find kwarg direction=strategy.<dir>
        by_name = {a.name: a for a in call.args}
        assert "direction" in by_name
        val = by_name["direction"].value
        assert isinstance(val, ir.Attribute)
        assert val.attr == direction
        assert isinstance(val.value, ir.Name)
        assert val.value.id == "strategy"


# ---------------------------------------------------------------------------
# 4. Cross-version — v5 grammar (a v6 copy today) admits the same surface
# ---------------------------------------------------------------------------


class TestStrategyV5:
    """The v5 grammar is a byte-for-byte copy of v6 with a different header
    (bead C7 will diverge it later). Bead pxc mirrors the ``KW_STRATEGY``
    primary carve-out into v5 so we don't regress once C7 lands and starts
    running v5 fixtures against v5 grammar directly.

    We bypass ``compile_pine_to_program`` here because its facade auto-migrates
    v5 sources to v6 before parsing (dropping the pragma). ``parse(tokens,
    pine_version=5)`` exercises the v5 grammar directly, mirroring the
    approach in ``test_parser.py::TestPineV5Placeholder``.
    """

    def test_v5_strategy_entry(self) -> None:
        src = (
            '//@version=5\n'
            'strategy("Test", overlay=true)\n'
            'strategy.entry("long", strategy.long, qty=1)\n'
        )
        prog = parse(tokenize(src), pine_version=5)
        assert prog.version == 5
        assert prog.directive.kind == "strategy"
        assert len(prog.body) == 1
        stmt = prog.body[0]
        assert isinstance(stmt, ir.ExprStmt)
        assert isinstance(stmt.expr, ir.CallExpr)
        assert isinstance(stmt.expr.func, ir.Attribute)
        assert stmt.expr.func.attr == "entry"
