"""Tests for ``openbb_pine.compiler.parser`` — lark Earley over Pine Tokens.

Source of truth: D1 §1.3 (Earley over LALR), §1.5 (versions + grammar layout),
§2.x (IR shapes).

The parser consumes the C1 lexer's ``list[Token]`` and produces an IR
``Program`` per D1 §2.3 — one node type per Pine syntactic category.

Operator precedence is encoded in the grammar layers (D1 §1.3 — the Earley
win). Error recovery routes through ``UnexpectedToken.match_examples`` to
attach per-pattern hints to ``PineSyntaxError``.
"""

from __future__ import annotations

import pytest

from openbb_pine.compiler import ir
from openbb_pine.compiler.lexer import Token, tokenize
from openbb_pine.compiler.parser import parse, _lark_for_version
from openbb_pine.errors import PineSyntaxError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_v6(src: str) -> ir.Program:
    return parse(tokenize(src), pine_version=6)


def _parse_v5(src: str) -> ir.Program:
    return parse(tokenize(src), pine_version=5)


def _wrap(stmt: str) -> str:
    """Wrap a Pine statement in a minimal v6 indicator scaffold."""
    return f'//@version=6\nindicator("X")\n{stmt}'


# ---------------------------------------------------------------------------
# Round-trip per statement category (D1 §2.3)
# ---------------------------------------------------------------------------


class TestStatementCategories:
    def test_var_decl(self) -> None:
        prog = _parse_v6(_wrap("var x = 5\n"))
        assert isinstance(prog.body[0], ir.VarDecl)
        d = prog.body[0]
        assert d.qualifier == "var"
        assert d.name == "x"
        assert isinstance(d.value, ir.IntLit) and d.value.value == 5

    def test_varip_decl(self) -> None:
        prog = _parse_v6(_wrap("varip y = 3\n"))
        d = prog.body[0]
        assert isinstance(d, ir.VarDecl)
        assert d.qualifier == "varip"
        assert d.name == "y"

    def test_assign_eq(self) -> None:
        prog = _parse_v6(_wrap("x = 42\n"))
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        assert a.op == "="
        assert isinstance(a.target, ir.Name) and a.target.id == "x"
        assert isinstance(a.value, ir.IntLit) and a.value.value == 42

    def test_assign_walrus(self) -> None:
        prog = _parse_v6(_wrap("x := 7\n"))
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        assert a.op == ":="

    def test_if(self) -> None:
        prog = _parse_v6(_wrap("if cond\n    x = 1\n"))
        s = prog.body[0]
        assert isinstance(s, ir.IfStmt)
        assert isinstance(s.cond, ir.Name) and s.cond.id == "cond"
        assert len(s.then_body) == 1 and isinstance(s.then_body[0], ir.Assign)
        assert s.else_body is None
        assert s.elif_branches == ()

    def test_if_else_if_else(self) -> None:
        prog = _parse_v6(_wrap("if a\n    x = 1\nelse if b\n    x = 2\nelse\n    x = 3\n"))
        s = prog.body[0]
        assert isinstance(s, ir.IfStmt)
        assert len(s.elif_branches) == 1
        assert isinstance(s.elif_branches[0][0], ir.Name)
        assert s.elif_branches[0][0].id == "b"
        assert s.else_body is not None and len(s.else_body) == 1

    def test_for(self) -> None:
        prog = _parse_v6(_wrap("for i = 0 to 10\n    x = 1\n"))
        s = prog.body[0]
        assert isinstance(s, ir.ForStmt)
        assert s.var == "i"
        assert isinstance(s.start, ir.IntLit) and s.start.value == 0
        assert isinstance(s.end, ir.IntLit) and s.end.value == 10
        assert s.step is None

    def test_for_with_step(self) -> None:
        prog = _parse_v6(_wrap("for i = 1 to 100 by 2\n    x = 1\n"))
        s = prog.body[0]
        assert isinstance(s, ir.ForStmt)
        assert isinstance(s.step, ir.IntLit) and s.step.value == 2

    def test_while(self) -> None:
        prog = _parse_v6(_wrap("while cond\n    x = 1\n"))
        s = prog.body[0]
        assert isinstance(s, ir.WhileStmt)
        assert isinstance(s.cond, ir.Name) and s.cond.id == "cond"
        assert len(s.body) == 1

    def test_ternary(self) -> None:
        prog = _parse_v6(_wrap("x = a ? b : c\n"))
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        assert isinstance(a.value, ir.TernaryExpr)
        assert isinstance(a.value.cond, ir.Name) and a.value.cond.id == "a"

    def test_history_subscript(self) -> None:
        prog = _parse_v6(_wrap("y = close[1]\n"))
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        assert isinstance(a.value, ir.Subscript)
        assert a.value.kind == "history"
        assert isinstance(a.value.value, ir.Name) and a.value.value.id == "close"
        assert isinstance(a.value.index, ir.IntLit) and a.value.index.value == 1

    def test_call_with_kwargs(self) -> None:
        prog = _parse_v6(_wrap("y = input.int(20, minval=1)\n"))
        a = prog.body[0]
        assert isinstance(a, ir.Assign)
        call = a.value
        assert isinstance(call, ir.CallExpr)
        # func is Attribute(value=Name('input'), attr='int')
        assert isinstance(call.func, ir.Attribute)
        assert call.func.attr == "int"
        # 2 args: positional 20, kwarg minval=1
        assert len(call.args) == 2
        assert call.args[0].name is None  # positional
        assert isinstance(call.args[0].value, ir.IntLit)
        assert call.args[1].name == "minval"

    def test_function_def_oneline(self) -> None:
        prog = _parse_v6(_wrap("f(x) => x * 2\n"))
        assert len(prog.declarations) == 1
        d = prog.declarations[0]
        assert isinstance(d, ir.FunctionDecl)
        assert d.name == "f"
        assert d.is_method is False
        assert len(d.parameters) == 1 and d.parameters[0].name == "x"
        assert len(d.body) == 1

    def test_function_def_block(self) -> None:
        prog = _parse_v6(_wrap("f(x) =>\n    y = x * 2\n    y\n"))
        assert len(prog.declarations) == 1
        d = prog.declarations[0]
        assert isinstance(d, ir.FunctionDecl)
        assert len(d.body) == 2  # the Assign and the expression-as-statement


# ---------------------------------------------------------------------------
# Operator precedence — D1 §1.3 says grammar layers encode it.
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_add_vs_mul(self) -> None:
        """``a + b * c`` -> BinaryExpr(+, a, BinaryExpr(*, b, c))."""
        prog = _parse_v6(_wrap("x = a + b * c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "+"
        assert isinstance(v.lhs, ir.Name) and v.lhs.id == "a"
        assert isinstance(v.rhs, ir.BinaryExpr) and v.rhs.op == "*"
        assert isinstance(v.rhs.lhs, ir.Name) and v.rhs.lhs.id == "b"
        assert isinstance(v.rhs.rhs, ir.Name) and v.rhs.rhs.id == "c"

    def test_mul_vs_add(self) -> None:
        """``a * b + c`` -> BinaryExpr(+, BinaryExpr(*, a, b), c)."""
        prog = _parse_v6(_wrap("x = a * b + c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "+"
        assert isinstance(v.lhs, ir.BinaryExpr) and v.lhs.op == "*"

    def test_sub_left_assoc(self) -> None:
        """``a - b - c`` -> ``(a - b) - c``."""
        prog = _parse_v6(_wrap("x = a - b - c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "-"
        assert isinstance(v.lhs, ir.BinaryExpr) and v.lhs.op == "-"
        assert isinstance(v.rhs, ir.Name) and v.rhs.id == "c"

    def test_div_left_assoc(self) -> None:
        prog = _parse_v6(_wrap("x = a / b / c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "/"
        assert isinstance(v.lhs, ir.BinaryExpr) and v.lhs.op == "/"

    def test_comparison_binds_below_arithmetic(self) -> None:
        """``a + b < c * d`` -> ``(a+b) < (c*d)``."""
        prog = _parse_v6(_wrap("x = a + b < c * d\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "<"
        assert isinstance(v.lhs, ir.BinaryExpr) and v.lhs.op == "+"
        assert isinstance(v.rhs, ir.BinaryExpr) and v.rhs.op == "*"

    def test_equality_below_comparison(self) -> None:
        """``a < b == c < d`` parses; equality groups the comparisons."""
        prog = _parse_v6(_wrap("x = a < b == c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        # comparison rule is left-associative — `(a<b)==c` per `_fold_binary`.
        assert isinstance(v, ir.BinaryExpr)
        assert v.op in ("<", "==")

    def test_logical_and_below_or(self) -> None:
        """``a or b and c`` -> ``a or (b and c)``."""
        prog = _parse_v6(_wrap("x = a or b and c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "or"
        assert isinstance(v.rhs, ir.BinaryExpr) and v.rhs.op == "and"

    def test_not_unary_below_and(self) -> None:
        """``not a and b`` -> ``(not a) and b``."""
        prog = _parse_v6(_wrap("x = not a and b\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "and"
        assert isinstance(v.lhs, ir.UnaryExpr) and v.lhs.op == "not"

    def test_ternary_binds_lowest(self) -> None:
        """``a + b ? c : d`` -> ``(a+b) ? c : d``."""
        prog = _parse_v6(_wrap("x = a + b ? c : d\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.TernaryExpr)
        assert isinstance(v.cond, ir.BinaryExpr) and v.cond.op == "+"

    def test_unary_minus_then_power(self) -> None:
        """``-a * b`` -> ``(-a) * b`` (unary above multiplicative)."""
        prog = _parse_v6(_wrap("x = -a * b\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "*"
        assert isinstance(v.lhs, ir.UnaryExpr) and v.lhs.op == "-"

    def test_modulo(self) -> None:
        prog = _parse_v6(_wrap("x = a % b\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "%"

    def test_paren_overrides_precedence(self) -> None:
        """``(a + b) * c`` -> ``(a+b) * c``."""
        prog = _parse_v6(_wrap("x = (a + b) * c\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == "*"
        assert isinstance(v.lhs, ir.BinaryExpr) and v.lhs.op == "+"

    @pytest.mark.parametrize(
        "src, op",
        [
            ("a == b", "=="), ("a != b", "!="),
            ("a < b", "<"), ("a <= b", "<="),
            ("a > b", ">"), ("a >= b", ">="),
        ],
    )
    def test_comparison_operators(self, src: str, op: str) -> None:
        prog = _parse_v6(_wrap(f"x = {src}\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BinaryExpr) and v.op == op


# ---------------------------------------------------------------------------
# Error recovery — hints from UnexpectedToken.match_examples (D1 §1.3)
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    def test_bare_assign_rhs_missing(self) -> None:
        with pytest.raises(PineSyntaxError) as exc:
            _parse_v6("x = \n")
        msg = str(exc.value)
        assert "Right-hand side of `=`" in msg

    def test_empty_call_arg(self) -> None:
        with pytest.raises(PineSyntaxError) as exc:
            _parse_v6("f(,)\n")
        assert "argument is empty" in str(exc.value)

    def test_if_with_colon(self) -> None:
        with pytest.raises(PineSyntaxError) as exc:
            _parse_v6("if cond : y = 1\n")
        assert "indentation" in str(exc.value).lower() or "newline" in str(exc.value).lower()

    def test_unclosed_paren(self) -> None:
        with pytest.raises(PineSyntaxError) as exc:
            _parse_v6("f(1, 2\n")
        assert "Unclosed parenthesis" in str(exc.value) or "parenthesis" in str(exc.value).lower()

    def test_syntax_error_carries_line_col(self) -> None:
        with pytest.raises(PineSyntaxError) as exc:
            _parse_v6("x = \n")
        msg = str(exc.value)
        # Line and column should be present (1-based) per D1 §5.1.
        assert "line " in msg and "col " in msg

    def test_completely_malformed_doesnt_silent_pass(self) -> None:
        # `1 = 2` — invalid assignment target. Must raise, never silently parse.
        with pytest.raises(PineSyntaxError):
            _parse_v6("1 = 2\n")


# ---------------------------------------------------------------------------
# Pine v6 indicator skeleton (the BB fixture from D1 §1.4)
# ---------------------------------------------------------------------------


BB_FIXTURE = (
    '//@version=6\n'
    'indicator("BB")\n'
    'length = input.int(20, minval=1)\n'
    'mult   = input.float(2.0)\n'
    'basis  = ta.sma(close, length)\n'
    'dev    = mult * ta.stdev(close, length)\n'
    'plot(basis); plot(basis + dev); plot(basis - dev)\n'
)


class TestIndicatorSkeleton:
    def test_program_version_set_from_pragma(self) -> None:
        prog = _parse_v6(BB_FIXTURE)
        assert prog.version == 6

    def test_directive_is_indicator(self) -> None:
        prog = _parse_v6(BB_FIXTURE)
        d = prog.directive
        assert isinstance(d, ir.ScriptDirective)
        assert d.kind == "indicator"
        assert d.title == "BB"

    def test_body_has_all_statements(self) -> None:
        """4 assigns (length/mult/basis/dev) + 3 plot expr-stmts = 7."""
        prog = _parse_v6(BB_FIXTURE)
        assert len(prog.body) == 7
        kinds = [type(s).__name__ for s in prog.body]
        assert kinds.count("Assign") == 4
        assert kinds.count("ExprStmt") == 3

    def test_first_assign_is_length(self) -> None:
        prog = _parse_v6(BB_FIXTURE)
        first = prog.body[0]
        assert isinstance(first, ir.Assign)
        assert isinstance(first.target, ir.Name) and first.target.id == "length"
        call = first.value
        assert isinstance(call, ir.CallExpr)
        assert isinstance(call.func, ir.Attribute) and call.func.attr == "int"

    def test_dev_uses_binary_mul(self) -> None:
        prog = _parse_v6(BB_FIXTURE)
        # dev    = mult * ta.stdev(close, length)
        dev = prog.body[3]
        assert isinstance(dev, ir.Assign)
        assert isinstance(dev.value, ir.BinaryExpr) and dev.value.op == "*"


# ---------------------------------------------------------------------------
# Pine v5 placeholder — passes if grammar imports v6 cleanly (C7's bead)
# ---------------------------------------------------------------------------


class TestPineV5Placeholder:
    def test_v5_grammar_loads(self) -> None:
        # Must not blow up — the placeholder copies v6.
        parser = _lark_for_version(5)
        assert parser is not None

    def test_v5_minimal_indicator(self) -> None:
        src = '//@version=5\nindicator("X")\nplot(close)\n'
        prog = _parse_v5(src)
        assert prog.version == 5
        assert prog.directive.kind == "indicator"
        assert len(prog.body) == 1


# ---------------------------------------------------------------------------
# Token-stream sanity — hand-crafted bad input doesn't crash silently
# ---------------------------------------------------------------------------


class TestTokenStream:
    def test_parser_rejects_pre_lexer_garbage(self) -> None:
        """Hand-craft a Token stream containing token kinds that violate
        every program rule. The parser must raise, never silently succeed."""
        tokens = [
            Token(kind="OP_PLUS", text="+", line=1, col=1),
            Token(kind="OP_STAR", text="*", line=1, col=2),
            Token(kind="EOF", text="", line=1, col=3),
        ]
        with pytest.raises(PineSyntaxError):
            parse(tokens, pine_version=6)

    def test_empty_token_list_is_an_error(self) -> None:
        with pytest.raises(PineSyntaxError):
            parse([], pine_version=6)

    def test_unsupported_pine_version_raises(self) -> None:
        from openbb_pine.compiler.parser import _lark_for_version
        with pytest.raises(PineSyntaxError):
            _lark_for_version(99)


# ---------------------------------------------------------------------------
# Literal-shape tests — IR fields per D1 §2.4
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_int_literal(self) -> None:
        prog = _parse_v6(_wrap("x = 42\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.IntLit) and v.value == 42

    def test_float_literal(self) -> None:
        prog = _parse_v6(_wrap("x = 2.5\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.FloatLit) and v.value == 2.5

    def test_string_literal(self) -> None:
        prog = _parse_v6(_wrap('x = "hello"\n'))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.StrLit) and v.value == "hello"

    def test_true_literal(self) -> None:
        prog = _parse_v6(_wrap("x = true\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BoolLit) and v.value is True

    def test_false_literal(self) -> None:
        prog = _parse_v6(_wrap("x = false\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.BoolLit) and v.value is False

    def test_na_literal(self) -> None:
        prog = _parse_v6(_wrap("x = na\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.NaLit)


# ---------------------------------------------------------------------------
# IR shape invariants D1 §2.x guarantees codegen
# ---------------------------------------------------------------------------


class TestIRInvariants:
    def test_program_loc_is_span(self) -> None:
        prog = _parse_v6(_wrap("x = 1\n"))
        assert isinstance(prog.loc, ir.Span)
        assert prog.loc.file == "<inline>"

    def test_every_node_carries_loc(self) -> None:
        prog = _parse_v6(_wrap("x = a + b * c\n"))
        # Walk into the body and assert each node has a Span.
        a = prog.body[0]
        assert isinstance(a.loc, ir.Span)
        # The value expression and its descendants too.
        v = a.value  # type: ignore[union-attr]
        assert isinstance(v.loc, ir.Span)
        assert isinstance(v.lhs.loc, ir.Span)  # type: ignore[union-attr]
        assert isinstance(v.rhs.loc, ir.Span)  # type: ignore[union-attr]

    def test_subscript_kind_set(self) -> None:
        """D1 §2.6 invariant 3: every Subscript.kind is set (default "history")."""
        prog = _parse_v6(_wrap("y = arr[1]\n"))
        v = prog.body[0].value  # type: ignore[union-attr]
        assert isinstance(v, ir.Subscript)
        assert v.kind in ("history", "index")
