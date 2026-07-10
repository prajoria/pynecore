"""Tests for ``openbb_pine.compiler.ir`` — frozen dataclass IR nodes.

Source of truth: D1 §2.3-§2.4 (Program / Declaration / Statement / Expression).
Every node is ``@dataclass(frozen=True, slots=True)`` with a uniform ``loc:
Span``. The contract codegen relies on is one node type per Pine syntactic
category — no AST-vs-IR split (D1 §2.5).
"""

from __future__ import annotations

import dataclasses

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_span():
    from openbb_pine.compiler.ir import Span

    return Span(
        file="<inline>",
        start_line=1,
        start_col=0,
        end_line=1,
        end_col=1,
        start_byte=0,
        end_byte=1,
    )


def scalar_int():
    from openbb_pine.compiler.types import PineType, Scalar

    return PineType(qualifier="simple", inner=Scalar(kind="int"))


# ---------------------------------------------------------------------------
# Span + Node base
# ---------------------------------------------------------------------------


class TestSpan:
    def test_span_carries_position_data(self) -> None:
        from openbb_pine.compiler.ir import Span

        s = Span(
            file="my.pine",
            start_line=2,
            start_col=4,
            end_line=2,
            end_col=10,
            start_byte=12,
            end_byte=18,
        )
        assert s.file == "my.pine"
        assert s.start_line == 2
        assert s.start_byte == 12

    def test_span_is_frozen_and_hashable(self) -> None:
        s = make_span()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.file = "other.pine"  # type: ignore[misc]
        hash(s)  # must not raise


class TestNodeBase:
    def test_node_has_loc(self) -> None:
        from openbb_pine.compiler.ir import IntLit

        n = IntLit(loc=make_span(), value=3)
        assert n.loc.start_line == 1

    def test_slots_no_dict(self) -> None:
        from openbb_pine.compiler.ir import IntLit

        n = IntLit(loc=make_span(), value=3)
        assert not hasattr(n, "__dict__")


# ---------------------------------------------------------------------------
# Expressions (D1 §2.4)
# ---------------------------------------------------------------------------


class TestExpressions:
    def test_literals(self) -> None:
        from openbb_pine.compiler.ir import (
            BoolLit,
            ColorLit,
            FloatLit,
            IntLit,
            NaLit,
            StrLit,
        )

        loc = make_span()
        assert IntLit(loc=loc, value=42).value == 42
        assert FloatLit(loc=loc, value=3.14).value == 3.14
        assert StrLit(loc=loc, value="hi").value == "hi"
        assert BoolLit(loc=loc, value=True).value is True
        assert isinstance(NaLit(loc=loc), NaLit)
        assert ColorLit(loc=loc, raw="#ff0000").raw == "#ff0000"

    def test_name_and_attribute_and_subscript(self) -> None:
        from openbb_pine.compiler.ir import Attribute, Name, Subscript

        loc = make_span()
        n = Name(loc=loc, id="close")
        attr = Attribute(loc=loc, value=Name(loc=loc, id="ta"), attr="sma")
        sub = Subscript(loc=loc, value=n, index=IntLitOne(loc), kind="history")
        assert n.id == "close"
        assert attr.attr == "sma"
        assert sub.kind == "history"

    def test_binary_unary_ternary(self) -> None:
        from openbb_pine.compiler.ir import (
            BinaryExpr,
            Name,
            TernaryExpr,
            UnaryExpr,
        )

        loc = make_span()
        lhs = Name(loc=loc, id="a")
        rhs = Name(loc=loc, id="b")
        be = BinaryExpr(loc=loc, op="+", lhs=lhs, rhs=rhs)
        ue = UnaryExpr(loc=loc, op="-", operand=lhs)
        te = TernaryExpr(loc=loc, cond=lhs, then_=lhs, else_=rhs)
        assert be.op == "+"
        assert ue.op == "-"
        assert te.cond is lhs

    def test_call_with_keyword_args(self) -> None:
        from openbb_pine.compiler.ir import CallExpr, KeywordArg, Name

        loc = make_span()
        positional = KeywordArg(loc=loc, name=None, value=Name(loc=loc, id="close"))
        kw = KeywordArg(loc=loc, name="length", value=IntLitOne(loc))
        call = CallExpr(
            loc=loc,
            func=Name(loc=loc, id="ta_sma"),
            args=(positional, kw),
            type_args=(),
        )
        assert call.args[0].name is None
        assert call.args[1].name == "length"
        assert call.type_args == ()

    def test_call_with_type_args(self) -> None:
        from openbb_pine.compiler.ir import CallExpr, Name

        loc = make_span()
        call = CallExpr(
            loc=loc,
            func=Name(loc=loc, id="array_new"),
            args=(),
            type_args=(scalar_int(),),
        )
        assert call.type_args[0].inner.kind == "int"

    def test_tuple_expr(self) -> None:
        from openbb_pine.compiler.ir import Name, TupleExpr

        loc = make_span()
        t = TupleExpr(loc=loc, elements=(Name(loc=loc, id="x"), Name(loc=loc, id="y")))
        assert len(t.elements) == 2


# ---------------------------------------------------------------------------
# Statements (D1 §2.3)
# ---------------------------------------------------------------------------


class TestStatements:
    def test_vardecl_and_assign(self) -> None:
        from openbb_pine.compiler.ir import Assign, Name, VarDecl

        loc = make_span()
        decl = VarDecl(loc=loc, qualifier="var", name="x", type=None, value=IntLitOne(loc))
        assign = Assign(loc=loc, target=Name(loc=loc, id="x"), op=":=", value=IntLitOne(loc))
        assert decl.qualifier == "var"
        assert assign.op == ":="

    def test_if_stmt(self) -> None:
        from openbb_pine.compiler.ir import IfStmt, Name

        loc = make_span()
        s = IfStmt(
            loc=loc,
            cond=Name(loc=loc, id="cond"),
            then_body=(),
            elif_branches=((Name(loc=loc, id="c2"), ()),),
            else_body=(),
        )
        assert s.else_body == ()
        assert s.elif_branches[0][0].id == "c2"

    def test_for_and_forin_and_while(self) -> None:
        from openbb_pine.compiler.ir import (
            ForInStmt,
            ForStmt,
            Name,
            WhileStmt,
        )

        loc = make_span()
        f = ForStmt(
            loc=loc,
            var="i",
            start=IntLitOne(loc),
            end=IntLitOne(loc),
            step=None,
            body=(),
        )
        fi = ForInStmt(loc=loc, var="x", iterable=Name(loc=loc, id="arr"), body=())
        w = WhileStmt(loc=loc, cond=Name(loc=loc, id="c"), body=())
        assert f.step is None
        assert fi.iterable.id == "arr"
        assert w.cond.id == "c"

    def test_switch_and_return_and_exprstmt(self) -> None:
        from openbb_pine.compiler.ir import (
            ExprStmt,
            Name,
            ReturnStmt,
            SwitchStmt,
        )

        loc = make_span()
        sw = SwitchStmt(
            loc=loc,
            scrutinee=Name(loc=loc, id="x"),
            cases=((IntLitOne(loc), ()), (None, ())),
        )
        r = ReturnStmt(loc=loc, value=None)
        e = ExprStmt(loc=loc, expr=Name(loc=loc, id="plot"))
        assert sw.cases[1][0] is None  # default branch
        assert r.value is None
        assert e.expr.id == "plot"


# ---------------------------------------------------------------------------
# Declarations + Program (D1 §2.3)
# ---------------------------------------------------------------------------


class TestDeclarationsAndProgram:
    def test_parameter_keywordarg(self) -> None:
        from openbb_pine.compiler.ir import KeywordArg, Name, Parameter

        loc = make_span()
        p = Parameter(loc=loc, name="length", type=scalar_int(), default=IntLitOne(loc))
        k = KeywordArg(loc=loc, name="overlay", value=Name(loc=loc, id="true"))
        assert p.name == "length"
        assert p.type.inner.kind == "int"
        assert k.value.id == "true"

    def test_function_decl(self) -> None:
        from openbb_pine.compiler.ir import FunctionDecl, Parameter

        loc = make_span()
        p = Parameter(loc=loc, name="x", type=scalar_int(), default=None)
        fn = FunctionDecl(
            loc=loc,
            name="my_fn",
            is_method=False,
            receiver=None,
            type_params=(),
            parameters=(p,),
            return_type=None,
            body=(),
        )
        assert fn.name == "my_fn"
        assert fn.is_method is False
        assert fn.parameters[0].name == "x"

    def test_type_decl_and_enum_decl(self) -> None:
        from openbb_pine.compiler.ir import EnumDecl, Parameter, TypeDecl

        loc = make_span()
        td = TypeDecl(
            loc=loc,
            name="Bar",
            fields=(Parameter(loc=loc, name="o", type=scalar_int(), default=None),),
            extends=None,
        )
        ed = EnumDecl(loc=loc, name="State", members=(("A", None), ("B", IntLitOne(loc))))
        assert td.name == "Bar"
        assert ed.members[0] == ("A", None)

    def test_script_directive(self) -> None:
        from openbb_pine.compiler.ir import KeywordArg, Name, ScriptDirective

        loc = make_span()
        sd = ScriptDirective(
            loc=loc,
            kind="indicator",
            title="BB",
            shorttitle=None,
            overlay=True,
            arguments=(
                KeywordArg(loc=loc, name="title", value=Name(loc=loc, id="bb_title")),
            ),
        )
        assert sd.kind == "indicator"
        assert sd.overlay is True
        assert sd.arguments[0].name == "title"

    def test_program_root(self) -> None:
        from openbb_pine.compiler.ir import Program, ScriptDirective

        loc = make_span()
        sd = ScriptDirective(
            loc=loc,
            kind="indicator",
            title="BB",
            shorttitle=None,
            overlay=None,
            arguments=(),
        )
        prog = Program(
            loc=loc,
            version=6,
            directive=sd,
            declarations=(),
            body=(),
        )
        assert prog.version == 6
        assert prog.directive.kind == "indicator"


# ---------------------------------------------------------------------------
# Equality / hash + frozen + slots semantics
# ---------------------------------------------------------------------------


class TestEqualityAndHash:
    def test_equal_nodes_compare_equal_and_hash_same(self) -> None:
        from openbb_pine.compiler.ir import IntLit

        loc = make_span()
        a = IntLit(loc=loc, value=5)
        b = IntLit(loc=loc, value=5)
        assert a == b
        assert hash(a) == hash(b)

    def test_distinct_values_unequal(self) -> None:
        from openbb_pine.compiler.ir import IntLit

        loc = make_span()
        assert IntLit(loc=loc, value=1) != IntLit(loc=loc, value=2)

    def test_frozen(self) -> None:
        from openbb_pine.compiler.ir import IntLit

        n = IntLit(loc=make_span(), value=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.value = 9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper to avoid repeating IntLit(loc=loc, value=1) literals everywhere
# ---------------------------------------------------------------------------


def IntLitOne(loc):
    from openbb_pine.compiler.ir import IntLit

    return IntLit(loc=loc, value=1)
