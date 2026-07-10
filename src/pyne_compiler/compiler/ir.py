"""Frozen-dataclass IR for the Pine compiler.

Authoritative source: D1 §2.2-§2.5. Philosophy:

* One ``@dataclass(frozen=True, slots=True)`` per Pine node category.
* Uniform ``loc: Span`` field on every node — D1 §2.2.
* No AST-vs-IR split; the type checker annotates via a sidecar ``types: dict[
  id(node), PineType]`` (D1 §2.5), so codegen never sees two views of the same
  ``BinaryExpr``.

These types are pure data — no behavior. C3 (type checker) and C5 (codegen)
consume them; this module has no import-time work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openbb_pine.compiler.types import PineType

__all__ = [
    "Span",
    "Node",
    # Program / declarations
    "Program",
    "ScriptDirective",
    "Declaration",
    "FunctionDecl",
    "TypeDecl",
    "EnumDecl",
    "Parameter",
    "KeywordArg",
    # Statements
    "Statement",
    "VarDecl",
    "Assign",
    "IfStmt",
    "ForStmt",
    "ForInStmt",
    "WhileStmt",
    "SwitchStmt",
    "ReturnStmt",
    "ExprStmt",
    # Expressions
    "Expression",
    "IntLit",
    "FloatLit",
    "StrLit",
    "BoolLit",
    "NaLit",
    "ColorLit",
    "Name",
    "Attribute",
    "Subscript",
    "BinaryExpr",
    "UnaryExpr",
    "TernaryExpr",
    "CallExpr",
    "TupleExpr",
    # Operator literals
    "BinOp",
    "UnaryOp",
]


# ---------------------------------------------------------------------------
# Position + Node base (D1 §2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """Source position carrying line/col + byte-offset range (D1 §2.2)."""

    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class Node:
    """Marker base — every IR node carries a ``loc: Span``."""

    loc: Span


# ---------------------------------------------------------------------------
# Expressions (D1 §2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Expression(Node):
    """Marker base for expression nodes."""


# Literals -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntLit(Expression):
    value: int


@dataclass(frozen=True, slots=True)
class FloatLit(Expression):
    value: float


@dataclass(frozen=True, slots=True)
class StrLit(Expression):
    value: str


@dataclass(frozen=True, slots=True)
class BoolLit(Expression):
    value: bool


@dataclass(frozen=True, slots=True)
class NaLit(Expression):
    """Pine's ``na`` literal."""


@dataclass(frozen=True, slots=True)
class ColorLit(Expression):
    """Color literal, canonical lower-case hex or symbolic name."""

    raw: str


# Names + access -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Name(Expression):
    id: str


@dataclass(frozen=True, slots=True)
class Attribute(Expression):
    """``ta.sma`` or ``obj.field``."""

    value: Expression
    attr: str


@dataclass(frozen=True, slots=True)
class Subscript(Expression):
    """``arr[i]`` (index) or history access ``x[n]``; ``kind`` set by C3."""

    value: Expression
    index: Expression
    kind: Literal["history", "index"]


# Operators ----------------------------------------------------------------


BinOp = Literal[
    "+",
    "-",
    "*",
    "/",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
]

UnaryOp = Literal["+", "-", "not"]


@dataclass(frozen=True, slots=True)
class BinaryExpr(Expression):
    op: BinOp
    lhs: Expression
    rhs: Expression


@dataclass(frozen=True, slots=True)
class UnaryExpr(Expression):
    op: UnaryOp
    operand: Expression


@dataclass(frozen=True, slots=True)
class TernaryExpr(Expression):
    """``cond ? then : else``."""

    cond: Expression
    then_: Expression
    else_: Expression


@dataclass(frozen=True, slots=True)
class KeywordArg(Node):
    """Argument carrier; ``name is None`` marks a positional."""

    name: str | None
    value: Expression


@dataclass(frozen=True, slots=True)
class CallExpr(Expression):
    """A function/method call. ``type_args`` carries explicit v6 type params."""

    func: Expression  # Name or Attribute
    args: tuple[KeywordArg, ...]
    type_args: tuple[PineType, ...] = ()


@dataclass(frozen=True, slots=True)
class TupleExpr(Expression):
    """Destructuring / multi-return tuple."""

    elements: tuple[Expression, ...]


# ---------------------------------------------------------------------------
# Statements (D1 §2.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Statement(Node):
    """Marker base for statement nodes."""


@dataclass(frozen=True, slots=True)
class VarDecl(Statement):
    """``[var|varip] name [: type] = value``; init must be non-None (D1 §2.6 inv 5)."""

    qualifier: Literal["var", "varip"] | None
    name: str
    type: PineType | None
    value: Expression


@dataclass(frozen=True, slots=True)
class Assign(Statement):
    """``target = value`` (declare/reassign) or ``target := value`` (mutate)."""

    target: Expression  # Name | Subscript | Attribute
    op: Literal["=", ":="]
    value: Expression


@dataclass(frozen=True, slots=True)
class IfStmt(Statement):
    """If/elif*/else; ``else_body is None`` when no ``else`` branch is present."""

    cond: Expression
    then_body: tuple[Statement, ...]
    elif_branches: tuple[tuple[Expression, tuple[Statement, ...]], ...]
    else_body: tuple[Statement, ...] | None


@dataclass(frozen=True, slots=True)
class ForStmt(Statement):
    """``for v = start to end [by step]``."""

    var: str
    start: Expression
    end: Expression
    step: Expression | None
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class ForInStmt(Statement):
    """``for v in iterable`` (v6)."""

    var: str
    iterable: Expression
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class WhileStmt(Statement):
    cond: Expression
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class SwitchStmt(Statement):
    """``switch scrutinee ... key => body``; ``scrutinee is None`` for bare ``switch``;
    ``cases[i][0] is None`` marks the ``=>`` default branch."""

    scrutinee: Expression | None
    cases: tuple[tuple[Expression | None, tuple[Statement, ...]], ...]


@dataclass(frozen=True, slots=True)
class ReturnStmt(Statement):
    value: Expression | None


@dataclass(frozen=True, slots=True)
class ExprStmt(Statement):
    """An expression used in statement position — e.g. ``plot(close)``."""

    expr: Expression


# ---------------------------------------------------------------------------
# Declarations + Program (D1 §2.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Parameter(Node):
    """Formal parameter (function signature or UDT field)."""

    name: str
    type: PineType
    default: Expression | None


@dataclass(frozen=True, slots=True)
class Declaration(Node):
    """Marker base for declarations."""


@dataclass(frozen=True, slots=True)
class FunctionDecl(Declaration):
    """User function or v6 ``method``."""

    name: str
    is_method: bool
    receiver: Parameter | None
    type_params: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    return_type: PineType | None
    body: tuple[Statement, ...]


@dataclass(frozen=True, slots=True)
class TypeDecl(Declaration):
    """``type Foo`` UDT definition."""

    name: str
    fields: tuple[Parameter, ...]
    extends: str | None


@dataclass(frozen=True, slots=True)
class EnumDecl(Declaration):
    """v6 ``enum`` declaration; member value ``None`` means auto-assigned."""

    name: str
    members: tuple[tuple[str, Expression | None], ...]


@dataclass(frozen=True, slots=True)
class ScriptDirective(Node):
    """The ``indicator()`` / ``strategy()`` / ``library()`` declaration."""

    kind: Literal["indicator", "strategy", "library"]
    title: str
    shorttitle: str | None
    overlay: bool | None
    arguments: tuple[KeywordArg, ...]


@dataclass(frozen=True, slots=True)
class Program(Node):
    """Root IR node — one per source file."""

    version: Literal[5, 6]
    directive: ScriptDirective
    declarations: tuple[Declaration, ...]
    body: tuple[Statement, ...]
