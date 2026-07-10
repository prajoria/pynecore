"""Pine Script parser — lark Earley over the C1 lexer's Token stream.

Authoritative source: D1 §1.3 (Earley over LALR), §1.5 (versions + grammar
layout), §2.x (IR shapes).

Public surface:

* :func:`parse` — ``list[Token]`` → :class:`openbb_pine.compiler.ir.Program`
  for ``pine_version`` 5 or 6. Raises
  :class:`openbb_pine.errors.PineSyntaxError` with a hint sourced from
  :meth:`lark.exceptions.UnexpectedToken.match_examples` on parse failure.

Design notes per D1 §1.3:

* The standard lark lexer is **bypassed**. We have a hand-rolled lexer
  (`openbb_pine.compiler.lexer`) that already emits a clean Token stream,
  including INDENT/DEDENT bookkeeping and version-pragma routing — wedging
  lark's own lexer in front of that would lose those features.
* A custom :class:`lark.lexer.Lexer` (``__future_interface__ = True``)
  adapts our ``list[Token]`` into ``lark.Token`` objects on demand.
* Pine keywords (``if``, ``else``, ``for``, ``var``, ``and``, etc.) come
  from the lexer as ``NAME`` tokens; the adapter reclassifies them into
  the dedicated ``KW_*`` terminals the grammar branches on. Keeping the
  C1 lexer dialect-agnostic (it just emits NAMEs) and reclassifying here
  keeps grammar / lexer concerns layered correctly.
* Earley (not LALR) per D1 §1.3 — Pine has genuine grammar ambiguities
  (``f(x)[i]`` etc.) we want the parser to disambiguate deterministically
  at grammar-development time, not LALR-style table conflicts at build.

Implementation also notes (resolves a 1.2.2 mechanism question called out
in the bead): ``Lark.parse_interactive`` is LALR-only in 1.2.2, so we
cannot use it for token feeding. Instead we pass our token list as the
``parse(text=...)`` argument; the custom lexer's ``lex()`` receives it on
``lexer_state.text`` and walks it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterator

from lark import Lark, Token as LarkToken, Transformer, Tree
from lark.exceptions import LarkError, UnexpectedInput, UnexpectedToken
from lark.lexer import Lexer as LarkLexer

from openbb_pine.compiler import ir
from openbb_pine.compiler.lexer import Token as PineToken
from openbb_pine.compiler.types import (
    PineType,
    Scalar,
    UDT,
    ArrayT,
    MapT,
    MatrixT,
)
from openbb_pine.compiler_errors import PineSyntaxError

__all__ = ["parse"]


# ---------------------------------------------------------------------------
# Token-class adapter
# ---------------------------------------------------------------------------


# Pine reserved-word -> dedicated grammar terminal. Reclassifying NAMEs here
# (rather than carving keyword detection into the C1 lexer) keeps the lexer
# dialect-agnostic: same Token stream feeds whichever pine_version's grammar
# we route to. D1 §1.2 — lexer stays trivially extensible.
_KEYWORDS: dict[str, str] = {
    # Script directives
    "indicator": "KW_INDICATOR",
    "strategy":  "KW_STRATEGY",
    "library":   "KW_LIBRARY",
    # Control flow
    "if":       "KW_IF",
    "else":     "KW_ELSE",
    "for":      "KW_FOR",
    "to":       "KW_TO",
    "by":       "KW_BY",
    "in":       "KW_IN",
    "while":    "KW_WHILE",
    "switch":   "KW_SWITCH",
    # Declarations
    "var":      "KW_VAR",
    "varip":    "KW_VARIP",
    "method":   "KW_METHOD",
    "enum":     "KW_ENUM",
    "type":     "KW_TYPE",
    "import":   "KW_IMPORT",
    "export":   "KW_EXPORT",
    "return":   "KW_RETURN",
    "continue": "KW_CONTINUE",
    "break":    "KW_BREAK",
    # Literals
    "true":  "KW_TRUE",
    "false": "KW_FALSE",
    "na":    "KW_NA",
    # Logical operators (Pine's `and` / `or` / `not` are spelt as words)
    "and": "KW_AND",
    "or":  "KW_OR",
    "not": "KW_NOT",
    # NOTE: Pine's type-qualifier words (`const`/`input`/`simple`/`series`) are
    # NOT reclassified here. They appear as keywords only in type-annotation
    # position (`simple int`, `series float`) — everywhere else they are
    # legitimate identifiers (`input.int(...)` is the builtin input namespace;
    # `simple_var = ...` is a legal name). The type-annotation grammar uses
    # NAME and the IRBuilder pattern-matches on the string value.
}


def _to_lark_token(t: PineToken, kind_override: str | None = None) -> LarkToken:
    """Convert a Pine Token into a lark Token, optionally retyping it.

    Carries 1-based line/col verbatim so ``UnexpectedToken.line`` /
    ``column`` line up with the source (and with what ``Span`` records).
    """
    kind = kind_override or t.kind
    # lark.Token signature: Token(type_, value, ...positional or keyword).
    return LarkToken(kind, t.text, None, t.line, t.col, None, t.line, t.col + max(1, len(t.text)))


class _TokenFeederLexer(LarkLexer):
    """A lark-compatible Lexer wrapping our Pine Token stream.

    ``__future_interface__ = True`` tells lark to call ``lex(lexer_state, parser_state)``
    directly (instead of wrapping us in a string-only shim). We don't use
    ``parser_state``; we just walk ``lexer_state.text`` (which is whatever
    object the caller passed to ``Lark.parse(...)`` — in our case, the
    Pine Token list).
    """

    __future_interface__ = True

    def __init__(self, lexer_conf: Any) -> None:  # noqa: D401 — lark contract
        # lexer_conf is unused; we don't have a grammar-derived terminal set
        # because all terminals are %declare'd.
        pass

    def lex(self, lexer_state: Any, parser_state: Any) -> Iterator[LarkToken]:
        tokens = lexer_state.text  # we stuffed the Pine list in here
        for tok in tokens:
            # NEWLINE/INDENT/DEDENT/EOF pass through.
            # NAMEs that match Pine keywords are reclassified.
            if tok.kind == "NAME" and tok.text in _KEYWORDS:
                yield _to_lark_token(tok, _KEYWORDS[tok.text])
            else:
                yield _to_lark_token(tok)


# ---------------------------------------------------------------------------
# Lark factory — cache per (pine_version, grammar_mtime)
# ---------------------------------------------------------------------------


_GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), "grammar")


@lru_cache(maxsize=4)
def _lark_for_version(pine_version: int) -> Lark:
    """Construct (and cache) a Lark instance for the given Pine version."""
    if pine_version not in (5, 6):
        raise PineSyntaxError(
            f"unsupported pine_version {pine_version!r}; expected 5 or 6"
        )
    grammar_path = os.path.join(_GRAMMAR_DIR, f"pine_v{pine_version}.lark")
    if not os.path.exists(grammar_path):
        raise PineSyntaxError(
            f"missing grammar file for pine v{pine_version} at {grammar_path}"
        )
    return Lark.open(
        grammar_path,
        parser="earley",
        lexer=_TokenFeederLexer,
        start="start",
        # Earley + ambiguity='resolve' picks the leftmost derivation —
        # which, given our precedence-layered grammar, gives the
        # mathematically-conventional binding.
        ambiguity="resolve",
        # propagate_positions tells lark to copy line/col onto Tree nodes;
        # the transformer uses them for Span construction.
        propagate_positions=True,
        maybe_placeholders=False,
    )


# ---------------------------------------------------------------------------
# Error-recovery hint registry (D1 §1.3 — UnexpectedToken.match_examples)
# ---------------------------------------------------------------------------


# Map from a hint label -> a small set of malformed Pine snippets that
# `UnexpectedToken.match_examples` should recognise. The label is the hint
# string the parser attaches to PineSyntaxError. Examples are intentionally
# minimal so they parse the same way under both v5 and v6.
_ERROR_EXAMPLES: dict[str, list[str]] = {
    "If statement is missing a newline after its condition. "
    "Pine uses indentation, not `:` — try `if x\\n    y = 1`.": [
        "if x : y = 1\n",
        "if cond : x = 1\n",
    ],
    "Right-hand side of `=` is missing. Provide a value, "
    "e.g. `x = 0` or `x = ta.sma(close, 20)`.": [
        "x = \n",
        "y =\n",
        "z =\nq = 1\n",
    ],
    "Function-call argument is empty. Remove the stray comma or add an argument: "
    "`f(a, b)` not `f(,)`.": [
        "f(,)\n",
        "g( , 1)\n",
    ],
    "Unclosed parenthesis. Add the matching `)` to close the call.": [
        "f(1, 2\n",
        "x = (1 + 2\n",
    ],
}


def _match_hint(err: UnexpectedToken, pine_version: int) -> str | None:
    """Try matching a parse error against the example registry.

    Lark's ``match_examples`` requires a parse function that can be called
    repeatedly with snippets and that raises on errors — so we lex each
    example and feed it through the cached parser.
    """
    from openbb_pine.compiler.lexer import tokenize as _pine_tokenize

    parser = _lark_for_version(pine_version)

    def _parse_example(text: str) -> Tree:
        toks = _pine_tokenize(text)
        return parser.parse(toks)

    try:
        return err.match_examples(_parse_example, _ERROR_EXAMPLES)
    except Exception:
        # match_examples is best-effort polish; it never blocks the real error.
        return None


# ---------------------------------------------------------------------------
# IR Builder — Transformer walks the parse tree, emitting IR dataclasses
# ---------------------------------------------------------------------------


def _span(file: str, tree_or_tok: Any) -> ir.Span:
    """Pull (line, col) out of a Tree / Token / nested tuple, build a Span.

    Best-effort. lark's ``propagate_positions`` sets ``meta.line`` /
    ``meta.column`` on Trees; tokens carry ``.line`` / ``.column``. We
    derive ``start_byte``/``end_byte`` from line/col only weakly (0 / 0
    when unknown) — exact byte offsets are a Phase-3 polish (C3 type
    checker doesn't need them yet, and codegen pulls source via Span.file
    only for error rendering).
    """
    line = 1
    col = 1
    end_line = 1
    end_col = 1
    if isinstance(tree_or_tok, LarkToken):
        line = tree_or_tok.line or 1
        col = tree_or_tok.column or 1
        end_line = tree_or_tok.end_line or line
        end_col = tree_or_tok.end_column or col + len(str(tree_or_tok))
    elif isinstance(tree_or_tok, Tree):
        m = tree_or_tok.meta
        if not getattr(m, "empty", True):
            line = m.line
            col = m.column
            end_line = getattr(m, "end_line", line) or line
            end_col = getattr(m, "end_column", col) or col
    return ir.Span(
        file=file,
        start_line=line,
        start_col=col,
        end_line=end_line,
        end_col=end_col,
        start_byte=0,
        end_byte=0,
    )


# Operator-symbol -> IR Literal value, for binary / comparison ops.
_BIN_OPS: dict[str, str] = {
    "OP_PLUS": "+", "OP_MINUS": "-",
    "OP_STAR": "*", "OP_SLASH": "/", "OP_PERCENT": "%",
    "OP_EQ": "==", "OP_NEQ": "!=",
    "OP_LT": "<", "OP_LE": "<=", "OP_GT": ">", "OP_GE": ">=",
    "KW_AND": "and", "KW_OR": "or",
}


class _IRBuilder(Transformer):
    """Walk lark parse tree -> IR dataclasses (D1 §1.5)."""

    def __init__(self, file: str = "<inline>") -> None:
        super().__init__()
        self.file = file

    # -- helpers -----------------------------------------------------------

    def _sp(self, anchor: Any) -> ir.Span:
        return _span(self.file, anchor)

    def _fresh_span(self) -> ir.Span:
        return ir.Span(
            file=self.file,
            start_line=1, start_col=1, end_line=1, end_col=1,
            start_byte=0, end_byte=0,
        )

    # -- literals ----------------------------------------------------------

    def number_literal(self, children: list[LarkToken]) -> ir.Expression:
        tok = children[0]
        text = str(tok)
        if "." in text or "e" in text or "E" in text:
            return ir.FloatLit(loc=self._sp(tok), value=float(text))
        return ir.IntLit(loc=self._sp(tok), value=int(text))

    def string_literal(self, children: list[LarkToken]) -> ir.StrLit:
        tok = children[0]
        return ir.StrLit(loc=self._sp(tok), value=str(tok))

    def true_literal(self, children: list[LarkToken]) -> ir.BoolLit:
        return ir.BoolLit(loc=self._sp(children[0]), value=True)

    def false_literal(self, children: list[LarkToken]) -> ir.BoolLit:
        return ir.BoolLit(loc=self._sp(children[0]), value=False)

    def na_literal(self, children: list[LarkToken]) -> ir.NaLit:
        return ir.NaLit(loc=self._sp(children[0]))

    def literal(self, children: list[Any]) -> Any:
        return children[0]

    # -- names / primary ---------------------------------------------------

    def name_expr(self, children: list[LarkToken]) -> ir.Name:
        tok = children[0]
        return ir.Name(loc=self._sp(tok), id=str(tok))

    def strategy_name_expr(self, children: list[LarkToken]) -> ir.Name:
        # ``strategy`` reclassified to KW_STRATEGY by the token adapter, but
        # legitimately appears in expression position as the Pine ``strategy``
        # namespace (``strategy.entry``, ``strategy.long``, kwargs like
        # ``default_qty_type=strategy.percent_of_equity``). Grammar branch
        # ``primary: KW_STRATEGY -> strategy_name_expr`` reifies it as a
        # plain ``ir.Name`` so the postfix chain (`.entry`, `.long`, etc.)
        # composes normally.
        tok = children[0]
        return ir.Name(loc=self._sp(tok), id=str(tok))

    def paren_expr(self, children: list[Any]) -> Any:
        # children: LPAREN expr RPAREN — return the inner expression
        return next(c for c in children if isinstance(c, ir.Expression))

    def primary(self, children: list[Any]) -> Any:
        return children[0]

    def tuple_literal(self, children: list[Any]) -> ir.TupleExpr:
        exprs = tuple(c for c in children if isinstance(c, ir.Expression))
        return ir.TupleExpr(loc=exprs[0].loc, elements=exprs)

    # -- postfix chain -----------------------------------------------------

    def postfix(self, children: list[Any]) -> ir.Expression:
        expr: ir.Expression = children[0]
        for tail in children[1:]:
            kind, payload, span = tail
            if kind == "call":
                expr = ir.CallExpr(loc=span, func=expr, args=tuple(payload))
            elif kind == "attribute":
                expr = ir.Attribute(loc=span, value=expr, attr=payload)
            elif kind == "subscript":
                # ``[n]`` defaults to history; type checker (C3) may rewrite
                # to "index" once it knows the receiver is array-typed.
                expr = ir.Subscript(loc=span, value=expr, index=payload, kind="history")
        return expr

    def call_tail(self, children: list[Any]) -> tuple[str, list[ir.KeywordArg], ir.Span]:
        # children: LPAREN [call_args=list[KeywordArg]] RPAREN
        args: list[ir.KeywordArg] = []
        for c in children:
            if isinstance(c, list):
                args.extend(c)
            elif isinstance(c, ir.KeywordArg):
                args.append(c)
        span = self._fresh_span()
        return ("call", args, span)

    def attribute_tail(self, children: list[Any]) -> tuple[str, str, ir.Span]:
        # children: DOT NAME — pick the NAME
        tok = next(c for c in children if isinstance(c, LarkToken) and c.type == "NAME")
        return ("attribute", str(tok), self._sp(tok))

    def subscript_tail(self, children: list[Any]) -> tuple[str, ir.Expression, ir.Span]:
        # children: LBRACKET expr RBRACKET — pick the expr
        expr = next(c for c in children if isinstance(c, ir.Expression))
        return ("subscript", expr, expr.loc)

    def postfix_tail(self, children: list[Any]) -> Any:
        return children[0]

    def call_args(self, children: list[Any]) -> list[ir.KeywordArg]:
        return [c for c in children if isinstance(c, ir.KeywordArg)]

    def kwarg(self, children: list[Any]) -> ir.KeywordArg:
        # children: [NAME_tok, ASSIGN_tok, value_expr] — pick by type
        name_tok = next(c for c in children if isinstance(c, LarkToken) and c.type == "NAME")
        value = next(c for c in children if isinstance(c, ir.Expression))
        return ir.KeywordArg(loc=self._sp(name_tok), name=str(name_tok), value=value)

    def posarg(self, children: list[ir.Expression]) -> ir.KeywordArg:
        value = children[0]
        return ir.KeywordArg(loc=value.loc, name=None, value=value)

    # -- unary / arithmetic / comparison ----------------------------------

    def unary_minus(self, children: list[Any]) -> ir.UnaryExpr:
        operand = next(c for c in children if isinstance(c, ir.Expression))
        return ir.UnaryExpr(loc=operand.loc, op="-", operand=operand)

    def unary_plus(self, children: list[Any]) -> ir.UnaryExpr:
        operand = next(c for c in children if isinstance(c, ir.Expression))
        return ir.UnaryExpr(loc=operand.loc, op="+", operand=operand)

    def not_expr(self, children: list[Any]) -> ir.UnaryExpr:
        operand = next(c for c in children if isinstance(c, ir.Expression))
        return ir.UnaryExpr(loc=operand.loc, op="not", operand=operand)

    def unary(self, children: list[Any]) -> Any:
        return children[0]

    def _fold_binary(self, children: list[Any]) -> ir.Expression:
        """Fold a left-associative chain ``a OP b OP c`` -> BinaryExpr.

        children alternate: [expr, op_token, expr, op_token, expr, ...]
        op_token is either a raw LarkToken or a 1-element Tree wrapping one
        (depending on the rule). We accept both.
        """
        result = children[0]
        i = 1
        while i < len(children):
            op_thing = children[i]
            rhs = children[i + 1]
            op_tok = op_thing
            if isinstance(op_thing, Tree):
                op_tok = op_thing.children[0]
            op_kind = op_tok.type if isinstance(op_tok, LarkToken) else str(op_tok)
            op_sym = _BIN_OPS[op_kind]
            result = ir.BinaryExpr(loc=result.loc, op=op_sym, lhs=result, rhs=rhs)
            i += 2
        return result

    def comparison(self, children: list[Any]) -> ir.Expression:
        if len(children) == 1:
            return children[0]
        return self._fold_binary(children)

    def additive(self, children: list[Any]) -> ir.Expression:
        if len(children) == 1:
            return children[0]
        return self._fold_binary(children)

    def multiplicative(self, children: list[Any]) -> ir.Expression:
        if len(children) == 1:
            return children[0]
        return self._fold_binary(children)

    def bool_and(self, children: list[Any]) -> ir.Expression:
        if len(children) == 1:
            return children[0]
        # Children: [expr, KW_AND, expr, KW_AND, expr, ...]
        result = children[0]
        i = 1
        while i < len(children):
            kw = children[i]
            rhs = children[i + 1]
            kw_kind = kw.type if isinstance(kw, LarkToken) else "KW_AND"
            result = ir.BinaryExpr(loc=result.loc, op=_BIN_OPS[kw_kind], lhs=result, rhs=rhs)
            i += 2
        return result

    def bool_or(self, children: list[Any]) -> ir.Expression:
        if len(children) == 1:
            return children[0]
        result = children[0]
        i = 1
        while i < len(children):
            kw = children[i]
            rhs = children[i + 1]
            kw_kind = kw.type if isinstance(kw, LarkToken) else "KW_OR"
            result = ir.BinaryExpr(loc=result.loc, op=_BIN_OPS[kw_kind], lhs=result, rhs=rhs)
            i += 2
        return result

    # comparison_op, add_op, mul_op — pass-through Trees so _fold_binary can
    # introspect the wrapped token. (Lark folds 1-child rules into the child
    # automatically unless we keep them as Tree explicitly; the default works
    # for us — we accept either form in _fold_binary.)
    def comparison_op(self, children: list[Any]) -> Any:
        return children[0]

    def add_op(self, children: list[Any]) -> Any:
        return children[0]

    def mul_op(self, children: list[Any]) -> Any:
        return children[0]

    # -- ternary -----------------------------------------------------------

    def ternary(self, children: list[Any]) -> ir.Expression:
        # children: [cond]  or  [cond, QMARK, then, COLON, else]
        exprs = [c for c in children if isinstance(c, ir.Expression)]
        if len(exprs) == 1:
            return exprs[0]
        cond, then_, else_ = exprs[0], exprs[1], exprs[2]
        return ir.TernaryExpr(loc=cond.loc, cond=cond, then_=then_, else_=else_)

    def expression(self, children: list[ir.Expression]) -> ir.Expression:
        return children[0]

    def bool_not(self, children: list[Any]) -> Any:
        return children[0]

    # -- type annotations --------------------------------------------------

    def qualifier(self, children: list[Any]) -> str:
        # children: a NAME token whose text should be one of the qualifier
        # words. We don't reserve those words at the lexer layer — see the
        # _KEYWORDS comment for why — so we accept whatever NAME got matched
        # here and default to ``simple`` if it isn't a real qualifier.
        tok = children[0]
        text = str(tok)
        if text in ("const", "input", "simple", "series"):
            return text
        return "simple"

    def type_inner(self, children: list[Any]) -> Any:
        # Just the leaf name for now; type_arg_list ignored for IR purposes
        # (C3 will refine when generics matter).
        name_tok = children[0]
        return ("inner", str(name_tok))

    def type_annotation(self, children: list[Any]) -> PineType:
        # Children: [maybe-qualifier-str, ("inner", name)]
        qualifier = "simple"
        inner_repr: tuple[str, str] | None = None
        for c in children:
            if isinstance(c, str):
                qualifier = c  # type: ignore[assignment]
            elif isinstance(c, tuple) and c and c[0] == "inner":
                inner_repr = c
        assert inner_repr is not None
        name = inner_repr[1]
        scalar_kinds = {"int", "float", "bool", "string", "color"}
        if name == "array":
            # Generic args ignored at this layer (placeholder shape)
            inner_type = ArrayT(element=PineType(qualifier="simple", inner=Scalar(kind="float")))
        elif name == "matrix":
            inner_type = MatrixT(element=PineType(qualifier="simple", inner=Scalar(kind="float")))
        elif name == "map":
            inner_type = MapT(
                key=PineType(qualifier="simple", inner=Scalar(kind="string")),
                value=PineType(qualifier="simple", inner=Scalar(kind="float")),
            )
        elif name in scalar_kinds:
            inner_type = Scalar(kind=name)  # type: ignore[arg-type]
        else:
            inner_type = UDT(name=name)
        return PineType(qualifier=qualifier, inner=inner_type)  # type: ignore[arg-type]

    def type_arg_list(self, children: list[Any]) -> list[PineType]:
        return list(children)

    # -- declarations ------------------------------------------------------

    def parameter(self, children: list[Any]) -> ir.Parameter:
        type_ann: PineType | None = None
        name_tok: LarkToken | None = None
        default: ir.Expression | None = None
        for c in children:
            if isinstance(c, PineType):
                type_ann = c
            elif isinstance(c, LarkToken) and c.type == "NAME":
                name_tok = c
            elif isinstance(c, ir.Expression):
                default = c
        assert name_tok is not None
        return ir.Parameter(
            loc=self._sp(name_tok),
            name=str(name_tok),
            type=type_ann or PineType(qualifier="simple", inner=Scalar(kind="float")),
            default=default,
        )

    def parameters(self, children: list[ir.Parameter]) -> list[ir.Parameter]:
        return list(children)

    def method_modifier(self, children: list[Any]) -> bool:
        return True

    def function_decl(self, children: list[Any]) -> ir.FunctionDecl:
        is_method = False
        name_tok: LarkToken | None = None
        params: list[ir.Parameter] = []
        body: tuple[ir.Statement, ...] = ()
        for c in children:
            if c is True:
                is_method = True
            elif isinstance(c, LarkToken) and c.type == "NAME" and name_tok is None:
                name_tok = c
            elif isinstance(c, list) and all(isinstance(p, ir.Parameter) for p in c):
                params = c
            elif isinstance(c, tuple) and c and isinstance(c[0], ir.Statement):
                body = c  # type: ignore[assignment]
            elif isinstance(c, ir.Statement):
                body = (c,)
            elif isinstance(c, ir.Expression):
                # `f(...) = expr` shorthand: lift expression into ExprStmt
                body = (ir.ExprStmt(loc=c.loc, expr=c),)
        assert name_tok is not None
        return ir.FunctionDecl(
            loc=self._sp(name_tok),
            name=str(name_tok),
            is_method=is_method,
            receiver=None,
            type_params=(),
            parameters=tuple(params),
            return_type=None,
            body=body,
        )

    def type_decl(self, children: list[Any]) -> ir.TypeDecl:
        # Children: KW_TYPE NAME [KW_EXPORT] NEWLINE INDENT [type_field+] DEDENT
        name_tok: LarkToken | None = None
        fields: list[ir.Parameter] = []
        for c in children:
            if isinstance(c, LarkToken) and c.type == "NAME" and name_tok is None:
                name_tok = c
            elif isinstance(c, ir.Parameter):
                fields.append(c)
        assert name_tok is not None
        return ir.TypeDecl(
            loc=self._sp(name_tok),
            name=str(name_tok),
            fields=tuple(fields),
            extends=None,
        )

    def type_field(self, children: list[Any]) -> ir.Parameter:
        type_ann: PineType | None = None
        name_tok: LarkToken | None = None
        default: ir.Expression | None = None
        for c in children:
            if isinstance(c, PineType):
                type_ann = c
            elif isinstance(c, LarkToken) and c.type == "NAME":
                name_tok = c
            elif isinstance(c, ir.Expression):
                default = c
        assert name_tok is not None
        return ir.Parameter(
            loc=self._sp(name_tok),
            name=str(name_tok),
            type=type_ann or PineType(qualifier="simple", inner=Scalar(kind="float")),
            default=default,
        )

    def enum_decl(self, children: list[Any]) -> ir.EnumDecl:
        name_tok: LarkToken | None = None
        members: list[tuple[str, ir.Expression | None]] = []
        for c in children:
            if isinstance(c, LarkToken) and c.type == "NAME" and name_tok is None:
                name_tok = c
            elif isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], str):
                members.append(c)
        assert name_tok is not None
        return ir.EnumDecl(loc=self._sp(name_tok), name=str(name_tok), members=tuple(members))

    def enum_member(self, children: list[Any]) -> tuple[str, ir.Expression | None]:
        name_tok = children[0]
        value: ir.Expression | None = None
        for c in children[1:]:
            if isinstance(c, ir.Expression):
                value = c
        return (str(name_tok), value)

    def declaration(self, children: list[ir.Declaration]) -> ir.Declaration:
        return children[0]

    # -- statements --------------------------------------------------------

    def var_qualifier(self, children: list[LarkToken]) -> str:
        kind = children[0].type  # KW_VAR or KW_VARIP
        return "var" if kind == "KW_VAR" else "varip"

    def var_decl(self, children: list[Any]) -> ir.VarDecl:
        qualifier: str | None = None
        type_ann: PineType | None = None
        name_tok: LarkToken | None = None
        value: ir.Expression | None = None
        for c in children:
            if isinstance(c, str) and c in ("var", "varip"):
                qualifier = c  # type: ignore[assignment]
            elif isinstance(c, PineType):
                type_ann = c
            elif isinstance(c, LarkToken) and c.type == "NAME":
                name_tok = c
            elif isinstance(c, ir.Expression):
                value = c
        assert name_tok is not None
        assert value is not None  # D1 §2.6 invariant 5: VarDecl.value non-None
        return ir.VarDecl(
            loc=self._sp(name_tok),
            qualifier=qualifier,  # type: ignore[arg-type]
            name=str(name_tok),
            type=type_ann,
            value=value,
        )

    def assign_target(self, children: list[Any]) -> ir.Expression:
        # NAME (attribute_tail | subscript_tail)*
        if isinstance(children[0], ir.Expression):
            return children[0]
        head_tok = children[0]
        node: ir.Expression = ir.Name(loc=self._sp(head_tok), id=str(head_tok))
        for tail in children[1:]:
            kind, payload, span = tail
            if kind == "attribute":
                node = ir.Attribute(loc=span, value=node, attr=payload)
            elif kind == "subscript":
                node = ir.Subscript(loc=span, value=node, index=payload, kind="index")
        return node

    def tuple_target(self, children: list[LarkToken]) -> ir.TupleExpr:
        elems = tuple(
            ir.Name(loc=self._sp(t), id=str(t))
            for t in children
            if isinstance(t, LarkToken) and t.type == "NAME"
        )
        return ir.TupleExpr(loc=elems[0].loc, elements=elems)

    def assignment_eq(self, children: list[Any]) -> ir.Assign:
        # children may be [target, ASSIGN_tok, value] or [target, value] depending
        # on whether lark filters anonymous-literal tokens. Pull the first
        # non-token expression as the value.
        target = children[0]
        value = next(c for c in children[1:] if isinstance(c, ir.Expression))
        return ir.Assign(loc=target.loc, target=target, op="=", value=value)

    def assignment_walrus(self, children: list[Any]) -> ir.Assign:
        target = children[0]
        value = next(c for c in children[1:] if isinstance(c, ir.Expression))
        return ir.Assign(loc=target.loc, target=target, op=":=", value=value)

    def return_stmt(self, children: list[Any]) -> ir.ReturnStmt:
        value = children[0] if children else None
        loc = value.loc if value is not None else self._fresh_span()
        return ir.ReturnStmt(loc=loc, value=value)

    def break_stmt(self, children: list[Any]) -> ir.Statement:
        # Use an ExprStmt(Name("break")) as a portable carrier; the IR
        # doesn't define BreakStmt/ContinueStmt — codegen reads .id when it
        # sees an ExprStmt with that special name. (Phase-2 polish: D1 IR
        # follow-up can promote these to first-class.)
        loc = self._fresh_span()
        return ir.ExprStmt(loc=loc, expr=ir.Name(loc=loc, id="break"))

    def continue_stmt(self, children: list[Any]) -> ir.Statement:
        loc = self._fresh_span()
        return ir.ExprStmt(loc=loc, expr=ir.Name(loc=loc, id="continue"))

    def expr_stmt(self, children: list[ir.Expression]) -> ir.ExprStmt:
        expr = children[0]
        return ir.ExprStmt(loc=expr.loc, expr=expr)

    def simple_stmt_body(self, children: list[Any]) -> ir.Statement:
        return children[0]

    def simple_stmt(self, children: list[Any]) -> list[ir.Statement]:
        # children: [stmt, (SEMICOLON, stmt)*, NEWLINE] — return all stmts
        return [c for c in children if isinstance(c, ir.Statement)]

    def compound_stmt(self, children: list[Any]) -> ir.Statement:
        return children[0]

    def statement(self, children: list[Any]) -> list[ir.Statement] | ir.Statement | None:
        # NEWLINE-only statements collapse to None; simple_stmt may return a
        # list of stmts (semicolon-joined); compound_stmt is a single Statement.
        c = children[0]
        if isinstance(c, LarkToken):
            return None
        return c

    def block(self, children: list[Any]) -> tuple[ir.Statement, ...]:
        # INDENT statement+ DEDENT  -or-  simple_stmt
        out: list[ir.Statement] = []
        for c in children:
            if isinstance(c, LarkToken):
                # INDENT / DEDENT — skip
                continue
            if c is None:
                continue
            if isinstance(c, list):
                # simple_stmt returns list[Statement] (semicolon-joined line)
                out.extend(c)
            elif isinstance(c, ir.Statement):
                out.append(c)
        return tuple(out)

    # -- if / for / while / switch ----------------------------------------

    def if_stmt(self, children: list[Any]) -> ir.IfStmt:
        # Build a flat list of meaningful children (no NEWLINEs).
        cond_expr: ir.Expression | None = None
        then_body: tuple[ir.Statement, ...] = ()
        elif_branches: list[tuple[ir.Expression, tuple[ir.Statement, ...]]] = []
        else_body: tuple[ir.Statement, ...] | None = None

        # The lark Earley parse tree threads the rule's children in source
        # order: KW_IF expr NEWLINE block (KW_ELSE KW_IF expr NEWLINE block)* (KW_ELSE NEWLINE block)?
        # Filter to non-token children + keyword tokens we care about.
        items = [c for c in children if not (isinstance(c, LarkToken) and c.type == "NEWLINE")]

        i = 0
        n = len(items)
        # First: KW_IF expr block
        if i < n and isinstance(items[i], LarkToken) and items[i].type == "KW_IF":
            i += 1
        cond_expr = items[i]; i += 1
        then_body = items[i]; i += 1
        # Remaining: KW_ELSE KW_IF cond block ... / KW_ELSE block
        while i < n:
            tok = items[i]
            if isinstance(tok, LarkToken) and tok.type == "KW_ELSE":
                i += 1
                if i < n and isinstance(items[i], LarkToken) and items[i].type == "KW_IF":
                    i += 1
                    elif_cond = items[i]; i += 1
                    elif_body = items[i]; i += 1
                    elif_branches.append((elif_cond, elif_body))
                else:
                    else_body = items[i]; i += 1
            else:
                i += 1

        return ir.IfStmt(
            loc=cond_expr.loc,
            cond=cond_expr,
            then_body=then_body,
            elif_branches=tuple(elif_branches),
            else_body=else_body,
        )

    def for_stmt(self, children: list[Any]) -> ir.ForStmt:
        # KW_FOR NAME ASSIGN expr KW_TO expr (KW_BY expr)? NEWLINE block
        items = [
            c
            for c in children
            if not (
                isinstance(c, LarkToken)
                and c.type in ("KW_FOR", "ASSIGN", "KW_TO", "KW_BY", "NEWLINE")
            )
        ]
        # items: NAME, start, end, [step], block
        name_tok = items[0]
        start = items[1]
        end = items[2]
        if len(items) == 5:
            step = items[3]; body = items[4]
        else:
            step = None; body = items[3]
        return ir.ForStmt(
            loc=self._sp(name_tok),
            var=str(name_tok),
            start=start,
            end=end,
            step=step,
            body=body,
        )

    def for_in_stmt(self, children: list[Any]) -> ir.ForInStmt:
        # Two cases: `for v in iter NEWLINE block` or `for [k, v] in iter NEWLINE block`
        items = [
            c
            for c in children
            if not (
                isinstance(c, LarkToken)
                and c.type in ("KW_FOR", "KW_IN", "NEWLINE", "LBRACKET", "RBRACKET", "COMMA")
            )
        ]
        # items: NAME [NAME] expr block  (we ignore tuple-destructure secondary name)
        name_tok = items[0]
        # Skip optional second NAME for tuple destructure
        idx = 1
        while idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == "NAME":
            idx += 1
        iterable = items[idx]; idx += 1
        body = items[idx]
        return ir.ForInStmt(
            loc=self._sp(name_tok),
            var=str(name_tok),
            iterable=iterable,
            body=body,
        )

    def while_stmt(self, children: list[Any]) -> ir.WhileStmt:
        items = [
            c
            for c in children
            if not (isinstance(c, LarkToken) and c.type in ("KW_WHILE", "NEWLINE"))
        ]
        cond = items[0]
        body = items[1]
        return ir.WhileStmt(loc=cond.loc, cond=cond, body=body)

    def switch_case(self, children: list[Any]) -> tuple[ir.Expression, tuple[ir.Statement, ...]]:
        # expr ARROW (block | expression NEWLINE)
        items = [
            c
            for c in children
            if not (isinstance(c, LarkToken) and c.type in ("ARROW", "NEWLINE"))
        ]
        case_expr = items[0]
        body_thing = items[1]
        if isinstance(body_thing, tuple):
            body = body_thing
        else:
            # bare expression case body
            body = (ir.ExprStmt(loc=body_thing.loc, expr=body_thing),)
        return (case_expr, body)

    def switch_default(self, children: list[Any]) -> tuple[None, tuple[ir.Statement, ...]]:
        items = [
            c
            for c in children
            if not (isinstance(c, LarkToken) and c.type in ("ARROW", "NEWLINE"))
        ]
        body_thing = items[0]
        if isinstance(body_thing, tuple):
            body = body_thing
        else:
            body = (ir.ExprStmt(loc=body_thing.loc, expr=body_thing),)
        return (None, body)

    def switch_stmt(self, children: list[Any]) -> ir.SwitchStmt:
        items = [
            c
            for c in children
            if not (isinstance(c, LarkToken) and c.type in ("KW_SWITCH", "NEWLINE", "INDENT", "DEDENT"))
        ]
        scrutinee: ir.Expression | None = None
        cases: list[tuple[ir.Expression | None, tuple[ir.Statement, ...]]] = []
        for c in items:
            if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], tuple):
                cases.append(c)
            elif isinstance(c, ir.Expression):
                scrutinee = c
        loc = scrutinee.loc if scrutinee is not None else self._fresh_span()
        return ir.SwitchStmt(loc=loc, scrutinee=scrutinee, cases=tuple(cases))

    # -- script directives -------------------------------------------------

    def _build_directive(self, kind: str, children: list[Any]) -> ir.ScriptDirective:
        # children: KW_*, LPAREN, [call_args], RPAREN
        kwargs: list[ir.KeywordArg] = []
        for c in children:
            if isinstance(c, list):
                kwargs.extend(c)
            elif isinstance(c, ir.KeywordArg):
                kwargs.append(c)
        # Extract title/shorttitle/overlay from the kwargs for the Directive.
        title = ""
        shorttitle: str | None = None
        overlay: bool | None = None
        for kw in kwargs:
            if kw.name is None and isinstance(kw.value, ir.StrLit) and not title:
                # first positional string is the title
                title = kw.value.value
            elif kw.name == "title" and isinstance(kw.value, ir.StrLit):
                title = kw.value.value
            elif kw.name == "shorttitle" and isinstance(kw.value, ir.StrLit):
                shorttitle = kw.value.value
            elif kw.name == "overlay" and isinstance(kw.value, ir.BoolLit):
                overlay = kw.value.value
        return ir.ScriptDirective(
            loc=self._fresh_span(),
            kind=kind,  # type: ignore[arg-type]
            title=title,
            shorttitle=shorttitle,
            overlay=overlay,
            arguments=tuple(kwargs),
        )

    def indicator_directive(self, children: list[Any]) -> ir.ScriptDirective:
        return self._build_directive("indicator", children)

    def strategy_directive(self, children: list[Any]) -> ir.ScriptDirective:
        return self._build_directive("strategy", children)

    def library_directive(self, children: list[Any]) -> ir.ScriptDirective:
        return self._build_directive("library", children)

    def script_directive(self, children: list[ir.ScriptDirective]) -> ir.ScriptDirective:
        return children[0]

    def top_item(self, children: list[Any]) -> list[Any] | ir.Statement | ir.Declaration | ir.ScriptDirective | None:
        c = children[0]
        if isinstance(c, LarkToken):
            return None
        return c

    def program(self, children: list[Any]) -> tuple[
        int | None,
        ir.ScriptDirective | None,
        list[ir.Declaration],
        list[ir.Statement],
    ]:
        version: int | None = None
        directive: ir.ScriptDirective | None = None
        decls: list[ir.Declaration] = []
        body: list[ir.Statement] = []
        for c in children:
            if isinstance(c, LarkToken) and c.type == "AT_VERSION":
                version = int(str(c))
            elif isinstance(c, LarkToken):
                # NEWLINE / EOF — skip
                continue
            elif c is None:
                continue
            elif isinstance(c, ir.ScriptDirective):
                directive = c
            elif isinstance(c, ir.Declaration):
                decls.append(c)
            elif isinstance(c, list):
                # semicolon-joined simple_stmt list
                body.extend(s for s in c if isinstance(s, ir.Statement))
            elif isinstance(c, ir.Statement):
                body.append(c)
        return version, directive, decls, body

    def start(self, children: list[Any]) -> Any:
        return children[0]


# ---------------------------------------------------------------------------
# Public surface — parse(tokens, *, pine_version) -> Program
# ---------------------------------------------------------------------------


def parse(tokens: list[PineToken], *, pine_version: int) -> ir.Program:
    """Parse a C1 Token stream into an IR :class:`Program` node.

    ``pine_version``: 5 or 6 — selects grammar file. v5 currently uses the
    v6 grammar as a placeholder (real v5->v6 migration shim is C7's bead).

    Raises :class:`openbb_pine.errors.PineSyntaxError` on parse failure,
    with a hint sourced from
    :meth:`lark.exceptions.UnexpectedToken.match_examples` when the failure
    matches a known pattern (D1 §1.3).
    """
    if not isinstance(tokens, list) or not tokens:
        raise PineSyntaxError("parse: tokens must be a non-empty list[Token]")
    parser = _lark_for_version(pine_version)
    try:
        tree = parser.parse(tokens)
    except UnexpectedToken as err:
        hint = _match_hint(err, pine_version) or ""
        line = err.line or 0
        col = err.column or 0
        tok = getattr(err, "token", None)
        tok_repr = repr(str(tok)) if tok is not None else "?"
        prefix = (
            f"unexpected token {tok_repr} at line {line}, col {col}"
        )
        msg = f"{prefix}\n  hint: {hint}" if hint else prefix
        raise PineSyntaxError(msg) from err
    except UnexpectedInput as err:
        # UnexpectedCharacters / UnexpectedEOF / other lark input failures.
        raise PineSyntaxError(f"parse error: {err}") from err
    except LarkError as err:
        # Catch-all for VisitError and other lark-side wrapper exceptions.
        raise PineSyntaxError(f"parse failed: {err}") from err

    # The Transformer wasn't auto-applied at parse time (we want post-parse
    # control). Apply it now.
    builder = _IRBuilder()
    result = builder.transform(tree)

    # `result` is whatever `start` returns -> `program()` tuple.
    version, directive, decls, body = result  # type: ignore[misc]
    if directive is None:
        # No directive: synthesize a stub indicator so downstream consumers
        # see a Program. The type checker will flag this in C3.
        directive = ir.ScriptDirective(
            loc=ir.Span(file="<inline>", start_line=1, start_col=1,
                        end_line=1, end_col=1, start_byte=0, end_byte=0),
            kind="indicator", title="", shorttitle=None,
            overlay=None, arguments=(),
        )
    # `version` comes from the //@version= pragma if any; else fall back
    # to the caller-supplied pine_version.
    final_version = version if version in (5, 6) else pine_version
    return ir.Program(
        loc=ir.Span(file="<inline>", start_line=1, start_col=1,
                    end_line=1, end_col=1, start_byte=0, end_byte=0),
        version=final_version,  # type: ignore[arg-type]
        directive=directive,
        declarations=tuple(decls),
        body=tuple(body),
    )
