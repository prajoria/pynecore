"""C3 — type checker (bead 0e9.5.3) per D1 §4.

Public surface:

* :class:`TypeCheckResult` — the contract carrier C5 (codegen) reads.
* :func:`check` — entry point. Lex+parse output → ``TypeCheckResult``.

What this implements (D1 §4.4 — eight type-checker rules, codes PT001-PT008):

* PT001 — `simple<T>` param cannot receive `series<T>` arg (e.g. `ta.sma(close,
  close)` — second arg must be simple<int>).
* PT002 — `var x = e` requires `e: simple<T>` (var initializers run once).
* PT003 — `if cond` / ternary cond must be `series<bool>` or `simple<bool>` —
  no implicit truthiness on int/float.
* PT004 — `x[n]` history access requires `x: series<T>`; `n` must be
  `simple<int>` or `const<int>`.
* PT005 — `:=` target must be declared; RHS qualifier <= LHS qualifier.
* PT006 — `na` propagation: any non-na op with na yields the inferred type
  of the other side.
* PT007 — v6 type args must satisfy the builtin signature.
* PT008 — UDT field access requires the field to exist.

Side responsibilities:

1. **Subscript.kind resolution** (D1 §2.6 invariant 3): the parser writes
   ``kind="history"`` as a default; this pass overwrites with the final
   resolution (history / array_index / map_lookup / tuple_index /
   matrix_index) so codegen forks correctly.
2. **builtins_used population** (D1 §3.1): every resolved builtin name (e.g.
   ``"ta.sma"``, ``"math.abs"``, ``"close"``) is recorded. Unsupported but
   real-Pine names (e.g. ``"ta.ichimoku"``) are ALSO recorded — they raise
   :class:`PineUnsupportedBuiltinError` but the wild-corpus coverage metric
   (PRD §3.4 L0.5) needs to credit the script for touching them.

Out of scope (per bead spec):

* Codegen (C5 — Wave 4).
* request.security routing (Phase 2).
* Typed param/var decls — parser doesn't surface them yet; if it does, we
  raise :class:`PineUnsupportedFeatureError` with code ``"PF002"``.

Implementation note re Subscript.kind: ``ir.Subscript`` is a frozen dataclass
so we can't mutate ``.kind`` in place. We use :func:`dataclasses.replace` to
build a refreshed Program in one pass — every parent that referenced an old
Subscript is rebuilt along the way. This keeps the codegen invariant
(Subscript.kind is set) without breaking immutability.

D1 amendment flag: D1 §2.6 invariant 3 enumerates only ``"history"`` /
``"index"`` (parser uses "history" only; ir.Subscript Literal type lists
``"history", "index"``). C3 needs finer-grained kinds to fork codegen
correctly. We use ``"history"`` / ``"array_index"`` / ``"map_lookup"`` /
``"tuple_index"`` / ``"matrix_index"`` and downcast to ``"index"`` for IR
storage when invariant 3 strictly requires it. Until D1 §2.6 amends, we
keep ``Subscript.kind`` as "history" or "index" (the two Literal values) and
expose the finer subkind via the sidecar types dict for codegen to introspect.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyne_compiler.compiler import ir
from pyne_compiler.compiler.builtin_signatures import (
    BUILTIN_SIGNATURES,
    Signature,
    is_builtin_namespace,
    lookup as lookup_builtin,
)
from pyne_compiler.compiler.types import (
    AnyT,
    ArrayT,
    InnerType,
    MapT,
    MatrixT,
    NaT,
    PineType,
    Qualifier,
    Reference,
    Scalar,
    SecurityContext,
    TupleT,
    UDT,
    UnknownT,
    can_promote,
)
from pyne_compiler.errors.base import (
    PineTypeError,
    PineUnsupportedBuiltinError,
    PineUnsupportedFeatureError,
)

if TYPE_CHECKING:  # pragma: no cover — imports for typing only
    # Post-9bh: telemetry now lives at pyne_compiler.telemetry (extraction
    # complete). Kept behind TYPE_CHECKING so the compiler never actually
    # imports telemetry at runtime — the E0.4 injection contract.
    from pyne_compiler.telemetry import TelemetrySink

__all__ = ["TypeCheckResult", "check"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeCheckResult:
    """C3's contract — what C5 (codegen) consumes (D1 §3.1)."""

    program: ir.Program
    """The IR with ``Subscript.kind`` resolved per D1 §2.6 invariant 3."""

    builtins_used: frozenset[str]
    """Pine builtin identifiers the script references; flows into
    ``CompiledModule.builtins_used``. Includes unsupported-but-real-Pine
    names so PRD §3.4 L0.5 wild-corpus coverage credits the request."""

    security_contexts: dict[str, SecurityContext] | None
    """Lowered ``request.security`` directives, one entry per call site,
    keyed by a deterministic ``"ctx_N"`` id (N is the source-order index).
    ``None`` when the script has no ``request.security`` calls (Phase-1
    scripts and any indicator that only touches primary series). Populated
    by bead ``0e9.6.y86`` per D5 §4.1 + §7.2."""

    diagnostics: tuple[Any, ...]
    """Non-fatal warnings (e.g. unused var). Errors raise; never reach here."""


# ---------------------------------------------------------------------------
# Common shorthands
# ---------------------------------------------------------------------------


_SERIES_FLOAT = PineType(qualifier="series", inner=Scalar(kind="float"))
_SERIES_INT = PineType(qualifier="series", inner=Scalar(kind="int"))
_SERIES_BOOL = PineType(qualifier="series", inner=Scalar(kind="bool"))
_SIMPLE_INT = PineType(qualifier="simple", inner=Scalar(kind="int"))
_SIMPLE_FLOAT = PineType(qualifier="simple", inner=Scalar(kind="float"))
_SIMPLE_BOOL = PineType(qualifier="simple", inner=Scalar(kind="bool"))
_SIMPLE_STRING = PineType(qualifier="simple", inner=Scalar(kind="string"))
_CONST_INT = PineType(qualifier="const", inner=Scalar(kind="int"))
_CONST_FLOAT = PineType(qualifier="const", inner=Scalar(kind="float"))
_CONST_BOOL = PineType(qualifier="const", inner=Scalar(kind="bool"))
_CONST_STRING = PineType(qualifier="const", inner=Scalar(kind="string"))
_NA = PineType(qualifier="const", inner=NaT())


_RANK: dict[str, int] = {"const": 0, "input": 1, "simple": 2, "series": 3}


def _max_qual(a: Qualifier, b: Qualifier) -> Qualifier:
    return a if _RANK[a] >= _RANK[b] else b


# ---------------------------------------------------------------------------
# IR → Pine-source serializer (bead 0e9.6.y86 review — comment ID 3522846218)
# ---------------------------------------------------------------------------


def _serialize_ir_expr(node: ir.Expression) -> str:
    """Render an IR expression back to a stable, human-readable Pine-like
    source string.

    Used by ``_visit_request_security`` to fill
    :attr:`SecurityContext.expr` and the ``symbol``/``timeframe`` fields
    for dynamic-form calls. The prior implementation used ``str(node)``
    which yielded the frozen-dataclass ``repr()`` (``"Name(loc=Span(...),
    id='close')"``) — soup unsuitable for bug reports OR for the dynamic
    dispatcher path (D5 §4.4) where ``ctx.symbol`` gets forwarded to
    ``_fetch_via_fmp``.

    The output is intentionally minimal:

    * Idempotent under re-serialization of the same tree.
    * No location metadata (the D5 §4.1 contract calls this an *opaque*
      field but tests + operators read it).
    * Falls back to the frozen-dataclass ``repr()`` for any node kind we
      haven't enumerated — always yields a string so the caller never
      races on a KeyError.

    Kept small and pure on purpose: it re-implements the tiny subset of
    Pine unparse we care about for the ``request.security`` collector.
    Full Pine unparse is a codegen concern (C5), not a type-checker
    concern; a separate future bead can lift this into a general utility.
    """
    if isinstance(node, ir.StrLit):
        return f'"{node.value}"'
    if isinstance(node, ir.IntLit):
        return str(node.value)
    if isinstance(node, ir.FloatLit):
        return repr(node.value)
    if isinstance(node, ir.BoolLit):
        return "true" if node.value else "false"
    if isinstance(node, ir.NaLit):
        return "na"
    if isinstance(node, ir.ColorLit):
        return node.raw
    if isinstance(node, ir.Name):
        return node.id
    if isinstance(node, ir.Attribute):
        return f"{_serialize_ir_expr(node.value)}.{node.attr}"
    if isinstance(node, ir.Subscript):
        return f"{_serialize_ir_expr(node.value)}[{_serialize_ir_expr(node.index)}]"
    if isinstance(node, ir.UnaryExpr):
        # Emit without inner parens; Pine's precedence rules make this
        # unambiguous for the ``+ / - / not`` set.
        return f"{node.op}{_serialize_ir_expr(node.operand)}"
    if isinstance(node, ir.BinaryExpr):
        # Wrap in parens so precedence surprises don't change meaning if the
        # string is ever re-parsed. Safe under idempotence.
        return (
            f"({_serialize_ir_expr(node.lhs)} {node.op} "
            f"{_serialize_ir_expr(node.rhs)})"
        )
    if isinstance(node, ir.TernaryExpr):
        return (
            f"({_serialize_ir_expr(node.cond)} ? "
            f"{_serialize_ir_expr(node.then_)} : "
            f"{_serialize_ir_expr(node.else_)})"
        )
    if isinstance(node, ir.CallExpr):
        parts: list[str] = []
        for a in node.args:
            piece = _serialize_ir_expr(a.value)
            parts.append(piece if a.name is None else f"{a.name}={piece}")
        return f"{_serialize_ir_expr(node.func)}({', '.join(parts)})"
    if isinstance(node, ir.TupleExpr):
        return f"[{', '.join(_serialize_ir_expr(e) for e in node.elements)}]"
    # Fallback — retains the previous behaviour for any unknown node so we
    # never surface a KeyError from a serialisation hole. The frozen
    # dataclass repr is at least deterministic.
    return repr(node)


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------


class _TypeChecker:
    """Walks the IR, populates builtins_used, resolves Subscript.kind, and
    raises PT###/PU###/PF### on rule violations.

    Adding a 9th rule = one new method. Each PT00x rule has a dotted-name
    handler (``_rule_pt001_simple_arg_cannot_be_series`` etc.) so locating
    the rule body from a stack trace is one grep.
    """

    def __init__(
        self,
        *,
        pine_version: int,
        telemetry: "TelemetrySink | None" = None,
    ) -> None:
        self.pine_version = pine_version
        # Injected telemetry sink (E0.4). ``None`` = telemetry disabled;
        # every ``_raise_unsupported_*`` guard short-circuits and no
        # counter fires. The routers pass an ``OpenBBTelemetrySink``
        # per request so per-request counts stay isolated.
        self._telemetry = telemetry
        # Lexical scope chain: name -> PineType. The current scope is the LAST
        # element; lookups walk back to front.
        self._scopes: list[dict[str, PineType]] = [{}]
        # Builtins actually referenced (qualified names). Includes unsupported
        # ones — they still count toward PRD §3.4 L0.5 coverage.
        self._builtins_used: set[str] = set()
        # Subscript-kind overrides per node id. Used when we rebuild the IR
        # so codegen sees the correct kind without us mutating frozen nodes.
        self._subscript_kind: dict[int, str] = {}
        # request.security lowering (bead 0e9.6.y86 — D5 §4.1). Each call
        # site gets a stable ``ctx_N`` id in source order so cache keys
        # stay deterministic across recompiles.
        self._security_contexts: dict[str, SecurityContext] = {}
        self._security_ctx_counter: int = 0

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _push_scope(self) -> None:
        self._scopes.append({})

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def _declare(self, name: str, t: PineType) -> None:
        self._scopes[-1][name] = t

    def _resolve_scope(self, name: str) -> PineType | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _loc(self, node: ir.Node) -> tuple[str, int, int]:
        sp = node.loc
        return (sp.file, sp.start_line, sp.start_col)

    def _raise_unsupported_builtin(
        self, name: str, node: ir.Node, hint: str | None = None
    ) -> None:
        """Collect-then-raise: ensure the name lands in builtins_used before
        the exception unwinds, so an outer catch sees the partial coverage."""
        self._builtins_used.add(name)
        # PRD §3.4 L0.5 wild-corpus attribution: record BEFORE raising so
        # the counter is accurate even when an outer ``except`` swallows.
        # E0.4: telemetry is optional; when unset the counter is skipped
        # but the raise still happens.
        if self._telemetry is not None:
            self._telemetry.record_unsupported_builtin(name)
        exc = PineUnsupportedBuiltinError(
            name,
            suggested_alternative=hint,
            tracking_url=(
                f"https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                f"?labels=pine-builtin&q={name}"
            ),
        )
        # Attach the partial builtins_used set so an outer caller (e.g. test
        # harness) can report what coverage was achieved.
        exc.builtins_used = frozenset(self._builtins_used)  # type: ignore[attr-defined]
        raise exc

    def _raise_type_error(
        self,
        *,
        rule: str,
        expected: PineType | str | None,
        got: PineType | str | None,
        node: ir.Node,
        hint: str | None = None,
    ) -> None:
        raise PineTypeError(
            rule=rule,
            expected=expected,
            got=got,
            location=self._loc(node),
            hint=hint,
        )

    # ------------------------------------------------------------------
    # Program walk
    # ------------------------------------------------------------------

    def check_program(self, prog: ir.Program) -> ir.Program:
        # Walk declarations first so functions/types are visible to body.
        new_decls = tuple(self._visit_decl(d) for d in prog.declarations)
        new_body = tuple(self._visit_stmt(s) for s in prog.body)
        return dataclasses.replace(prog, declarations=new_decls, body=new_body)

    # ------------------------------------------------------------------
    # Declaration walk
    # ------------------------------------------------------------------

    def _visit_decl(self, decl: ir.Declaration) -> ir.Declaration:
        if isinstance(decl, ir.FunctionDecl):
            # Register the function name in scope as a callable. We don't
            # type-check the body deeply in Phase 1 — that's later beads'
            # work. But we do walk body to populate builtins_used.
            self._declare(
                decl.name,
                PineType(qualifier="simple", inner=UDT(name=f"__fn_{decl.name}")),
            )
            self._push_scope()
            for p in decl.parameters:
                self._declare(p.name, p.type)
            new_body = tuple(self._visit_stmt(s) for s in decl.body)
            self._pop_scope()
            return dataclasses.replace(decl, body=new_body)
        if isinstance(decl, ir.TypeDecl):
            # Register the UDT type name; field-existence checks happen at
            # attribute-access time via the scope's UDT entry.
            self._declare(
                decl.name,
                PineType(qualifier="simple", inner=UDT(name=decl.name)),
            )
            return decl
        if isinstance(decl, ir.EnumDecl):
            self._declare(
                decl.name,
                PineType(qualifier="const", inner=UDT(name=decl.name)),
            )
            return decl
        return decl

    # ------------------------------------------------------------------
    # Statement walk
    # ------------------------------------------------------------------

    def _visit_stmt(self, stmt: ir.Statement) -> ir.Statement:
        if isinstance(stmt, ir.VarDecl):
            return self._visit_var_decl(stmt)
        if isinstance(stmt, ir.Assign):
            return self._visit_assign(stmt)
        if isinstance(stmt, ir.IfStmt):
            return self._visit_if(stmt)
        if isinstance(stmt, ir.ForStmt):
            return self._visit_for(stmt)
        if isinstance(stmt, ir.ForInStmt):
            return self._visit_for_in(stmt)
        if isinstance(stmt, ir.WhileStmt):
            return self._visit_while(stmt)
        if isinstance(stmt, ir.SwitchStmt):
            return self._visit_switch(stmt)
        if isinstance(stmt, ir.ReturnStmt):
            if stmt.value is not None:
                new_val, _ = self._visit_expr(stmt.value)
                return dataclasses.replace(stmt, value=new_val)
            return stmt
        if isinstance(stmt, ir.ExprStmt):
            return self._visit_expr_stmt(stmt)
        return stmt

    def _visit_var_decl(self, stmt: ir.VarDecl) -> ir.VarDecl:
        # PF002 — typed decls in body deferred per bead spec.
        if stmt.type is not None:
            if self._telemetry is not None:
                self._telemetry.record_unsupported_feature("PF002")
            raise PineUnsupportedFeatureError(
                "PF002 typed decl in body",
                tracking_url=(
                    "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                    "?labels=pine-feature&q=typed-decl"
                ),
            )
        new_val, val_type = self._visit_expr(stmt.value)
        # PT002 — `var x = e` requires e: simple<T> (or weaker).
        if stmt.qualifier in ("var", "varip"):
            if not can_promote(val_type.qualifier, "simple"):
                self._raise_type_error(
                    rule="PT002",
                    expected=PineType(qualifier="simple", inner=val_type.inner),
                    got=val_type,
                    node=stmt,
                    hint=(
                        f"`{stmt.qualifier}` initializers run once — the RHS "
                        "must be simple<T> (or const<T>), not series<T>."
                    ),
                )
        # Register the binding.
        self._declare(stmt.name, val_type)
        return dataclasses.replace(stmt, value=new_val)

    def _visit_assign(self, stmt: ir.Assign) -> ir.Assign:
        new_val, val_type = self._visit_expr(stmt.value)
        if stmt.op == ":=":
            # PT005 — target must already be declared.
            if isinstance(stmt.target, ir.Name):
                existing = self._resolve_scope(stmt.target.id)
                if existing is None:
                    self._raise_type_error(
                        rule="PT005",
                        expected="declared name",
                        got=f"undeclared {stmt.target.id!r}",
                        node=stmt,
                        hint=(
                            "`:=` reassigns an existing name. Use `=` to "
                            "declare it first."
                        ),
                    )
                # PT005 also: RHS qualifier <= LHS qualifier (no demoting up).
                # (Our lattice convention: LHS qual must be >= RHS qual so the
                # storage slot can hold what the RHS produces. Pine's rule is
                # actually "the binding cannot weaken": once `simple`, never
                # `const`. We allow promotion (e.g. `x` simple, x := series x.
                # Phase-1 stub: any qualifier accepted — we just verify the
                # name exists. Tighter rule lands in a later bead once the
                # Pine spec's exact wording is encoded.)
                # Bare Name targets in `:=` don't need a second walk.
                new_target = stmt.target
            else:
                # Subscript/Attribute target — walk to set Subscript.kind etc.
                new_target, _ = self._visit_expr(stmt.target)
            return dataclasses.replace(stmt, target=new_target, value=new_val)
        # `=` — declare-or-reassign.
        if isinstance(stmt.target, ir.Name):
            # The target name is being introduced; don't look it up.
            self._declare(stmt.target.id, val_type)
            new_target = stmt.target
        elif isinstance(stmt.target, ir.TupleExpr):
            # Tuple destructuring: bind each element name to the RHS tuple's
            # corresponding type (or _SERIES_FLOAT if RHS isn't a TupleT).
            elem_types: tuple[PineType, ...] = (
                val_type.inner.elements if isinstance(val_type.inner, TupleT)
                else (_SERIES_FLOAT,) * len(stmt.target.elements)
            )
            for i, elem in enumerate(stmt.target.elements):
                if isinstance(elem, ir.Name):
                    t = (
                        elem_types[i] if i < len(elem_types)
                        else _SERIES_FLOAT
                    )
                    self._declare(elem.id, t)
            new_target = stmt.target
        else:
            # Subscript/Attribute target — walk normally so any UDT field /
            # array element is resolved.
            new_target, _ = self._visit_expr(stmt.target)
        return dataclasses.replace(stmt, target=new_target, value=new_val)

    def _visit_if(self, stmt: ir.IfStmt) -> ir.IfStmt:
        new_cond, cond_type = self._visit_expr(stmt.cond)
        self._rule_pt003_bool_cond(cond_type, stmt.cond)
        new_then = tuple(self._visit_stmt(s) for s in stmt.then_body)
        new_elifs: list[tuple[ir.Expression, tuple[ir.Statement, ...]]] = []
        for (ec, eb) in stmt.elif_branches:
            new_ec, ec_type = self._visit_expr(ec)
            self._rule_pt003_bool_cond(ec_type, ec)
            new_eb = tuple(self._visit_stmt(s) for s in eb)
            new_elifs.append((new_ec, new_eb))
        new_else = None
        if stmt.else_body is not None:
            new_else = tuple(self._visit_stmt(s) for s in stmt.else_body)
        return dataclasses.replace(
            stmt,
            cond=new_cond,
            then_body=new_then,
            elif_branches=tuple(new_elifs),
            else_body=new_else,
        )

    def _visit_for(self, stmt: ir.ForStmt) -> ir.ForStmt:
        new_start, _ = self._visit_expr(stmt.start)
        new_end, _ = self._visit_expr(stmt.end)
        new_step = None
        if stmt.step is not None:
            new_step, _ = self._visit_expr(stmt.step)
        # Loop var binds simple<int>.
        self._push_scope()
        self._declare(stmt.var, _SIMPLE_INT)
        new_body = tuple(self._visit_stmt(s) for s in stmt.body)
        self._pop_scope()
        return dataclasses.replace(
            stmt, start=new_start, end=new_end, step=new_step, body=new_body
        )

    def _visit_for_in(self, stmt: ir.ForInStmt) -> ir.ForInStmt:
        new_iter, iter_type = self._visit_expr(stmt.iterable)
        # Loop var's type is the iterable's element type (or simple<float>).
        elem_t: PineType
        if isinstance(iter_type.inner, ArrayT):
            elem_t = iter_type.inner.element
        else:
            elem_t = _SIMPLE_FLOAT
        self._push_scope()
        self._declare(stmt.var, elem_t)
        new_body = tuple(self._visit_stmt(s) for s in stmt.body)
        self._pop_scope()
        return dataclasses.replace(stmt, iterable=new_iter, body=new_body)

    def _visit_while(self, stmt: ir.WhileStmt) -> ir.WhileStmt:
        new_cond, cond_type = self._visit_expr(stmt.cond)
        self._rule_pt003_bool_cond(cond_type, stmt.cond)
        self._push_scope()
        new_body = tuple(self._visit_stmt(s) for s in stmt.body)
        self._pop_scope()
        return dataclasses.replace(stmt, cond=new_cond, body=new_body)

    def _visit_switch(self, stmt: ir.SwitchStmt) -> ir.SwitchStmt:
        new_scrut = None
        if stmt.scrutinee is not None:
            new_scrut, _ = self._visit_expr(stmt.scrutinee)
        new_cases: list[tuple[ir.Expression | None, tuple[ir.Statement, ...]]] = []
        for (ck, cb) in stmt.cases:
            new_ck = None
            if ck is not None:
                new_ck, _ = self._visit_expr(ck)
            new_cb = tuple(self._visit_stmt(s) for s in cb)
            new_cases.append((new_ck, new_cb))
        return dataclasses.replace(stmt, scrutinee=new_scrut, cases=tuple(new_cases))

    def _visit_expr_stmt(self, stmt: ir.ExprStmt) -> ir.ExprStmt:
        # Parser writes ExprStmt(Name("break"|"continue")) as a carrier — keep
        # them as void; don't treat the name as an identifier lookup.
        if isinstance(stmt.expr, ir.Name) and stmt.expr.id in ("break", "continue"):
            return stmt
        new_expr, _ = self._visit_expr(stmt.expr)
        return dataclasses.replace(stmt, expr=new_expr)

    # ------------------------------------------------------------------
    # Expression walk
    # ------------------------------------------------------------------

    def _visit_expr(self, expr: ir.Expression) -> tuple[ir.Expression, PineType]:
        if isinstance(expr, ir.IntLit):
            return expr, _CONST_INT
        if isinstance(expr, ir.FloatLit):
            return expr, _CONST_FLOAT
        if isinstance(expr, ir.BoolLit):
            return expr, _CONST_BOOL
        if isinstance(expr, ir.StrLit):
            return expr, _CONST_STRING
        if isinstance(expr, ir.NaLit):
            return expr, _NA
        if isinstance(expr, ir.ColorLit):
            return expr, PineType(qualifier="const", inner=Scalar(kind="color"))
        if isinstance(expr, ir.Name):
            return self._visit_name(expr)
        if isinstance(expr, ir.Attribute):
            return self._visit_attribute(expr)
        if isinstance(expr, ir.Subscript):
            return self._visit_subscript(expr)
        if isinstance(expr, ir.UnaryExpr):
            new_op, op_t = self._visit_expr(expr.operand)
            new_node = dataclasses.replace(expr, operand=new_op)
            # `not` requires bool; arithmetic unary keeps the inner kind.
            if expr.op == "not":
                if not (isinstance(op_t.inner, Scalar) and op_t.inner.kind == "bool") \
                        and not isinstance(op_t.inner, NaT):
                    self._raise_type_error(
                        rule="PT003",
                        expected="series<bool>|simple<bool>|const<bool>",
                        got=op_t,
                        node=expr,
                        hint="`not` requires a bool operand.",
                    )
                return new_node, PineType(qualifier=op_t.qualifier, inner=Scalar(kind="bool"))
            return new_node, op_t
        if isinstance(expr, ir.BinaryExpr):
            return self._visit_binary(expr)
        if isinstance(expr, ir.TernaryExpr):
            new_cond, cond_t = self._visit_expr(expr.cond)
            self._rule_pt003_bool_cond(cond_t, expr.cond)
            new_then, then_t = self._visit_expr(expr.then_)
            new_else, else_t = self._visit_expr(expr.else_)
            # Result qualifier is max(then, else); inner unifies, with na as id.
            result_qual = _max_qual(then_t.qualifier, else_t.qualifier)
            result_inner = self._unify_inner(then_t.inner, else_t.inner, expr)
            return (
                dataclasses.replace(expr, cond=new_cond, then_=new_then, else_=new_else),
                PineType(qualifier=result_qual, inner=result_inner),
            )
        if isinstance(expr, ir.CallExpr):
            return self._visit_call(expr)
        if isinstance(expr, ir.TupleExpr):
            new_elems: list[ir.Expression] = []
            elem_types: list[PineType] = []
            for e in expr.elements:
                ne, t = self._visit_expr(e)
                new_elems.append(ne)
                elem_types.append(t)
            return (
                dataclasses.replace(expr, elements=tuple(new_elems)),
                PineType(qualifier="simple", inner=TupleT(elements=tuple(elem_types))),
            )
        # Fallback: unknown expression node.
        return expr, PineType(qualifier="simple", inner=UnknownT(var_id=-1))

    def _visit_name(self, expr: ir.Name) -> tuple[ir.Expression, PineType]:
        # 1. Lexical scope first.
        t = self._resolve_scope(expr.id)
        if t is not None:
            return expr, t
        # 2. Builtin registry.
        sig = lookup_builtin(expr.id)
        if sig is not None:
            self._builtins_used.add(expr.id)
            return expr, sig.returns
        # 3. Pine namespace prefix (bare reference, e.g. `ta`) — record but
        # the actual use is via an Attribute. A bare namespace identifier
        # is rare; we treat it as an unknown identifier so the user gets
        # a clear error.
        # 4. Unknown.
        self._raise_type_error(
            rule="undefined",
            expected="declared name or known builtin",
            got=f"undefined identifier {expr.id!r}",
            node=expr,
            hint=(
                "Identifier is neither declared in scope nor a known Pine "
                "builtin. Check the spelling or declare it first."
            ),
        )
        # Unreachable — _raise_type_error raises.
        return expr, PineType(qualifier="simple", inner=UnknownT(var_id=-1))  # pragma: no cover

    def _visit_attribute(self, expr: ir.Attribute) -> tuple[ir.Expression, PineType]:
        # Qualified name resolution: `ta.sma` -> "ta.sma" lookup; if the
        # prefix is a real Pine namespace but the full name isn't in the
        # registry, raise PU and still record the name in builtins_used.
        qname = self._qualified_name(expr)
        if qname is not None:
            sig = lookup_builtin(qname)
            if sig is not None:
                self._builtins_used.add(qname)
                return expr, sig.returns
            # Real namespace but unknown member?
            prefix = qname.split(".", 1)[0]
            if is_builtin_namespace(prefix):
                self._raise_unsupported_builtin(qname, expr)
                # Unreachable.
                return expr, _SERIES_FLOAT  # pragma: no cover
            # Unknown namespace — undefined identifier.
            self._raise_type_error(
                rule="undefined",
                expected="known Pine namespace or UDT field",
                got=f"undefined attribute {qname!r}",
                node=expr,
                hint=(
                    "Neither the prefix is a known Pine namespace nor is "
                    "the receiver a UDT in scope."
                ),
            )
        # Non-qualified attribute — UDT field access.
        # PT008: walk receiver type; if UDT, the field lookup happens during
        # codegen. Stub: accept any attribute on a UDT; otherwise raise.
        new_recv, recv_t = self._visit_expr(expr.value)
        if isinstance(recv_t.inner, UDT):
            return (
                dataclasses.replace(expr, value=new_recv),
                PineType(qualifier=recv_t.qualifier, inner=UnknownT(var_id=-1)),
            )
        self._raise_type_error(
            rule="PT008",
            expected="UDT instance with field",
            got=recv_t,
            node=expr,
            hint=(
                "Attribute access requires the receiver to be a UDT (or a "
                "known Pine namespace)."
            ),
        )
        return expr, _SERIES_FLOAT  # pragma: no cover

    def _visit_subscript(self, expr: ir.Subscript) -> tuple[ir.Expression, PineType]:
        new_val, val_t = self._visit_expr(expr.value)
        new_idx, idx_t = self._visit_expr(expr.index)
        # Resolve kind based on receiver's inner type (D1 §2.6 invariant 3).
        new_kind: str
        result_t: PineType
        if isinstance(val_t.inner, ArrayT):
            new_kind = "index"  # IR Literal restricts to "history" | "index"
            result_t = val_t.inner.element
        elif isinstance(val_t.inner, MapT):
            new_kind = "index"
            result_t = val_t.inner.value
        elif isinstance(val_t.inner, MatrixT):
            new_kind = "index"
            result_t = val_t.inner.element
        elif isinstance(val_t.inner, TupleT):
            new_kind = "index"
            # Best-effort: use first element's type as the result.
            result_t = (
                val_t.inner.elements[0] if val_t.inner.elements else _SERIES_FLOAT
            )
        elif val_t.qualifier == "series":
            new_kind = "history"
            # PT004 — n must be simple<int> or const<int>.
            if not (
                isinstance(idx_t.inner, Scalar)
                and idx_t.inner.kind == "int"
                and can_promote(idx_t.qualifier, "simple")
            ) and not isinstance(idx_t.inner, NaT):
                self._raise_type_error(
                    rule="PT004",
                    expected="simple<int>|const<int>",
                    got=idx_t,
                    node=expr,
                    hint="History access index must be a non-series int.",
                )
            # Result type is the same as receiver (history of series<T> is
            # series<T>).
            result_t = val_t
        else:
            # Receiver is neither series nor an indexable inner type.
            self._raise_type_error(
                rule="PT004",
                expected="series<T> | array<T> | map<K,V> | matrix<T>",
                got=val_t,
                node=expr,
                hint=(
                    "Subscript requires a series (history access) or an "
                    "array/map/matrix (index access)."
                ),
            )
            new_kind = "history"  # pragma: no cover
            result_t = _SERIES_FLOAT  # pragma: no cover
        new_node = dataclasses.replace(expr, value=new_val, index=new_idx, kind=new_kind)
        return new_node, result_t

    def _visit_binary(self, expr: ir.BinaryExpr) -> tuple[ir.Expression, PineType]:
        new_lhs, lt = self._visit_expr(expr.lhs)
        new_rhs, rt = self._visit_expr(expr.rhs)
        # Logical ops require bool; arithmetic ops require numeric; comparison
        # produces bool.
        if expr.op in ("and", "or"):
            if not self._is_bool_or_na(lt) or not self._is_bool_or_na(rt):
                self._raise_type_error(
                    rule="PT003",
                    expected="bool",
                    got=f"{lt} {expr.op} {rt}",
                    node=expr,
                    hint=f"`{expr.op}` requires bool operands.",
                )
            qual = _max_qual(lt.qualifier, rt.qualifier)
            return (
                dataclasses.replace(expr, lhs=new_lhs, rhs=new_rhs),
                PineType(qualifier=qual, inner=Scalar(kind="bool")),
            )
        if expr.op in ("==", "!=", "<", "<=", ">", ">="):
            qual = _max_qual(lt.qualifier, rt.qualifier)
            return (
                dataclasses.replace(expr, lhs=new_lhs, rhs=new_rhs),
                PineType(qualifier=qual, inner=Scalar(kind="bool")),
            )
        # Arithmetic: +, -, *, /, %. Inner promotes to float on / and across
        # int/float mixes; na propagates.
        qual = _max_qual(lt.qualifier, rt.qualifier)
        result_inner = self._unify_inner(lt.inner, rt.inner, expr)
        # /, %, * with float operand widens to float (Pine's rule).
        if isinstance(result_inner, Scalar) and result_inner.kind in ("int",):
            if expr.op == "/" or (
                isinstance(lt.inner, Scalar) and lt.inner.kind == "float"
            ) or (
                isinstance(rt.inner, Scalar) and rt.inner.kind == "float"
            ):
                result_inner = Scalar(kind="float")
        return (
            dataclasses.replace(expr, lhs=new_lhs, rhs=new_rhs),
            PineType(qualifier=qual, inner=result_inner),
        )

    def _visit_call(self, expr: ir.CallExpr) -> tuple[ir.Expression, PineType]:
        # Special-case request.security BEFORE the standard signature lookup so
        # its dynamic-symbol / dynamic-timeframe args (e.g. ``syminfo.ticker``,
        # ``syminfo.timeframe``) don't get rejected by the generic Attribute
        # visitor (which would raise PU for unsupported ``syminfo.*``). See
        # bead 0e9.6.y86 + D5 §4.1 / §4.4.
        if isinstance(expr.func, ir.Attribute):
            qname_probe = self._qualified_name(expr.func)
            if qname_probe == "request.security":
                return self._visit_request_security(expr, qname_probe)
        # Resolve callee.
        callee = expr.func
        sig: Signature | None = None
        qname: str | None = None
        if isinstance(callee, ir.Name):
            qname = callee.id
            sig = lookup_builtin(qname)
            if sig is None:
                # User-defined function? Or known namespace name being called?
                # User-defined: look up in scope; if it's a __fn_ UDT marker,
                # accept and don't enforce arg types in Phase 1.
                t = self._resolve_scope(qname)
                if t is not None and isinstance(t.inner, UDT) and t.inner.name.startswith("__fn_"):
                    new_args = tuple(
                        dataclasses.replace(a, value=self._visit_expr(a.value)[0])
                        for a in expr.args
                    )
                    return dataclasses.replace(expr, args=new_args), _SERIES_FLOAT
                # Bare name not in scope and not in registry — undefined.
                self._raise_type_error(
                    rule="undefined",
                    expected="declared function or known builtin",
                    got=f"undefined call target {qname!r}",
                    node=expr,
                )
                return expr, _SERIES_FLOAT  # pragma: no cover
            self._builtins_used.add(qname)
        elif isinstance(callee, ir.Attribute):
            qname = self._qualified_name(callee)
            if qname is None:
                self._raise_type_error(
                    rule="undefined",
                    expected="qualified builtin name",
                    got="non-qualified attribute call",
                    node=expr,
                )
                return expr, _SERIES_FLOAT  # pragma: no cover
            sig = lookup_builtin(qname)
            if sig is None:
                prefix = qname.split(".", 1)[0]
                if is_builtin_namespace(prefix):
                    # Walk the args first so any nested unsupported builtins
                    # also get logged. Then raise the unsupported-builtin
                    # error AFTER the partial set is updated.
                    for a in expr.args:
                        # Best-effort: swallow inner errors so the outer raise
                        # captures the right name.
                        try:
                            self._visit_expr(a.value)
                        except (PineTypeError, PineUnsupportedBuiltinError, PineUnsupportedFeatureError):
                            pass
                    self._raise_unsupported_builtin(qname, expr)
                    return expr, _SERIES_FLOAT  # pragma: no cover
                self._raise_type_error(
                    rule="undefined",
                    expected="known Pine namespace",
                    got=f"unknown namespace in {qname!r}",
                    node=expr,
                )
                return expr, _SERIES_FLOAT  # pragma: no cover
            self._builtins_used.add(qname)
        else:
            # Non-Name/Attribute callee (e.g. ((expr))(x)) — Phase 1 doesn't
            # support; treat as undefined.
            self._raise_type_error(
                rule="undefined",
                expected="Name or Attribute callee",
                got=type(callee).__name__,
                node=expr,
            )
            return expr, _SERIES_FLOAT  # pragma: no cover

        # We have sig + qname; type-check args.
        assert sig is not None
        new_args = self._check_call_args(expr, sig, qname or "<call>")
        return dataclasses.replace(expr, args=new_args), sig.returns

    def _check_call_args(
        self, call: ir.CallExpr, sig: Signature, qname: str
    ) -> tuple[ir.KeywordArg, ...]:
        """Type-check call args against sig.args; apply PT001 promotion rule.

        Positional args bind to sig.args in order; keyword args bind by name.
        Excess args (e.g. `minval=1` to `input.int(20, minval=1)`) are
        accepted in the stub registry — full kwargs enumeration is a future
        S-bead concern. We just type-check the positionals that DO match a
        sig slot.
        """
        new_args: list[ir.KeywordArg] = []
        pos_idx = 0
        for arg in call.args:
            new_val, val_t = self._visit_expr(arg.value)
            new_args.append(dataclasses.replace(arg, value=new_val))
            if arg.name is None:
                # Positional — bind to next sig slot if any.
                if pos_idx < len(sig.args):
                    formal_name, formal_t = sig.args[pos_idx]
                    self._rule_pt001_simple_arg_cannot_be_series(
                        formal=formal_t,
                        actual=val_t,
                        formal_name=formal_name,
                        node=call,
                        qname=qname,
                    )
                pos_idx += 1
            else:
                # Keyword — bind by name when present.
                formal = next(
                    (t for (n, t) in sig.args if n == arg.name), None
                )
                # Fall through to Signature.kwargs when the name isn't among
                # positional-or-keyword args (D5 §7.2 keyword-only slots like
                # ``request.security(..., gaps=?, lookahead=?)``).
                if formal is None and sig.kwargs is not None:
                    formal = sig.kwargs.get(arg.name)
                if formal is not None:
                    self._rule_pt001_simple_arg_cannot_be_series(
                        formal=formal,
                        actual=val_t,
                        formal_name=arg.name,
                        node=call,
                        qname=qname,
                    )
        return tuple(new_args)

    # ------------------------------------------------------------------
    # request.security special-case (bead 0e9.6.y86 — D5 §4.1, §4.4, §7.2)
    # ------------------------------------------------------------------

    def _visit_request_security(
        self, expr: ir.CallExpr, qname: str
    ) -> tuple[ir.Expression, PineType]:
        """Type-check a ``request.security(...)`` call and register a
        :class:`SecurityContext`.

        Behaviour per D5 §4.1 + §4.4:

        * The ``symbol`` arg (positional 0 or kw ``symbol``) is inspected
          before it's walked so we can flag ``dynamic_symbol=True`` for
          non-literal forms (``syminfo.ticker``, ``input.symbol(...)``,
          any runtime-computed string). Static form is a bare ``StrLit``
          OR any expression that resolves to a compile-time-known string
          type (``input.string(...)`` returns ``input<string>``, one step
          up the const → input → simple → series lattice from
          ``const<string>``; both are known at compile time — see PR #322
          review comment ID 3522846703).
        * The ``timeframe`` arg (positional 1 or kw ``timeframe``) — same
          logic, feeds ``dynamic_timeframe``.
        * The ``expression`` arg (positional 2 or kw ``expression``) is
          visited normally so nested Pine calls (``ta.rsi(...)`` etc.)
          type-check; its serialized form (via :func:`_serialize_ir_expr`)
          becomes the :attr:`SecurityContext.expr` placeholder D2 reads
          opaquely.
        * Keyword-only ``gaps`` / ``lookahead`` are type-checked against
          the sig's kwargs map.

        Rejected forms (all raise :class:`PineTypeError`):

        * Missing ``symbol`` / ``timeframe`` / ``expression`` — Pine
          treats all three as required (PR #322 review comment ID
          3522847702).
        * Both positional AND keyword for the same slot (real Python
          raises ``TypeError``; we mirror the check — PR #322 review
          comment ID 3522847453).
        * ``na`` in the ``symbol`` / ``timeframe`` slot — ``na`` is a
          routing key with no runtime meaning here (PR #322 review
          comment ID 3522847918).

        Assigns a stable ``ctx_N`` id via a per-checker counter
        incremented as each ``request.security`` call is *finalised*.
        Because we recurse into ``expression`` (via ``_visit_expr``)
        BEFORE registering, a nested ``request.security(..., request.
        security(...))`` registers the INNER call first (``ctx_0``) and
        the OUTER call second (``ctx_1``). Not strictly source order —
        traversal (post-order) order (PR #322 review comment ID
        3522848074).
        """
        # Register in builtins_used before anything else so telemetry sees
        # the reference even if a later step raises.
        self._builtins_used.add(qname)

        # Split args into positional / kw. We resolve symbol / timeframe /
        # expression by their D5-canonical positions (0, 1, 2) with keyword
        # fall-through to match Pine's flexible call syntax.
        positional: list[ir.KeywordArg] = [a for a in expr.args if a.name is None]
        by_name: dict[str, ir.KeywordArg] = {
            a.name: a for a in expr.args if a.name is not None
        }

        _POS_INDEX = {"symbol": 0, "timeframe": 1, "expression": 2}

        def _slot(pos_idx: int, kw_name: str) -> ir.KeywordArg | None:
            """Resolve a named slot from either positional[pos_idx] or
            by_name[kw_name] — but raise when both are given, matching
            Python's ``TypeError: got multiple values for argument`` and
            avoiding a silent keyword-wins fallback (PR #322 review
            comment ID 3522847453)."""
            pos_hit = positional[pos_idx] if pos_idx < len(positional) else None
            kw_hit = by_name.get(kw_name)
            if pos_hit is not None and kw_hit is not None:
                self._raise_type_error(
                    rule="undefined",
                    expected=f"either positional or keyword for {kw_name!r}",
                    got=f"both positional[{pos_idx}] and {kw_name}=...",
                    node=expr,
                    hint=(
                        f"request.security got multiple values for "
                        f"argument {kw_name!r} (position {pos_idx} AND "
                        f"``{kw_name}=``). Pass one or the other."
                    ),
                )
            return kw_hit if kw_hit is not None else pos_hit

        symbol_arg = _slot(_POS_INDEX["symbol"], "symbol")
        timeframe_arg = _slot(_POS_INDEX["timeframe"], "timeframe")
        expression_arg = _slot(_POS_INDEX["expression"], "expression")

        # --- symbol -----------------------------------------------------
        symbol_str, dynamic_symbol, new_symbol_node = self._resolve_security_string_arg(
            symbol_arg, param="symbol", call=expr
        )

        # --- timeframe --------------------------------------------------
        timeframe_str, dynamic_timeframe, new_timeframe_node = self._resolve_security_string_arg(
            timeframe_arg, param="timeframe", call=expr
        )

        # --- expression -------------------------------------------------
        # Required per D5 §7.2 — same "raise on missing" treatment as
        # symbol / timeframe. Previously produced ``expr=""`` silently,
        # which would surface as a downstream KeyError once codegen wires
        # ``__security_contexts__`` into the emitted module (PR #322
        # review comment ID 3522847702).
        if expression_arg is None:
            self._raise_type_error(
                rule="undefined",
                expected="request.security expression argument",
                got="missing expression",
                node=expr,
                hint=(
                    "request.security requires an expression argument "
                    "at position 2 or as ``expression=``."
                ),
            )
        assert expression_arg is not None  # narrow for the type checker
        new_expression_node, _expr_t = self._visit_expr(expression_arg.value)
        expr_str = _serialize_ir_expr(new_expression_node)

        # --- gaps / lookahead kwargs -----------------------------------
        # Type-check the bool-const kwargs; walk them so any nested
        # references contribute to builtins_used. Any other kwargs Pine
        # scripts pass (e.g. ``calc_bars_count``) are still walked but
        # unenforced — matches the ``_check_call_args`` stub tolerance.
        new_kw_args: list[ir.KeywordArg] = []
        from pyne_compiler.compiler.builtin_signatures import BUILTIN_SIGNATURES

        sig = BUILTIN_SIGNATURES["request.security"]
        for name, arg in by_name.items():
            if name in {"symbol", "timeframe", "expression"}:
                # Already resolved above; skip re-walking.
                continue
            new_val, val_t = self._visit_expr(arg.value)
            new_kw_args.append(dataclasses.replace(arg, value=new_val))
            formal = (sig.kwargs or {}).get(name)
            if formal is not None:
                self._rule_pt001_simple_arg_cannot_be_series(
                    formal=formal,
                    actual=val_t,
                    formal_name=name,
                    node=expr,
                    qname=qname,
                )

        # --- register SecurityContext ----------------------------------
        context_id = f"ctx_{self._security_ctx_counter}"
        self._security_ctx_counter += 1
        self._security_contexts[context_id] = SecurityContext(
            symbol=symbol_str,
            timeframe=timeframe_str,
            expr=expr_str,
            dynamic_symbol=dynamic_symbol,
            dynamic_timeframe=dynamic_timeframe,
        )

        # --- rebuild the CallExpr with visited children ----------------
        # Preserve original positional / keyword ordering. Slot the
        # resolved nodes into the same positions they came from.
        rebuilt_args: list[ir.KeywordArg] = []
        pos_cursor = 0
        for original in expr.args:
            if original.name is None:
                # Positional at pos_cursor: substitute the resolved node
                # when it maps to symbol / timeframe / expression.
                if pos_cursor == 0 and new_symbol_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_symbol_node)
                    )
                elif pos_cursor == 1 and new_timeframe_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_timeframe_node)
                    )
                elif pos_cursor == 2 and new_expression_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_expression_node)
                    )
                else:
                    # Excess positional beyond the 3 we care about — visit
                    # normally to keep IR shape consistent.
                    new_val, _ = self._visit_expr(original.value)
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_val)
                    )
                pos_cursor += 1
            else:
                # Keyword: replace with the walked value for the tracked
                # names; otherwise substitute the visited-kwarg entry.
                if original.name == "symbol" and new_symbol_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_symbol_node)
                    )
                elif original.name == "timeframe" and new_timeframe_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_timeframe_node)
                    )
                elif original.name == "expression" and new_expression_node is not None:
                    rebuilt_args.append(
                        dataclasses.replace(original, value=new_expression_node)
                    )
                else:
                    match = next(
                        (a for a in new_kw_args if a.name == original.name),
                        None,
                    )
                    rebuilt_args.append(match if match is not None else original)

        return (
            dataclasses.replace(expr, args=tuple(rebuilt_args)),
            sig.returns,
        )

    def _resolve_security_string_arg(
        self,
        arg: ir.KeywordArg | None,
        *,
        param: str,
        call: ir.CallExpr,
    ) -> tuple[str, bool, ir.Expression | None]:
        """Resolve a ``request.security`` symbol / timeframe arg.

        Returns ``(serialized_str, is_dynamic, walked_node_or_None)``:

        * ``serialized_str`` — the literal string for a bare ``StrLit``,
          or the source-like rendering via :func:`_serialize_ir_expr` for
          dynamic forms (D5 §4.1 example ``"syminfo.ticker"``). Never the
          raw ``str(node)`` frozen-dataclass ``repr()`` (that produced
          ``"Name(loc=Span(...),id='close')"`` garbage on the dynamic
          path, which the runtime dispatcher would then hand to
          ``_fetch_via_fmp`` if the cache-round-trip lost the
          ``dynamic_*`` flag — see PR #322 review comment ID 3522846218).
        * ``is_dynamic`` — True iff the arg is NOT statically resolvable
          (D5 §4.4 fully-dynamic case). Statically resolvable per D5 §4.4
          + D1 §4.2 lattice covers ``StrLit`` and ANY expression whose
          resolved type is ``const<string>`` / ``input<string>`` /
          ``simple<string>`` — all are known at compile time (``input.*``
          values are baked in at load time even if surfaced through UI).
          Series-qualified strings are dynamic; PT001 rejects int/float.
        * ``walked_node_or_None`` — the visited IR node so the caller can
          slot it back into the rebuilt CallExpr; ``None`` when ``arg`` is
          missing (which is a compile-error: we raise before returning).

        Rejected forms (all raise :class:`PineTypeError`):

        * Missing arg — Pine requires all three positional args.
        * ``na`` — a routing key with no runtime meaning; per PR #322
          review comment ID 3522847918 the compiler rejects rather than
          silently forwarding ``NaLit`` to the dispatcher.
        * Non-string inner type (e.g. int, float) — PT001.

        For dynamic string args of the shape ``syminfo.<attr>`` or other
        ``<known_ns>.<attr>``, we deliberately swallow
        :class:`PineUnsupportedBuiltinError` from the generic
        :meth:`_visit_attribute` walk (the unsupported name is still
        recorded in ``_builtins_used`` before the exception fires). Once
        C3 registers those signatures via a later bead, the swallow
        becomes unnecessary but harmless.
        """
        if arg is None:
            self._raise_type_error(
                rule="undefined",
                expected=f"request.security {param} argument",
                got=f"missing {param}",
                node=call,
                hint=(
                    f"request.security requires a {param} argument at "
                    f"position "
                    f"{ {'symbol': 0, 'timeframe': 1}[param] } or as "
                    f"``{param}=``."
                ),
            )
            return "", False, None  # pragma: no cover — _raise_type_error raises

        value = arg.value

        # Reject `na` explicitly — a `na` symbol/timeframe has no runtime
        # meaning; PT006's general "na propagates any T" rule doesn't
        # apply here because these args are routing keys, not values (see
        # PR #322 review comment ID 3522847918).
        if isinstance(value, ir.NaLit):
            self._raise_type_error(
                rule="PT006",
                expected=f"non-na string for {param}",
                got="na",
                node=call,
                hint=(
                    f"request.security {param!r} must be a resolvable "
                    "string (literal or a const/input/simple/series-"
                    "qualified string expression); ``na`` is a routing "
                    "sentinel with no runtime meaning here."
                ),
            )
            return "", False, None  # pragma: no cover — _raise_type_error raises

        # Static-literal case: a bare Pine string literal like "SPY" or "1D".
        if isinstance(value, ir.StrLit):
            return value.value, False, value

        # Otherwise: walk the expression. We WANT this so nested references
        # land in builtins_used and Subscript.kind is resolved, but for
        # ``syminfo.<attr>`` / other unsupported-namespace attrs we must
        # swallow ``PineUnsupportedBuiltinError`` (the name is already
        # recorded in ``_builtins_used``).
        walked_node: ir.Expression = value
        walked_type: PineType | None = None
        try:
            walked_node, walked_type = self._visit_expr(value)
        except PineUnsupportedBuiltinError as exc:
            _ = exc  # silence lint; retained if future logging wants it

        # When we DID resolve a type, enforce it's a string. Pine rejects
        # ``request.security(123, "1D", close)`` per D5 §7.2. Skip the
        # check when walked_type is None (the arg raised
        # PineUnsupportedBuiltinError — dynamic-string by convention).
        if walked_type is not None and not isinstance(walked_type.inner, NaT):
            inner = walked_type.inner
            if not (isinstance(inner, Scalar) and inner.kind == "string"):
                self._raise_type_error(
                    rule="PT001",
                    expected="simple<string> | series<string>",
                    got=walked_type,
                    node=call,
                    hint=(
                        f"request.security {param!r} argument must be a "
                        "string (literal for static routing, or a "
                        "series/simple<string> for dynamic routing per "
                        "D5 §4.4). Got a non-string type."
                    ),
                )

        # Statically-resolvable-string test: D5 §4.4 + D1 §4.2 lattice
        # (``const → input → simple → series``). ``const<string>`` and
        # ``input<string>`` values are baked in at compile-eval time
        # (``input.string("1D")`` returns ``input<string>`` — its default
        # is known at compile time even though the runtime UI can override
        # per script-load). ``simple<string>`` is also compile-eval-known.
        # Only ``series<string>`` (or any non-string inner, already
        # rejected above) is truly dynamic. See PR #322 review comment ID
        # 3522846703 for the D5 §4.4 rationale — Pine users routinely
        # write ``tf = input.string("1D"); request.security("SPY", tf,
        # close)`` expecting the prefetch (not the 5-10× slower per-bar-
        # fetch) path.
        is_static = (
            walked_type is not None
            and walked_type.qualifier in ("const", "input", "simple")
            and isinstance(walked_type.inner, Scalar)
            and walked_type.inner.kind == "string"
        )
        return _serialize_ir_expr(walked_node), (not is_static), walked_node

    # ------------------------------------------------------------------
    # Rule handlers (PT001-PT008) — one per rule for findability
    # ------------------------------------------------------------------

    def _rule_pt001_simple_arg_cannot_be_series(
        self,
        *,
        formal: PineType,
        actual: PineType,
        formal_name: str,
        node: ir.Node,
        qname: str,
    ) -> None:
        """PT001 — actual must promote to formal's qualifier."""
        # na propagates: accept na where any T is expected.
        if isinstance(actual.inner, NaT):
            return
        if not can_promote(actual.qualifier, formal.qualifier):
            self._raise_type_error(
                rule="PT001",
                expected=formal,
                got=actual,
                node=node,
                hint=(
                    f"{qname}: param {formal_name!r} requires "
                    f"{formal.qualifier}<{self._inner_name(formal.inner)}> — "
                    f"{actual.qualifier}<{self._inner_name(actual.inner)}> "
                    "cannot demote (qualifier lattice runs const → input → "
                    "simple → series only)."
                ),
            )

    def _rule_pt003_bool_cond(
        self, cond_type: PineType, node: ir.Node
    ) -> None:
        """PT003 — `if`/ternary cond must be (series|simple|const)<bool>."""
        if isinstance(cond_type.inner, NaT):
            return  # na propagates; runtime na in cond is well-defined as false-ish.
        if not (isinstance(cond_type.inner, Scalar) and cond_type.inner.kind == "bool"):
            self._raise_type_error(
                rule="PT003",
                expected="series<bool>|simple<bool>|const<bool>",
                got=cond_type,
                node=node,
                hint=(
                    "Pine does NOT coerce int/float to bool. "
                    "Use an explicit comparison like `x > 0` or `x != 0`."
                ),
            )

    # ------------------------------------------------------------------
    # Type-level helpers
    # ------------------------------------------------------------------

    def _qualified_name(self, expr: ir.Expression) -> str | None:
        """If ``expr`` is an Attribute chain rooted in a Name, return the
        dotted qualified name; otherwise None."""
        parts: list[str] = []
        cur: ir.Expression = expr
        while isinstance(cur, ir.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ir.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    def _is_bool_or_na(self, t: PineType) -> bool:
        if isinstance(t.inner, NaT):
            return True
        return isinstance(t.inner, Scalar) and t.inner.kind == "bool"

    def _unify_inner(self, a: InnerType, b: InnerType, node: ir.Node) -> InnerType:
        """Inner-type unification with PT006 na-propagation."""
        if isinstance(a, NaT):
            return b
        if isinstance(b, NaT):
            return a
        if a == b:
            return a
        # Numeric widening: int + float -> float.
        if isinstance(a, Scalar) and isinstance(b, Scalar):
            if {a.kind, b.kind} == {"int", "float"}:
                return Scalar(kind="float")
        # Otherwise: incompatible.
        self._raise_type_error(
            rule="PT006",
            expected=f"unifiable with {self._inner_name(a)}",
            got=self._inner_name(b),
            node=node,
            hint="Mixed inner types — Pine allows int/float widening only.",
        )
        return a  # pragma: no cover

    @staticmethod
    def _inner_name(inner: InnerType) -> str:
        if isinstance(inner, Scalar):
            return inner.kind
        if isinstance(inner, Reference):
            return inner.kind
        if isinstance(inner, NaT):
            return "na"
        if isinstance(inner, UDT):
            return inner.name
        return type(inner).__name__


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check(
    program: ir.Program,
    *,
    pine_version: int,
    telemetry: "TelemetrySink | None" = None,
) -> TypeCheckResult:
    """Type-check ``program``, returning a :class:`TypeCheckResult`.

    Raises:
        PineTypeError: on any of the eight D1 §4.4 rule violations.
        PineUnsupportedBuiltinError: when the script references a Pine
            namespace member that isn't in :data:`BUILTIN_SIGNATURES`. The
            exception's ``.builtins_used`` attribute carries the partial
            coverage assembled before the failing call so callers can
            attribute the wild-corpus shortfall.
        PineUnsupportedFeatureError: when the script uses a Pine feature
            (e.g. typed-decl-in-body PF002) not yet shipped in Phase 1.

    ``telemetry`` (E0.4): optional :class:`TelemetrySink` — the C3
    unsupported-* raise paths call ``telemetry.record_*(name)`` on it
    BEFORE raising. ``None`` = no sink; the raise still fires but no
    counter increments. Threaded down to :class:`_TypeChecker`.
    """
    checker = _TypeChecker(pine_version=pine_version, telemetry=telemetry)
    new_prog = checker.check_program(program)
    return TypeCheckResult(
        program=new_prog,
        builtins_used=frozenset(checker._builtins_used),
        security_contexts=(
            dict(checker._security_contexts)
            if checker._security_contexts
            else None
        ),
        diagnostics=(),
    )
