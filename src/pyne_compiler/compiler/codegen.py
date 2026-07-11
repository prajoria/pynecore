"""C5 — Pine → Python codegen (bead 0e9.5.5).

Authoritative source: D1 §3 (codegen contract — read every subsection,
especially §3.1 CompiledModule, §3.2 allowlists, §3.3 the gate, §3.4 what
the gate is NOT, §3.5 telemetry); §2.5/§2.6 (IR invariants codegen relies
on). C6 owns the compile cache (:mod:`openbb_pine.compiler.compile_cache`);
this module's :func:`emit` returns a ``CompiledModule`` with an EMPTY
``sha`` and ``cache_status="bypass"``. The :func:`compile_pine` facade
overwrites both fields when the C6 cache is active.

Public surface
--------------

* :func:`emit` — ``(typed_program, builtins_used, security_contexts, …)``
  → :class:`CompiledModule`. Pure tree-walk Visitor over the typed IR; no
  analysis happens here (that's C3's job). Cache-agnostic — the returned
  module carries ``sha=""`` (empty) and ``cache_status="bypass"``.
* :data:`NODE_TYPE_ALLOWLIST`, :data:`MODULE_ALLOWLIST`,
  :data:`GLOBAL_NAME_ALLOWLIST` — the three frozensets that close the
  T1 mitigation per D1 §3.2.
* :func:`_enforce_allowlist` — the gate (D1 §3.3). Walks every ``ast.*``
  node and raises :class:`PineCodegenError(CG001/CG002/CG003)` on any
  violation. Runs **before** the source string is built, never after.

Two callers, two states
-----------------------

* Via :func:`compile_pine` facade (production path): facade overwrites
  ``sha`` with the C6 cache key and flips ``cache_status`` to
  ``"hit"`` / ``"miss"`` / ``"bypass"`` per the cache-probe outcome.
* Direct :func:`emit` call (tests + tooling that hand-construct an IR):
  ``sha=""`` + ``cache_status="bypass"`` — signalling "no cache
  key, no cache write". Direct callers who need a cache key can compute
  one via :func:`compile_cache.make_cache_key` and thread it through.

Output shape (D1 §3.1, verified against
``third_party/pynecore/tests/t01_lib/t30_strategy/test_002_bollinger.py``)::

    \"\"\"@pyne
    openbb-pine: compiled module
    script_sha = "…"; compiler_version = "0.1.0"; pine_version = 6
    \"\"\"
    from pynecore.lib import close, input, script, ta

    @script.indicator(title="BB", overlay=True)
    def main(length=input.int(20, minval=1), mult=input.float(2.0)):
        basis = ta.sma(close, length)
        dev = mult * ta.stdev(close, length)
        plot(basis, "basis")
        plot(basis + dev, "upper")
        plot(basis - dev, "lower")

* **First statement is the @pyne magic docstring** — PyneCore's
  ``import_script()`` (``script_runner.py:44``) keys off this exact
  pattern to fire its AST transformer.
* **One** ``from pynecore.lib import …`` line, consolidating every
  distinct namespace the script touched (drawn from
  ``compiled.builtins_used`` — the C3-populated set).
* **The function name is `main`** per PyneCore convention
  (``ScriptRunner.run_iter()`` expects this exact name).
* **Inputs become default-valued kwargs** — Pine's
  ``length = input.int(20, minval=1)`` is lifted out of the body into
  the ``def main`` signature so PyneCore's ``@script.indicator``
  decorator can inspect them.
* **plots are statement-position calls** — PyneCore's ``plot()``
  writes into ``lib._plot_data`` keyed by title; the executor
  (``runtime/_pynecore_glue.py``) snapshots that dict per-yield and
  the executor's ``_collect_results`` turns it into the DataFrame the
  ``/pine/run`` endpoint returns. Codegen therefore does NOT build a
  return dict — it just emits the plot() calls.

The closed allowlist (T1 mitigation, D1 §3.3)
---------------------------------------------

Defense-in-depth with D2's restricted-exec namespace (D1 §3.4):
even with a parser/typechecker bug producing IR that would translate
to something dangerous, the emitted Python cannot reference ``os``,
``sys``, ``subprocess``, ``socket``, ``open``, or ``__import__`` —
none are in :data:`MODULE_ALLOWLIST` or :data:`GLOBAL_NAME_ALLOWLIST`,
and ``ImportFrom`` / ``Name`` are codegen's only construction paths.

**CG001/CG002/CG003 are P0 incidents per D1 §3.5** — they should never
happen in production; one firing means a compiler bug. The structured
:class:`PineCodegenError` init (preempted in the prior commit) carries
``rule`` / ``node_kind`` / ``allowlist_member`` so the REST envelope
(D3 §4.1) surfaces the trace shape PRD §4.8 documents.

Out of scope
------------

* C6 compile cache is owned by :mod:`openbb_pine.compiler.compile_cache`;
  this module's ``emit`` sets ``sha=""`` + ``cache_status="bypass"`` and
  the :func:`compile_pine` facade wires the two together.
* C8 error-model wiring (separate bead — this only added
  PineCodegenError's structured init).
* The 36 stdlib bridges (S beads — each one ``.pine`` + ``.csv`` +
  Python impl).
* /pine/run end-to-end flip (still 503 until C5+C6+stdlib all land —
  a follow-up bead flips the route).
* ``strategy()`` decl support (M2 — raises PF010 if encountered).
* Type-args on calls (v6 ``array.new<float>(0)``) — Phase-1 ignores
  the type_args tuple. Codegen-level.
* ``request.security`` lowering — Phase 2; ``security_contexts`` is
  threaded through ``CompiledModule`` but Phase-1 codegen always
  passes ``None``.

Clean-room note: I have not viewed TradingView or PyneComp source
code. The output shape matches PyneCore's own published example file
(``test_002_bollinger.py`` — Apache-2.0) modulo the PyneComp credit
that we never emit and the cache-key metadata we add in the docstring.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from pyne_compiler.compiler import ir
from pyne_compiler.compiler.types import CompiledModule, SecurityContext
from pyne_compiler.errors.base import PineCodegenError, PineUnsupportedFeatureError

if TYPE_CHECKING:  # pragma: no cover — imports for typing only
    # Post-9bh: telemetry now lives at pyne_compiler.telemetry (extraction
    # complete). Kept behind TYPE_CHECKING so the compiler never actually
    # imports telemetry at runtime — the E0.4 injection contract (see
    # plan Task E0.4, Step 5 and design doc §6.E0.4).
    from pyne_compiler.telemetry import TelemetrySink

__all__ = [
    "NODE_TYPE_ALLOWLIST",
    "MODULE_ALLOWLIST",
    "GLOBAL_NAME_ALLOWLIST",
    "emit",
    "PYNE_HEADER_LINES",
]


# ---------------------------------------------------------------------------
# The closed allowlists (D1 §3.2)
# ---------------------------------------------------------------------------


# Every ``ast.*`` type the emitter is allowed to produce. Append-only across
# releases per D1 §3.2 — removing a type is a breaking change.
#
# Explicitly NOT in the set (the names that would let a script escape the
# language-level T1 sandbox if emitted):
#   ast.Import     — only ImportFrom (forces explicit ``from X import Y``)
#   ast.Lambda     — anonymous functions would let a script smuggle
#                    arbitrary code through builtins like map/filter
#   ast.ClassDef   — UDTs are Phase 2; we don't need it yet
#   ast.Global / ast.Nonlocal — no module-scope side effects from inside main
#   ast.Try / ast.TryStar / ast.ExceptHandler — Phase 2 (no Pine try/catch)
#   ast.With / ast.AsyncWith — Pine has no context managers
#   ast.Yield / ast.YieldFrom / ast.Await — no async/generator in Pine
#   ast.AsyncFunctionDef / ast.AsyncFor — no async/generator in Pine
#   ast.GeneratorExp / ListComp / SetComp / DictComp — Phase 2; codegen
#       doesn't emit comprehensions yet, so omitting these tightens the
#       gate without cost
#   ast.Delete / ast.Raise — codegen never emits these
#   ast.Starred — Pine has no *args spread (the lexer enforces this)
#   ast.NamedExpr — walrus operator; no analog in Pine
NODE_TYPE_ALLOWLIST: frozenset[type[ast.AST]] = frozenset({
    # --- Module structure ---
    ast.Module,
    ast.FunctionDef,
    ast.Return,
    ast.Pass,
    ast.arguments,
    ast.arg,
    ast.keyword,
    # --- Imports (restricted further to MODULE_ALLOWLIST below) ---
    ast.ImportFrom,
    ast.alias,
    # --- Assignments and naming ---
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Tuple,
    ast.List,
    # --- Expressions ---
    ast.Expr,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.IfExp,  # ternary
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.Call,
    ast.Dict,
    ast.JoinedStr,
    ast.FormattedValue,
    # --- Control flow ---
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
})


# Modules codegen may emit ``from X import Y`` for (D1 §3.2). Phase-1 codegen
# only emits from ``pynecore.lib`` directly — the 22 submodules listed in
# D1 §3.2 are accessed via attribute (``ta.sma``), not via explicit imports.
# The submodule entries are kept in the allowlist anyway so a future codegen
# refactor that emits ``from pynecore.lib.ta import sma`` (e.g. for cache
# locality) doesn't have to amend this set.
#
# Explicitly NOT in the allowlist (a partial list — anything not listed is
# rejected): os, sys, subprocess, socket, pathlib, builtins, importlib,
# json, pickle, marshal, ctypes, mmap, posix, nt, winreg, asyncio, threading,
# multiprocessing, http, urllib, requests, httpx, openbb_pine itself
# (no self-import), openbb_core, pandas, numpy, sklearn — none are reachable
# from user-emitted code.
MODULE_ALLOWLIST: frozenset[str] = frozenset({
    # Primary surface: every Phase-1 builtin lives under here.
    "pynecore.lib",
    # The 22 submodules per D1 §3.2 — kept to ease a possible future emit
    # that uses ``from pynecore.lib.ta import sma`` form.
    "pynecore.lib.ta",
    "pynecore.lib.math",
    "pynecore.lib.array",
    "pynecore.lib.matrix",
    "pynecore.lib.map",
    "pynecore.lib.string",
    "pynecore.lib.strategy",
    "pynecore.lib.request",
    "pynecore.lib.input",
    "pynecore.lib.color",
    "pynecore.lib.plot",
    "pynecore.lib.alert",
    "pynecore.lib.chart",
    "pynecore.lib.session",
    "pynecore.lib.syminfo",
    "pynecore.lib.barstate",
    "pynecore.lib.timeframe",
    "pynecore.lib.box",
    "pynecore.lib.line",
    "pynecore.lib.label",
    "pynecore.lib.table",
    "pynecore.lib.polyline",
    # Type-side imports (Series, Persistent, NA).
    "pynecore.types",
})


# Top-level names codegen may reference at module / global scope (D1 §3.2).
# Local function-scope names (def main args + assignments inside main) are
# tracked separately via :func:`_collect_user_defined_names` and exempted
# from this check.
#
# Three categories:
#   1. PyneCore-injected globals — OHLCV sources, bar metadata, plot family,
#      and the small set of value-bearing functions PyneCore exposes at
#      ``lib.__init__`` (verified against
#      ``third_party/pynecore/src/pynecore/lib/__init__.py:39-66``).
#   2. The Pine namespace modules (``ta``, ``math``, …) — accessed via
#      attribute, e.g. ``ta.sma``.
#   3. Python builtins available in the restricted exec namespace
#      (matches ``runtime/restricted._ALLOWED_BUILTINS``) — needed
#      because the emitted code may use ``min``, ``max``, ``len`` etc.
#      directly via Python-level operations.
GLOBAL_NAME_ALLOWLIST: frozenset[str] = frozenset({
    # === Category 1: PyneCore-injected values + script-emitted globals ===
    # OHLCV sources.
    "open", "high", "low", "close", "volume",
    "hl2", "hlc3", "ohlc4", "hlcc4",
    "bid", "ask",
    # Bar metadata.
    "bar_index", "last_bar_index", "last_bar_time", "max_bars_back",
    "time", "time_close", "time_tradingday", "timenow",
    "timestamp",
    # PyneCore exposes these as decorators / namespaces.
    "input", "script",
    # Plot family — top-level functions.
    "plot", "plotchar", "plotarrow", "plotbar", "plotcandle", "plotshape",
    "barcolor", "bgcolor", "fill", "linefill", "alertcondition", "alert",
    # NA helpers.
    "fixnan", "nz", "na",
    # Date / time helpers.
    "dayofmonth", "dayofweek", "hour", "minute", "month", "second",
    "weekofyear", "year",
    # hline (also a namespace) — used as a value.
    "hline",
    # === Category 2: Pine namespace modules ===
    "ta", "math", "array", "matrix", "map", "strategy", "request",
    "color", "alert", "chart", "session", "syminfo", "barstate",
    "timeframe", "box", "line", "label", "table", "polyline", "string",
    "currency", "earnings", "display", "extend", "location",
    "library", "log", "price", "runtime", "sym",
    # === Category 3: Python builtins available in the restricted namespace ===
    # (must match runtime/restricted._ALLOWED_BUILTINS exactly so the gate
    # and the exec namespace agree on the surface area)
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "int", "isinstance", "issubclass", "len",
    "list", "map", "max", "min", "object", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    # Python keywords PyCodegen emits as bare names through ast.Constant —
    # True / False / None are ast.Constant, not ast.Name in Python 3.8+, so
    # not listed here.
})


# The @pyne magic docstring lines. PyneCore's ``import_script()`` matches
# this prefix (``script_runner.py:44``); the executor's ``ensure_pyne_header``
# prepends the same shape defensively when a CompiledModule.source is hand-
# constructed in a test — we satisfy that check natively so line numbers in
# tracebacks match the emitted layout.
PYNE_HEADER_LINES: tuple[str, str] = ("@pyne", "openbb-pine: compiled module")


# Pine builtin namespace → which name must be imported from pynecore.lib.
# When the script references ``ta.sma`` we need ``ta`` available; when it
# references the bare ``close`` we need ``close``. The map keys mirror the
# qualified-name prefix and ``builtin_signatures.PINE_NAMESPACES``.
_NAMESPACE_IMPORTS: dict[str, str] = {
    "ta": "ta",
    "math": "math",
    "input": "input",
    "array": "array",
    "matrix": "matrix",
    "map": "map",
    "strategy": "strategy",
    "request": "request",
    "color": "color",
    "alert": "alert",
    "chart": "chart",
    "session": "session",
    "syminfo": "syminfo",
    "barstate": "barstate",
    "timeframe": "timeframe",
    "box": "box",
    "line": "line",
    "label": "label",
    "table": "table",
    "polyline": "polyline",
    "string": "string",
    "library": "library",
}


# Pine binary-op string → ast operator class.
_BINOP_MAP: dict[str, type[ast.operator] | type[ast.cmpop] | type[ast.boolop]] = {
    "+": ast.Add,
    "-": ast.Sub,
    "*": ast.Mult,
    "/": ast.Div,
    "%": ast.Mod,
    "==": ast.Eq,
    "!=": ast.NotEq,
    "<": ast.Lt,
    "<=": ast.LtE,
    ">": ast.Gt,
    ">=": ast.GtE,
    "and": ast.And,
    "or": ast.Or,
}

_UNARYOP_MAP: dict[str, type[ast.unaryop]] = {
    "+": ast.UAdd,
    "-": ast.USub,
    "not": ast.Not,
}


# ---------------------------------------------------------------------------
# Codegen visitor
# ---------------------------------------------------------------------------


class _CodegenVisitor:
    """Walk typed IR, build an ast.Module. Pure tree-walk — no analysis.

    All analysis is C3's job (type checker). Codegen assumes the §2.6
    invariants hold:

    1. Every Name resolves to scope or a known builtin (else C3 raised).
    2. Every CallExpr.func is Name or Attribute (else C3 raised).
    3. Every Subscript.kind is set (else C3 raised).
    4. Every IfStmt.cond / TernaryExpr.cond is bool (else C3 raised).
    5. var / varip have non-None value (else parser raised).

    The visitor accumulates input-lift assignments (``length = input.int(20)``)
    into ``_inputs`` rather than emitting them in the body, so they can be
    lifted into the ``def main`` signature as kwargs.
    """

    def __init__(
        self,
        builtins_used: frozenset[str],
        pine_version: int,
        *,
        telemetry: "TelemetrySink | None" = None,
    ) -> None:
        self.builtins_used = builtins_used
        self.pine_version = pine_version
        # Injected telemetry sink (E0.4). ``None`` = telemetry disabled;
        # the PF010/PF011 guards below short-circuit and no counter fires.
        # The :func:`emit` entry point threads this from
        # ``compile_pine(telemetry=...)``.
        self._telemetry = telemetry
        # (name, default_call_expr) pairs to be lifted into def main(...) kwargs.
        # Order preserved — Python kwargs are positional-by-default-value.
        self._inputs: list[tuple[str, ast.expr]] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def visit_Program(self, prog: ir.Program) -> ast.Module:
        # bd-aeh (D5 §7.3 C5): strategy() now emits @script.strategy(...),
        # mirroring the existing @script.indicator branch. library() remains
        # deferred (PF011).
        if prog.directive.kind == "library":
            if self._telemetry is not None:
                self._telemetry.record_unsupported_feature("PF011")
            raise PineUnsupportedFeatureError(
                "PF011 library decl — deferred (no Phase-1 use-case)",
                tracking_url=(
                    "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                    "?labels=pine-feature&q=PF011+library"
                ),
            )

        # Walk body, lifting input.* assignments out of it.
        body_stmts: list[ast.stmt] = []
        for s in prog.body:
            lifted = self._try_lift_input(s)
            if lifted is not None:
                self._inputs.append(lifted)
                continue
            body_stmts.append(self._visit_stmt(s))

        if not body_stmts:
            # Empty body — emit ``pass`` so the def is syntactically valid.
            body_stmts = [ast.Pass()]

        # Build the @script.indicator(...) decorator from the directive.
        decorator = self._build_script_decorator(prog.directive)

        # Build the def main(...) — input-lifted params become kwargs.
        main_args = self._build_main_args()

        main_fn = ast.FunctionDef(
            name="main",
            args=main_args,
            body=body_stmts,
            decorator_list=[decorator],
            returns=None,
            type_comment=None,
            type_params=[],
        )

        # Header docstring (PyneCore import-hook trigger) + import + def.
        module_body: list[ast.stmt] = [
            self._build_header_docstring(),
        ]
        import_stmt = self._build_import_statement()
        if import_stmt is not None:
            module_body.append(import_stmt)
        module_body.append(main_fn)

        module = ast.Module(body=module_body, type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    # ------------------------------------------------------------------
    # Header / imports / decorator
    # ------------------------------------------------------------------

    def _build_header_docstring(self) -> ast.Expr:
        """Emit the @pyne magic docstring as a module-level ``ast.Expr``."""
        text = (
            "@pyne\n"
            "openbb-pine: compiled module\n"
            f"compiler_version = \"<placeholder>\"; pine_version = {self.pine_version}"
        )
        return ast.Expr(value=ast.Constant(value=text))

    def _build_import_statement(self) -> ast.ImportFrom | None:
        """Emit a single ``from pynecore.lib import X, Y, …`` line.

        The set of names is derived from ``self.builtins_used``: each
        qualified name's first segment maps to either a Pine namespace
        (e.g. ``ta``) or a bare-value name (e.g. ``close``). We always
        import ``script`` (the indicator decorator needs it) and
        ``input`` (any input.* call needs it).

        Returns None when the script doesn't use any pynecore symbols
        (rare — every Pine script at least has ``script.indicator`` from
        the directive, so this branch is mostly defensive).
        """
        names: set[str] = {"script"}  # always needed for the decorator
        for qname in self.builtins_used:
            head = qname.split(".", 1)[0]
            if head in _NAMESPACE_IMPORTS:
                names.add(_NAMESPACE_IMPORTS[head])
            else:
                # Bare value name (close, open, high, low, volume, etc.).
                names.add(head)
        if not names:
            return None
        # Sort for deterministic output (cache-key stability).
        sorted_names = sorted(names)
        return ast.ImportFrom(
            module="pynecore.lib",
            names=[ast.alias(name=n, asname=None) for n in sorted_names],
            level=0,
        )

    def _build_script_decorator(self, directive: ir.ScriptDirective) -> ast.Call:
        """Build ``@script.indicator(title="…", overlay=…)``.

        Pine's ``indicator()`` call carries arbitrary keyword args (title,
        overlay, format, precision, max_bars_back, …). We pass through every
        scalar kwarg — non-scalar kwargs (rare) are emitted via the same
        expression-walker that handles call args elsewhere, which keeps
        the allowlist gate honest.
        """
        kwargs: list[ast.keyword] = []
        # title is positional in Pine but we surface it as a keyword for
        # PyneCore's @script.indicator decorator (which expects it as the
        # first positional or by name).
        pos_args: list[ast.expr] = []
        if directive.title:
            pos_args.append(ast.Constant(value=directive.title))
        # Re-emit explicit kwargs from the IR arguments, skipping the first
        # positional string (already used as title) and any anonymous title=.
        title_used_positionally = bool(directive.title)
        for kw in directive.arguments:
            if kw.name is None:
                # First positional string already consumed as title.
                if title_used_positionally and isinstance(kw.value, ir.StrLit):
                    title_used_positionally = False  # consume once
                    continue
                # Other positional args on indicator() are unusual — pass through.
                pos_args.append(self._visit_expr(kw.value))
            elif kw.name == "title":
                # Already consumed; skip the duplicate.
                continue
            else:
                kwargs.append(
                    ast.keyword(arg=kw.name, value=self._visit_expr(kw.value))
                )
        if directive.overlay is not None and not any(k.arg == "overlay" for k in kwargs):
            kwargs.append(
                ast.keyword(arg="overlay", value=ast.Constant(value=directive.overlay))
            )
        if directive.shorttitle is not None and not any(
            k.arg == "shorttitle" for k in kwargs
        ):
            kwargs.append(
                ast.keyword(
                    arg="shorttitle",
                    value=ast.Constant(value=directive.shorttitle),
                )
            )

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="script", ctx=ast.Load()),
                attr=directive.kind,  # "indicator" / "strategy" / "library"
                ctx=ast.Load(),
            ),
            args=pos_args,
            keywords=kwargs,
        )

    def _build_main_args(self) -> ast.arguments:
        """Lift the collected ``self._inputs`` into ``def main(…)`` kwargs."""
        args: list[ast.arg] = []
        defaults: list[ast.expr] = []
        for name, default in self._inputs:
            args.append(ast.arg(arg=name, annotation=None, type_comment=None))
            defaults.append(default)
        return ast.arguments(
            posonlyargs=[],
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=defaults,
        )

    # ------------------------------------------------------------------
    # Input lifting
    # ------------------------------------------------------------------

    def _try_lift_input(self, stmt: ir.Statement) -> tuple[str, ast.expr] | None:
        """If ``stmt`` is ``name = input.X(...)``, return (name, ast.Call).

        Otherwise return None — caller emits the statement normally.
        Pine's input declarations are statement-level (top-level
        ``length = input.int(20)``); we lift them so ``@script.indicator``'s
        introspection of ``main()``'s signature picks them up.
        """
        if not isinstance(stmt, ir.Assign):
            return None
        if stmt.op != "=":
            return None
        if not isinstance(stmt.target, ir.Name):
            return None
        if not isinstance(stmt.value, ir.CallExpr):
            return None
        if not isinstance(stmt.value.func, ir.Attribute):
            return None
        if not (
            isinstance(stmt.value.func.value, ir.Name)
            and stmt.value.func.value.id == "input"
        ):
            return None
        # It IS an input.* assignment — emit the call as the default.
        default_expr = self._visit_expr(stmt.value)
        return (stmt.target.id, default_expr)

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def _visit_stmt(self, stmt: ir.Statement) -> ast.stmt:
        if isinstance(stmt, ir.Assign):
            return self._visit_Assign(stmt)
        if isinstance(stmt, ir.VarDecl):
            return self._visit_VarDecl(stmt)
        if isinstance(stmt, ir.IfStmt):
            return self._visit_IfStmt(stmt)
        if isinstance(stmt, ir.ForStmt):
            return self._visit_ForStmt(stmt)
        if isinstance(stmt, ir.ForInStmt):
            return self._visit_ForInStmt(stmt)
        if isinstance(stmt, ir.WhileStmt):
            return self._visit_WhileStmt(stmt)
        if isinstance(stmt, ir.ReturnStmt):
            return self._visit_ReturnStmt(stmt)
        if isinstance(stmt, ir.ExprStmt):
            return self._visit_ExprStmt(stmt)
        if isinstance(stmt, ir.SwitchStmt):
            return self._visit_SwitchStmt(stmt)
        raise PineCodegenError(
            rule="CG004",
            node_kind=f"IR {type(stmt).__name__}",
            allowlist_member="any Phase-1 IR statement node",
            tracking_url=(
                "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                "?labels=pine-codegen&q=CG004"
            ),
        )

    def _visit_Assign(self, stmt: ir.Assign) -> ast.stmt:
        value = self._visit_expr(stmt.value)
        target = self._visit_target(stmt.target)
        # Pine's ``=`` and ``:=`` both lower to Python's ``=`` — PyneCore's
        # AST transformer turns local writes into Series-buffer updates,
        # so we don't need to fork on the operator here.
        return ast.Assign(targets=[target], value=value, type_comment=None)

    def _visit_VarDecl(self, stmt: ir.VarDecl) -> ast.stmt:
        # ``var x = e`` / ``varip x = e`` — PyneCore's @pyne magic handles
        # var lifecycle via Persistent[T] annotations injected by its
        # transformer. Phase-1 codegen just emits the plain assignment;
        # the persistence semantics live in PyneCore.
        # TODO(P3): emit ``x: Persistent[T] = e`` for var/varip so PyneCore's
        # transformer keys off the annotation rather than the @pyne docstring.
        # Phase-1 verifies the smoke-fixture path; persistence land in a
        # follow-up bead once the conformance corpus has a var-using fixture.
        value = self._visit_expr(stmt.value)
        target = ast.Name(id=stmt.name, ctx=ast.Store())
        return ast.Assign(targets=[target], value=value, type_comment=None)

    def _visit_IfStmt(self, stmt: ir.IfStmt) -> ast.stmt:
        test = self._visit_expr(stmt.cond)
        body = self._visit_block(stmt.then_body)
        # elif chain: nest each as the orelse of the previous.
        orelse: list[ast.stmt] = []
        if stmt.else_body is not None:
            orelse = self._visit_block(stmt.else_body)
        # Walk elif backwards to build the nested If chain.
        for ec, eb in reversed(stmt.elif_branches):
            etest = self._visit_expr(ec)
            ebody = self._visit_block(eb)
            elif_if = ast.If(test=etest, body=ebody, orelse=orelse)
            orelse = [elif_if]
        return ast.If(test=test, body=body, orelse=orelse)

    def _visit_ForStmt(self, stmt: ir.ForStmt) -> ast.stmt:
        # ``for v = a to b [by step]`` → ``for v in range(a, b+1, step)``.
        # Pine's ``to`` is inclusive; Python's ``range`` end is exclusive,
        # so we adjust by +1 when emitting the iterator. This is the simplest
        # correct lowering for ascending loops; PyneCore's transformer
        # accepts plain Python ``range`` so we don't need a custom iterator.
        start = self._visit_expr(stmt.start)
        end = self._visit_expr(stmt.end)
        # range(start, end + 1[, step])
        range_args: list[ast.expr] = [
            start,
            ast.BinOp(left=end, op=ast.Add(), right=ast.Constant(value=1)),
        ]
        if stmt.step is not None:
            range_args.append(self._visit_expr(stmt.step))
        iter_call = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=range_args,
            keywords=[],
        )
        return ast.For(
            target=ast.Name(id=stmt.var, ctx=ast.Store()),
            iter=iter_call,
            body=self._visit_block(stmt.body),
            orelse=[],
            type_comment=None,
        )

    def _visit_ForInStmt(self, stmt: ir.ForInStmt) -> ast.stmt:
        return ast.For(
            target=ast.Name(id=stmt.var, ctx=ast.Store()),
            iter=self._visit_expr(stmt.iterable),
            body=self._visit_block(stmt.body),
            orelse=[],
            type_comment=None,
        )

    def _visit_WhileStmt(self, stmt: ir.WhileStmt) -> ast.stmt:
        return ast.While(
            test=self._visit_expr(stmt.cond),
            body=self._visit_block(stmt.body),
            orelse=[],
        )

    def _visit_ReturnStmt(self, stmt: ir.ReturnStmt) -> ast.stmt:
        value = self._visit_expr(stmt.value) if stmt.value is not None else None
        return ast.Return(value=value)

    def _visit_ExprStmt(self, stmt: ir.ExprStmt) -> ast.stmt:
        # Carrier nodes for break / continue (parser convention).
        if isinstance(stmt.expr, ir.Name):
            if stmt.expr.id == "break":
                return ast.Break()
            if stmt.expr.id == "continue":
                return ast.Continue()
        return ast.Expr(value=self._visit_expr(stmt.expr))

    def _visit_SwitchStmt(self, stmt: ir.SwitchStmt) -> ast.stmt:
        # Phase-1 codegen: lower ``switch`` to a chain of if/elif so the
        # NODE_TYPE_ALLOWLIST stays tight. Python 3.10+ match/case is in
        # the allowlist of D1 §3.2 but adds ~6 ast types we don't otherwise
        # need; deferring keeps the gate's surface area minimal until
        # there's a conformance fixture that exercises switch.
        # TODO(P3): emit ast.Match once a switch-using fixture lands.
        if stmt.scrutinee is None:
            # Bare ``switch ... key => body`` lowers to a sequence of if's.
            return self._lower_switch_no_scrutinee(stmt)
        return self._lower_switch_with_scrutinee(stmt)

    def _lower_switch_no_scrutinee(self, stmt: ir.SwitchStmt) -> ast.stmt:
        """``switch ... cond => body, ... => default``."""
        default_body: list[ast.stmt] = []
        cases: list[tuple[ast.expr, list[ast.stmt]]] = []
        for ck, cb in stmt.cases:
            if ck is None:
                default_body = self._visit_block(cb)
            else:
                cases.append((self._visit_expr(ck), self._visit_block(cb)))
        if not cases and not default_body:
            return ast.Pass()
        orelse: list[ast.stmt] = default_body
        for test, body in reversed(cases):
            orelse = [ast.If(test=test, body=body, orelse=orelse)]
        return orelse[0] if orelse else ast.Pass()

    def _lower_switch_with_scrutinee(self, stmt: ir.SwitchStmt) -> ast.stmt:
        """``switch scrutinee   key => body   ... => default``.

        Lowered to ``if scrutinee == key: ...`` chain. The scrutinee may
        contain a side-effecting call; we don't hoist it into a temp
        because Phase-1 has no test that distinguishes the two
        semantics. TODO(P3) hoist into a fresh temp once a conformance
        fixture requires it.
        """
        assert stmt.scrutinee is not None
        scrut = self._visit_expr(stmt.scrutinee)
        default_body: list[ast.stmt] = []
        cases: list[tuple[ast.expr, list[ast.stmt]]] = []
        for ck, cb in stmt.cases:
            if ck is None:
                default_body = self._visit_block(cb)
            else:
                key = self._visit_expr(ck)
                test = ast.Compare(left=scrut, ops=[ast.Eq()], comparators=[key])
                cases.append((test, self._visit_block(cb)))
        if not cases and not default_body:
            return ast.Pass()
        orelse: list[ast.stmt] = default_body
        for test, body in reversed(cases):
            orelse = [ast.If(test=test, body=body, orelse=orelse)]
        return orelse[0] if orelse else ast.Pass()

    def _visit_block(self, stmts: tuple[ir.Statement, ...]) -> list[ast.stmt]:
        out = [self._visit_stmt(s) for s in stmts]
        return out or [ast.Pass()]

    def _visit_target(self, expr: ir.Expression) -> ast.expr:
        """Assign target — must produce a Store-context ast node."""
        if isinstance(expr, ir.Name):
            return ast.Name(id=expr.id, ctx=ast.Store())
        if isinstance(expr, ir.Attribute):
            return ast.Attribute(
                value=self._visit_expr(expr.value),
                attr=expr.attr,
                ctx=ast.Store(),
            )
        if isinstance(expr, ir.Subscript):
            return ast.Subscript(
                value=self._visit_expr(expr.value),
                slice=self._visit_expr(expr.index),
                ctx=ast.Store(),
            )
        if isinstance(expr, ir.TupleExpr):
            return ast.Tuple(
                elts=[self._visit_target(e) for e in expr.elements],
                ctx=ast.Store(),
            )
        # Catch-all: a non-target expression in target position is a parser
        # bug, surfaced through CG###.
        raise PineCodegenError(
            rule="CG005",
            node_kind=f"target IR {type(expr).__name__}",
            allowlist_member="Name | Attribute | Subscript | TupleExpr",
        )

    # ------------------------------------------------------------------
    # Expression dispatch
    # ------------------------------------------------------------------

    def _visit_expr(self, expr: ir.Expression) -> ast.expr:
        if isinstance(expr, ir.IntLit):
            return ast.Constant(value=expr.value)
        if isinstance(expr, ir.FloatLit):
            return ast.Constant(value=expr.value)
        if isinstance(expr, ir.StrLit):
            return ast.Constant(value=expr.value)
        if isinstance(expr, ir.BoolLit):
            return ast.Constant(value=expr.value)
        if isinstance(expr, ir.NaLit):
            # PyneCore exposes ``na`` as a top-level function/sentinel; we
            # emit the bare name and rely on the lib import for resolution.
            return ast.Name(id="na", ctx=ast.Load())
        if isinstance(expr, ir.ColorLit):
            # Color literal — PyneCore accepts the raw "#rrggbb" string and
            # also exposes color.red etc. via the color namespace. For a
            # raw hex literal we emit a string; symbolic names go through
            # Attribute (color.red) and arrive here as Attribute, not ColorLit.
            return ast.Constant(value=expr.raw)
        if isinstance(expr, ir.Name):
            return ast.Name(id=expr.id, ctx=ast.Load())
        if isinstance(expr, ir.Attribute):
            return ast.Attribute(
                value=self._visit_expr(expr.value),
                attr=expr.attr,
                ctx=ast.Load(),
            )
        if isinstance(expr, ir.Subscript):
            return self._visit_subscript(expr)
        if isinstance(expr, ir.BinaryExpr):
            return self._visit_binary(expr)
        if isinstance(expr, ir.UnaryExpr):
            return self._visit_unary(expr)
        if isinstance(expr, ir.TernaryExpr):
            return ast.IfExp(
                test=self._visit_expr(expr.cond),
                body=self._visit_expr(expr.then_),
                orelse=self._visit_expr(expr.else_),
            )
        if isinstance(expr, ir.CallExpr):
            return self._visit_call(expr)
        if isinstance(expr, ir.TupleExpr):
            return ast.Tuple(
                elts=[self._visit_expr(e) for e in expr.elements],
                ctx=ast.Load(),
            )
        raise PineCodegenError(
            rule="CG006",
            node_kind=f"IR expr {type(expr).__name__}",
            allowlist_member="any Phase-1 IR expression node",
        )

    def _visit_subscript(self, expr: ir.Subscript) -> ast.expr:
        # ``x[n]`` in Pine — history access for series, index access for
        # arrays. PyneCore's Series type overloads __getitem__ for history,
        # so the same Python expression handles both cases.
        return ast.Subscript(
            value=self._visit_expr(expr.value),
            slice=self._visit_expr(expr.index),
            ctx=ast.Load(),
        )

    def _visit_binary(self, expr: ir.BinaryExpr) -> ast.expr:
        op_cls = _BINOP_MAP[expr.op]
        lhs = self._visit_expr(expr.lhs)
        rhs = self._visit_expr(expr.rhs)
        if expr.op in ("and", "or"):
            return ast.BoolOp(op=op_cls(), values=[lhs, rhs])  # type: ignore[arg-type]
        if expr.op in ("==", "!=", "<", "<=", ">", ">="):
            return ast.Compare(left=lhs, ops=[op_cls()], comparators=[rhs])  # type: ignore[arg-type]
        return ast.BinOp(left=lhs, op=op_cls(), right=rhs)  # type: ignore[arg-type]

    def _visit_unary(self, expr: ir.UnaryExpr) -> ast.expr:
        return ast.UnaryOp(
            op=_UNARYOP_MAP[expr.op](),
            operand=self._visit_expr(expr.operand),
        )

    def _visit_call(self, expr: ir.CallExpr) -> ast.expr:
        func = self._visit_expr(expr.func)
        pos: list[ast.expr] = []
        kw: list[ast.keyword] = []
        for arg in expr.args:
            if arg.name is None:
                pos.append(self._visit_expr(arg.value))
            else:
                kw.append(ast.keyword(arg=arg.name, value=self._visit_expr(arg.value)))
        return ast.Call(func=func, args=pos, keywords=kw)


# ---------------------------------------------------------------------------
# The allowlist gate (D1 §3.3)
# ---------------------------------------------------------------------------


def _collect_user_defined_names(module: ast.Module) -> set[str]:
    """Names introduced by the user script — exempt from GLOBAL_NAME_ALLOWLIST.

    Phase-1 simple-pass implementation:

    * Function definitions at any level — their name.
    * Function parameters of every nested FunctionDef.
    * Assignment / AugAssign / AnnAssign LHS Name targets, anywhere.
    * For-loop iteration variables.
    * ImportFrom aliases — ``from X import Y`` introduces ``Y`` (or its
      ``asname`` if aliased) into scope. The MODULE_ALLOWLIST gate has
      already vetted that ``X`` is OK; the names that come out of it
      are legitimate references.

    This intentionally OVER-collects (a name introduced inside an inner
    function is treated as a global-allowed name everywhere) — false
    positives here can only LOOSEN the gate, but in Phase 1 the emitter
    is mechanical and produces no inner FunctionDefs other than ``main``,
    so the over-collection is harmless.

    What this does NOT track:

    * Comprehension scopes — codegen doesn't emit them in Phase 1.
    * Class scopes — codegen doesn't emit ast.ClassDef in Phase 1.
    * Lambda parameters — codegen doesn't emit ast.Lambda in Phase 1
      (and it's not in the allowlist anyway).
    * walrus operator — not in the allowlist.

    If a follow-up bead adds comprehensions / classes / lambdas the
    collector must grow accordingly.
    """
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
            args = node.args
            for arg in (
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
            ):
                names.add(arg.arg)
            if args.vararg is not None:
                names.add(args.vararg.arg)
            if args.kwarg is not None:
                names.add(args.kwarg.arg)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                _collect_target_names(tgt, names)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _collect_target_names(node.target, names)
        elif isinstance(node, (ast.For,)):
            _collect_target_names(node.target, names)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # ``from pynecore.lib import close, plot`` → {close, plot}.
                # ``from X import Y as Z`` → {Z}.
                names.add(alias.asname or alias.name)
    return names


def _collect_target_names(node: ast.expr, out: set[str]) -> None:
    """Recurse into a target ast expr, collecting bare Name ids."""
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            _collect_target_names(e, out)
    # ast.Subscript / ast.Attribute targets don't introduce a new name;
    # they mutate an existing one. No-op.


def _is_module_allowed(module_name: str | None) -> bool:
    """``ImportFrom.module`` allowlist check.

    A None module name (relative ``from . import x``) is rejected outright
    — codegen never emits relative imports.
    """
    if module_name is None:
        return False
    if module_name in MODULE_ALLOWLIST:
        return True
    # Allow submodules whose prefix is in the allowlist (e.g. a future
    # ``from pynecore.lib.ta.helpers import …`` if it ever lands).
    for prefix in MODULE_ALLOWLIST:
        if module_name.startswith(prefix + "."):
            return True
    return False


def _enforce_allowlist(module: ast.Module) -> None:
    """Walk the emitted module; raise PineCodegenError on any violation.

    Three rules per D1 §3.2 + §3.3:

    * **CG001** — any ``ast.*`` type outside NODE_TYPE_ALLOWLIST.
    * **CG002** — any ``ast.ImportFrom`` whose ``.module`` is outside
      MODULE_ALLOWLIST (or any ``ast.Import``, which is rejected as
      CG001 because Import isn't in NODE_TYPE_ALLOWLIST at all).
    * **CG003** — any ``ast.Name`` in ``Load`` context whose id is not
      in GLOBAL_NAME_ALLOWLIST and not in the user-defined-name set.

    Runs BEFORE the source string is built so a failing module is never
    handed to the cache or to PyneCore.
    """
    user_names = _collect_user_defined_names(module)

    for node in ast.walk(module):
        cls = type(node)
        if cls not in NODE_TYPE_ALLOWLIST:
            raise PineCodegenError(
                rule="CG001",
                node_kind=f"ast.{cls.__name__}",
                allowlist_member="any of NODE_TYPE_ALLOWLIST per D1 §3.2",
                tracking_url=(
                    "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                    "?labels=pine-codegen&q=CG001"
                ),
            )
        if isinstance(node, ast.ImportFrom):
            if not _is_module_allowed(node.module):
                raise PineCodegenError(
                    rule="CG002",
                    node_kind=f"ImportFrom({node.module!r})",
                    allowlist_member="any of MODULE_ALLOWLIST per D1 §3.2",
                    tracking_url=(
                        "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                        "?labels=pine-codegen&q=CG002"
                    ),
                )
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in GLOBAL_NAME_ALLOWLIST:
                continue
            if node.id in user_names:
                continue
            raise PineCodegenError(
                rule="CG003",
                node_kind=f"Name({node.id!r})",
                allowlist_member="any of GLOBAL_NAME_ALLOWLIST per D1 §3.2",
                tracking_url=(
                    "https://github.com/OpenBB-finance/OpenBBTerminal/issues/"
                    "?labels=pine-codegen&q=CG003"
                ),
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def emit(
    typed_program: ir.Program,
    *,
    builtins_used: frozenset[str],
    security_contexts: dict[str, SecurityContext] | None,
    pine_version: int,
    compiler_version: str,
    telemetry: "TelemetrySink | None" = None,
) -> CompiledModule:
    """Emit a :class:`CompiledModule` for ``typed_program`` (D1 §3.1).

    Pipeline:

    1. **Build the ast.Module** via :class:`_CodegenVisitor` (tree-walk).
    2. **Run the allowlist gate** (D1 §3.3) over the built module BEFORE
       any source string is produced. Any violation raises
       :class:`PineCodegenError` with structured ``rule`` / ``node_kind``.
    3. **Unparse to source** via :func:`ast.unparse` — produces the
       canonical formatted Python text.
    4. **Wrap in CompiledModule** with ``sha=""`` + ``cache_status="bypass"``
       — the :func:`compile_pine` facade overwrites both when the C6 cache
       is active. Direct callers of :func:`emit` who need a cache key can
       compute one via :func:`compile_cache.make_cache_key` and thread it
       through.

    Parameters
    ----------
    typed_program
        The C3 type-checker's output — IR with Subscript.kind resolved
        and every Name confirmed scope-resolved or builtin-resolved.
    builtins_used
        The set the C3 type-checker populated. Drives the
        ``from pynecore.lib import …`` line.
    security_contexts
        The C3 type-checker's lowered ``request.security`` map; None
        in Phase 1 (the type checker always returns None there).
    pine_version
        5 or 6 — surfaces in the @pyne docstring metadata.
    compiler_version
        ``openbb_pine.__version__`` — surfaces in the @pyne docstring
        and feeds into C6's cache key.
    telemetry
        Optional :class:`TelemetrySink` (E0.4). Threaded into the
        :class:`_CodegenVisitor` so PF010 (strategy) / PF011 (library)
        raises can record on the sink before raising. ``None`` = no sink;
        the raise still fires but no counter increments.

    Returns
    -------
    CompiledModule
        With ``source`` populated (validated by ``compile(...)``-ability
        in tests), ``sha=""`` (empty — C6's :func:`compile_pine` facade
        overwrites with the real cache key), ``cache_status="bypass"``
        (facade overwrites with ``"hit"`` / ``"miss"`` per the cache
        probe outcome), ``builtins_used`` echoed through, and
        ``security_contexts`` echoed through (None in Phase 1).

    Raises
    ------
    PineCodegenError
        Any CG### violation. Should never happen in production per
        D1 §3.5 — fire = compiler bug = P0.
    PineUnsupportedFeatureError
        ``strategy()`` (PF010) or ``library()`` (PF011) decl encountered;
        deferred to M2.
    """
    visitor = _CodegenVisitor(
        builtins_used=builtins_used,
        pine_version=pine_version,
        telemetry=telemetry,
    )
    module_ast = visitor.visit_Program(typed_program)

    # The gate — must fire BEFORE the source string is built.
    _enforce_allowlist(module_ast)

    source = ast.unparse(module_ast)

    # Patch in the real compiler_version into the header docstring. We
    # built the docstring with a ``<placeholder>`` token; replace it now
    # that the version is in hand. (Done at the text layer so the gate
    # sees the same string-Constant either way.)
    source = source.replace(
        'compiler_version = "<placeholder>"',
        f'compiler_version = "{compiler_version}"',
        1,
    )

    # sha="" + cache_status="bypass" is the "no cache" seam. The
    # :func:`compile_pine` facade wraps this call and overwrites both
    # fields when the C6 cache is active (with the real cache key + the
    # probe outcome). Direct callers of emit() opt into the "no cache"
    # contract by using these defaults.
    return CompiledModule(
        source=source,
        sha="",
        pine_version=pine_version,
        compiler_version=compiler_version,
        builtins_used=builtins_used,
        security_contexts=security_contexts,
        cache_status="bypass",
        script_type=typed_program.directive.kind,  # bd-aeh (D5 §5.1)
    )


# Module-load smoke check — fail fast if any allowlist set is unexpectedly
# empty (a programming error in this file). Cheap; runs once at import.
assert NODE_TYPE_ALLOWLIST, "NODE_TYPE_ALLOWLIST must be non-empty"
assert MODULE_ALLOWLIST, "MODULE_ALLOWLIST must be non-empty"
assert GLOBAL_NAME_ALLOWLIST, "GLOBAL_NAME_ALLOWLIST must be non-empty"
# Mirror-check against runtime/restricted: every name in _ALLOWED_BUILTINS
# must be in GLOBAL_NAME_ALLOWLIST (the gate and the exec namespace must
# agree on the Python-builtin surface area).
from pyne_compiler.runtime.restricted import _ALLOWED_BUILTINS  # noqa: E402

_runtime_only = _ALLOWED_BUILTINS - GLOBAL_NAME_ALLOWLIST
assert not _runtime_only, (
    "GLOBAL_NAME_ALLOWLIST is missing Python builtins that runtime/restricted "
    f"would allow at exec time: {sorted(_runtime_only)}. "
    "Either remove them from _ALLOWED_BUILTINS or add them here — the gate "
    "and the exec namespace must agree."
)
del _runtime_only
