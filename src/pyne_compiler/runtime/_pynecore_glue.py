"""Thin glue layer between R7 executor and the vendored PyneCore.

This module exists because D2 section 6 reads cleanly on paper but two pieces
of integration friction surface only when you actually drive PyneCore:

1. ``ScriptRunner.run_iter()`` yields ``(candle, lib._plot_data)`` BUT clears
   ``lib._plot_data`` after every yield (``script_runner.py:811`` / line 1085
   in the bar-magnifier path). The plot dict is **shared by reference** -- a
   naive ``list(runner.run_iter())`` captures the same (cleared) dict 5 times.
   The executor must copy the dict per-yield.

2. PyneCore's ``alert(message, ...)`` (``pynecore/lib/alert.py``) just prints
   the message; there is no built-in collection bucket. D2 section 6.1 requires
   ``OBBject.extra["alerts"]`` to be ``list[{bar_index, ts, message}]``. We
   monkey-patch ``pynecore.lib.alert.alert`` inside ``capture_alerts()`` for
   the duration of one ``run_iter()`` call so the print is replaced with an
   append to a local list.

3. ``ScriptRunner.__init__`` requires a real ``script_path: Path`` because its
   ``import_script()`` opens the file and re-imports the module through the
   ``@pyne`` AST hook. The executor cannot hand it an in-memory string; it
   must materialize the compiled source to a tempfile. The hook keys off the
   ``@pyne`` magic docstring in the file, not the path pattern, so any
   tempfile works as long as the docstring is present.

Keeping this glue out of ``executor.py`` makes the executor read like the D2
spec; the workarounds live here with the rationale next to them.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:  # pragma: no cover -- typing-only imports
    from pynecore.core.syminfo import SymInfo


# --- @pyne docstring helpers --------------------------------------------------

_PYNE_DOCSTRING_PREFIX = '"""\n@pyne\n"""\n'
"""The minimal magic-comment header that satisfies PyneCore's
``import_script()`` regex check (``script_runner.py:44``). Any module that
the executor hands to ``ScriptRunner`` must start with this exact docstring
shape (or one of its single-quote variants) for PyneCore's AST transformer
to fire on import."""


def ensure_pyne_header(source: str) -> str:
    """Return ``source`` with the ``@pyne`` magic docstring prepended if needed.

    PyneCore rejects modules that don't begin with a docstring containing
    ``@pyne`` (``script_runner.py:44``). The compiler (C5) is expected to emit
    this header naturally, but a manually-constructed ``CompiledModule.source``
    in tests may not -- so we pre-pend defensively here. Idempotent.
    """
    head = source.lstrip()[:256].lower()
    if "@pyne" in head:
        # Already present (typical C5 output) -- leave the source alone so
        # line numbers in tracebacks match the compiler's intended layout.
        return source
    return _PYNE_DOCSTRING_PREFIX + source


# --- Default SymInfo for BYO + smoke runs ------------------------------------


def make_default_syminfo(
    symbol: str,
    interval: str,
    *,
    asset_class: str = "equity",
    timezone_str: str = "UTC",
) -> "SymInfo":
    """Build a minimal-but-valid ``SymInfo`` for the primary OHLCV stream.

    Used when a caller does not provide their own ``SymInfo`` (which is the
    common path in M1 -- the rich metadata table is a Phase 2 nicety). The
    24/7 session layout matches CCXTProvider's ``_create_24_7_sessions`` so
    crypto / FX scripts behave correctly out of the box; equities behave as
    "always open" for the purposes of bar replay (TradingView treats missing
    bars the same way).

    The mapping ``asset_class -> pyne sym type`` mirrors the table at
    ``D2 §2.2`` but reduces to PyneCore's narrower ``Literal[...]`` set
    (``syminfo.py:46-50``): ``equity -> "stock"``, ``commodity -> "futures"``,
    ``currency -> "forex"``, ``crypto -> "crypto"``.
    """
    # Local imports so the sys.path bridge in openbb_pine/__init__.py has run.
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession  # noqa: PLC0415

    pyne_type_map: dict[str, str] = {
        "equity": "stock",
        "commodity": "futures",
        "currency": "forex",
        "crypto": "crypto",
    }
    pyne_type = pyne_type_map.get(asset_class, "stock")

    opening_hours: list[SymInfoInterval] = []
    session_starts: list[SymInfoSession] = []
    session_ends: list[SymInfoSession] = []
    for i in range(7):
        opening_hours.append(
            SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
        )
        session_starts.append(SymInfoSession(day=i, time=time(0, 0)))
        session_ends.append(SymInfoSession(day=i, time=time(23, 59, 59)))

    return SymInfo(
        prefix="BYO" if asset_class == "equity" else asset_class.upper(),
        description=symbol,
        ticker=symbol,
        currency="USD",
        basecurrency=None,
        period=_pine_period(interval),
        type=pyne_type,  # type: ignore[arg-type]
        mintick=0.01,
        pricescale=100,
        minmove=1,
        pointvalue=1.0,
        mincontract=1e-4 if asset_class == "crypto" else 1.0,
        opening_hours=opening_hours,
        session_starts=session_starts,
        session_ends=session_ends,
        timezone=timezone_str,
    )


def _pine_period(interval: str) -> str:
    """Translate an OBB / Pine interval string to PyneCore's ``period`` form.

    PyneCore's ``period`` is the TradingView spelling ("1D", "60", "5"). We
    accept the common OBB forms ("1d", "1h", "5m") and pass through the rest.
    """
    mapping = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240",
        "1d": "1D", "1D": "1D", "D": "1D",
        "1w": "1W", "W": "1W",
        "1M": "1M", "M": "1M",
    }
    return mapping.get(interval, interval)


# --- Alert-capture monkey-patch context -------------------------------------


@contextmanager
def capture_alerts(
    alert_sink: list[dict[str, Any]],
    *,
    bar_index_getter: Callable[[], int],
    timestamp_getter: Callable[[], int],
) -> Iterator[None]:
    """Monkey-patch ``pynecore.lib.alert.alert`` to append to ``alert_sink``.

    Per D2 section 6.1, every ``alert(message, ...)`` call during one
    ``run_iter()`` becomes an entry::

        {"bar_index": <int>, "ts": <ISO-8601 UTC str>, "message": <str>}

    The patch lives ONLY for the lifetime of the context manager; the
    original ``alert()`` is restored in ``finally``. ``alert_sink`` is the
    caller's list, mutated in place.

    PyneCore's ``alert()`` lives in ``pynecore.lib.alert.alert`` AND is
    re-exported into ``pynecore.lib`` at module load (see
    ``pynecore/lib/__init__.py``); the AST transformer rewrites bare
    ``alert(...)`` calls in user scripts into ``alert.alert(...)`` lookups
    against the module. We therefore patch the function on the ``alert``
    module so both the namespaced and bare invocations hit the shim.
    """
    # Local import keeps the sys.path bridge ordering intact.
    from pynecore.lib import alert as alert_module  # noqa: PLC0415

    original = alert_module.alert

    def _captured(message: Any, freq: Any = None) -> None:  # noqa: ARG001
        ts = timestamp_getter()
        bi = bar_index_getter()
        iso = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            if isinstance(ts, (int, float)) and ts > 0
            else ""
        )
        alert_sink.append(
            {"bar_index": int(bi), "ts": iso, "message": str(message)}
        )

    alert_module.alert = _captured  # type: ignore[assignment]
    try:
        yield
    finally:
        alert_module.alert = original  # type: ignore[assignment]


# --- AST-level T3 reinforcement scan -----------------------------------------

# Modules whose import from inside a compiled @pyne body is treated as a
# sandbox-escape attempt. The compiler's T1 allowlist forbids EMITTING these
# imports, but R7 is the runtime second line of defense per D2 §10.1.
#
# We do NOT use the restricted_namespace strategy directly for the user
# module exec because PyneCore's own machinery requires ``__import__`` to
# bootstrap its ``lib`` package (the AST transformer fires during
# ``import_script`` and pulls in dozens of pynecore submodules). Instead we
# scan the compiled source's AST for forbidden-module imports; if any are
# present, raise PineSecurityError BEFORE writing the tempfile.
_FORBIDDEN_TOP_LEVEL_MODULES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "ctypes", "multiprocessing", "threading", "asyncio",
    "http", "urllib", "ftplib", "telnetlib",
    "importlib",
    "builtins",
})


def scan_for_forbidden_imports(source: str) -> list[str]:
    """Return a list of forbidden imports in ``source`` (empty if clean).

    A defense-in-depth complement to R4's restricted-builtins namespace: when
    the compiled module has to be imported via PyneCore (so we can't strip
    ``__import__``), we scan its AST for any import that would have been
    blocked by D1's T1 allowlist. The scan is conservative -- it walks every
    ``Import`` / ``ImportFrom`` node and matches the *top-level* module name
    against ``_FORBIDDEN_TOP_LEVEL_MODULES``. Submodule names are matched
    too (``os.path`` triggers on ``os``).
    """
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Defer syntax errors to the actual import attempt -- that produces
        # a clearer traceback than re-raising from here.
        return []

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in _FORBIDDEN_TOP_LEVEL_MODULES:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".", 1)[0] if mod else ""
            if top in _FORBIDDEN_TOP_LEVEL_MODULES:
                offenders.append(mod)
    return offenders


__all__ = [
    "capture_alerts",
    "ensure_pyne_header",
    "make_default_syminfo",
    "scan_for_forbidden_imports",
]
