"""D1 §6 compile cache — content-addressed under ~/.openbb/pine_cache/.

Cache key = blake2b(source ‖ params ‖ compiler_version ‖ pine_version).
Atomic write via tempfile.NamedTemporaryFile + os.replace (POSIX-safe,
NTFS-safe via MoveFileEx). No LRU / TTL in Phase 1 — manual purge only.
Defense-in-depth against T4 (cache poisoning) via the 256-bit BLAKE2b key
(birthday bound ~2^128; PRD §5.2 T4).

On-disk layout (D1 §6.2 — first-byte shard avoids the one-million-files-in-
one-dir ext4/NTFS pathology)::

    ~/.openbb/pine_cache/
        <sha[:2]>/<sha>.py         # emitted @pyne Python
        <sha[:2]>/<sha>.meta.json  # {pine_version, compiler_version,
                                    #  builtins_used, security_contexts,
                                    #  cached_at, source_len}

Two-file layout so ``cache_read`` can validate freshness cheaply (the
``.meta.json`` is ~200 bytes) before loading the Python source. Corrupt
cache handling: missing / bogus meta.json is logged as a warning and
treated as a miss — cache corruption must never break compilation, per
D1 §6.

Public surface
--------------

* :data:`DEFAULT_CACHE_DIR` — resolved to ``~/.openbb/pine_cache``.
* :func:`make_cache_key` — deterministic 256-bit BLAKE2b hash over the
  four inputs that make a compilation unique. Returns 64-char lowercase hex.
* :func:`cache_read` — return the :class:`CompiledModule` at ``sha`` with
  ``cache_status="hit"``, or ``None`` on miss / corruption.
* :func:`cache_write` — atomic write of the two-file entry. Idempotent
  (repeat writes with identical content are safe; content-changed writes
  overwrite atomically).
* :func:`cache_purge` — operator/CI tool. Removes cache entries; optional
  ``older_than_days`` filter.

Why the module-level ``DEFAULT_CACHE_DIR`` is a Path, not a factory:
tests monkeypatch it directly (see ``test_compile_cache.py::
TestCompilePineIntegration::test_default_cache_dir_is_used_when_none``).
Import is side-effect free — the directory is not mkdir'd at import time.

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from pyne_compiler.compiler.types import CompiledModule, SecurityContext
from pyne_compiler.errors.base import PineCacheError

__all__ = [
    "DEFAULT_CACHE_DIR",
    "make_cache_key",
    "cache_read",
    "cache_write",
    "cache_purge",
]


_log = logging.getLogger(__name__)


# Resolved at module-import time (as a Path — this MUST NOT mkdir the target;
# see D1 §6.2 + TestLazyCacheDir). Tests monkeypatch to a tmp_path.
DEFAULT_CACHE_DIR: Path = Path.home() / ".openbb" / "pine_cache"


# Required fields the cache_write side stamps into meta.json — cache_read
# validates all are present before reconstructing the CompiledModule. A
# missing field means the layout drifted (older on-disk format after a
# compiler upgrade); we degrade to miss.
_REQUIRED_META_FIELDS = frozenset({
    "pine_version",
    "compiler_version",
    "builtins_used",
    "security_contexts",
    "cached_at",
    "source_len",
})


# ---------------------------------------------------------------------------
# Cache key (D1 §6.1)
# ---------------------------------------------------------------------------


def make_cache_key(
    source: str,
    *,
    params: dict[str, Any] | None,
    compiler_version: str,
    pine_version: int,
) -> str:
    """Deterministic 256-bit BLAKE2b hash of the four inputs that make a
    compilation unique (D1 §6.1). Returns 64-char lowercase hex.

    The digest is over the concatenation
    ``f"{source}\\x00{params_json}\\x00{compiler_version}\\x00{pine_version}"``
    — the ``\\x00`` separators prevent boundary-crossing collisions (see
    ``TestMakeCacheKey::test_boundary_separator_prevents_concat_collision``).

    ``params`` is canonicalised via ``json.dumps(..., sort_keys=True)`` so
    dict-order does not influence the key. ``None`` params serialises as
    the string ``"null"``.

    256-bit digest → birthday bound ~2^128 collisions (cryptographically
    infeasible). Confirms PRD §5.2 T4 directly.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    h.update(b"\x00")
    h.update(compiler_version.encode("ascii"))
    h.update(b"\x00")
    h.update(str(pine_version).encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _shard_dir(cache_dir: Path, sha: str) -> Path:
    """First-byte shard (D1 §6.2) — spreads 256 ways to avoid the
    one-million-files-in-one-dir ext4/NTFS pathology."""
    return cache_dir / sha[:2]


def _paths_for(cache_dir: Path, sha: str) -> tuple[Path, Path]:
    """Return (py_path, meta_path) for a given cache key.

    D1 §6.2 says the layout is ``<sha[:2]>/<sha>/module.py + meta.json`` (a
    per-script directory). We flatten to ``<sha[:2]>/<sha>.py + .meta.json``:
    same shard fan-out, one fewer inode per entry, and the ``.py`` /
    ``.meta.json`` suffixes carry the same disambiguation the per-script
    directory offered. Documented divergence.
    """
    shard = _shard_dir(cache_dir, sha)
    return (shard / f"{sha}.py", shard / f"{sha}.meta.json")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def cache_read(sha: str, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> CompiledModule | None:
    """Return the :class:`CompiledModule` at ``sha`` with ``cache_status="hit"``,
    or ``None`` on miss (unknown sha) OR on corruption (missing / bogus
    meta.json, missing .py, missing required meta fields).

    Corruption is logged at WARNING and treated as a miss so a broken cache
    entry can never break compilation — the caller re-compiles and rewrites
    the entry.
    """
    py_path, meta_path = _paths_for(cache_dir, sha)
    if not meta_path.exists():
        # Neither a hit nor corrupt — just a miss. No log noise.
        if not py_path.exists():
            return None
        # .py exists without .meta.json → corrupt. Log + treat as miss.
        _log.warning(
            "compile cache: orphan .py without .meta.json at %s (treating as miss)",
            py_path,
        )
        return None
    if not py_path.exists():
        _log.warning(
            "compile cache: orphan .meta.json without .py at %s (treating as miss)",
            meta_path,
        )
        return None

    try:
        meta_text = meta_path.read_text(encoding="utf-8")
        meta = json.loads(meta_text)
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning(
            "compile cache: could not parse .meta.json at %s: %s (treating as miss)",
            meta_path,
            exc,
        )
        return None

    if not isinstance(meta, dict):
        _log.warning(
            "compile cache: .meta.json at %s is not a JSON object (treating as miss)",
            meta_path,
        )
        return None

    missing = _REQUIRED_META_FIELDS - set(meta)
    if missing:
        _log.warning(
            "compile cache: .meta.json at %s missing required fields %s "
            "(treating as miss — layout drift)",
            meta_path,
            sorted(missing),
        )
        return None

    try:
        source = py_path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning(
            "compile cache: could not read .py at %s: %s (treating as miss)",
            py_path,
            exc,
        )
        return None

    # Rehydrate security_contexts from JSON. Phase 1: always None (C3 doesn't
    # populate them yet); Phase 2 will need a stub-back-to-SecurityContext
    # helper. For now, if the JSON is None we're done; if it's a dict we
    # reconstruct SecurityContext instances (defensive — Phase 2 wire-up).
    sec_raw = meta["security_contexts"]
    sec: dict[str, SecurityContext] | None
    if sec_raw is None:
        sec = None
    elif isinstance(sec_raw, dict):
        try:
            sec = {
                k: SecurityContext(**v) for k, v in sec_raw.items()
            }
        except (TypeError, ValueError) as exc:
            _log.warning(
                "compile cache: could not rehydrate security_contexts at %s: %s "
                "(treating as miss)",
                meta_path,
                exc,
            )
            return None
    else:
        _log.warning(
            "compile cache: security_contexts at %s is neither null nor object "
            "(treating as miss)",
            meta_path,
        )
        return None

    return CompiledModule(
        source=source,
        sha=sha,
        pine_version=int(meta["pine_version"]),
        compiler_version=str(meta["compiler_version"]),
        builtins_used=frozenset(meta["builtins_used"]),
        security_contexts=sec,
        cache_status="hit",
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _make_tempfile_name(dir_path: Path, sha: str, ext: str) -> Path:
    """Build a tempfile path in ``dir_path``: ``<sha>.tmp-<pid>-<rand>.<ext>``.

    Placing the tempfile in the SAME directory as the target guarantees
    ``os.replace`` is an atomic rename (not a cross-device copy+delete).
    """
    rand = secrets.token_hex(6)  # 12 hex chars — plenty for uniqueness
    return dir_path / f"{sha}.tmp-{os.getpid()}-{rand}.{ext}"


def _write_atomic(target: Path, content: str, *, sha: str) -> None:
    """Write ``content`` to ``target`` atomically.

    Creates a tempfile in the same directory, writes+fsyncs, then
    ``os.replace`` — POSIX-atomic, NTFS-atomic (MoveFileEx). On failure
    the tempfile is cleaned up before propagating the exception.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lstrip(".") or "tmp"
    tmp = _make_tempfile_name(target.parent, sha, ext)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f"{sha}.tmp-{os.getpid()}-",
            suffix=f".{ext}",
            delete=False,
        ) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
            tmp = Path(fh.name)
        os.replace(str(tmp), str(target))
    except Exception:
        # Clean up any partial tempfiles the writer left behind BEFORE
        # propagating — the caller may retry and we must not leak.
        _cleanup_tempfiles(target.parent, sha)
        raise


def _cleanup_tempfiles(shard: Path, sha: str) -> None:
    """Remove any lingering ``<sha>.tmp-*`` files in ``shard``."""
    if not shard.exists():
        return
    prefix = f"{sha}.tmp-"
    for p in shard.iterdir():
        if p.name.startswith(prefix):
            try:
                p.unlink()
            except OSError:  # pragma: no cover - defensive
                pass


def cache_write(compiled: CompiledModule, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Atomic write of the two-file cache entry (D1 §6.3).

    Tempfile-in-same-dir + ``os.replace`` — POSIX-atomic, NTFS-atomic.
    Idempotent: if the target .py already exists with identical content,
    the write is a no-op (avoids gratuitous mtime updates that would
    confuse the ``older_than_days`` purge filter).

    Raises the underlying OSError (or a wrapping :class:`PineCacheError`
    when the shape merits it) on atomic-write failure. Cache corruption
    on the READ side never raises — it degrades to a miss. WRITE-side
    failures are surfaced because there's no fallback: if we can't write,
    the caller loses the cache-fill and we shouldn't hide that.
    """
    sha = compiled.sha
    py_path, meta_path = _paths_for(cache_dir, sha)

    # Serialise meta first (cheap, catches JSON-encoding bugs before we
    # touch the filesystem).
    meta = {
        "pine_version": compiled.pine_version,
        "compiler_version": compiled.compiler_version,
        "builtins_used": sorted(compiled.builtins_used),
        "security_contexts": (
            None
            if compiled.security_contexts is None
            else {
                # Round-trip EVERY SecurityContext field — the dynamic_*
                # flags are load-bearing for the runtime dispatcher
                # (D5 §4.4); dropping them silently turns a working
                # dynamic-symbol script into a stale-flag failure on the
                # second compile (see PR #322 review — comment ID
                # 3522845893). ``cache_read`` reconstructs via
                # ``SecurityContext(**v)`` so any new field added to the
                # dataclass must be added here too.
                k: {
                    "symbol": v.symbol,
                    "timeframe": v.timeframe,
                    "expr": v.expr,
                    "dynamic_symbol": v.dynamic_symbol,
                    "dynamic_timeframe": v.dynamic_timeframe,
                }
                for k, v in compiled.security_contexts.items()
            }
        ),
        "cached_at": time.time(),
        "source_len": len(compiled.source),
    }
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    # Idempotence: if both files exist with identical content, no-op.
    if py_path.exists() and meta_path.exists():
        try:
            existing_py = py_path.read_text(encoding="utf-8")
        except OSError:
            existing_py = None
        if existing_py == compiled.source:
            # .py content unchanged — same for meta modulo cached_at.
            # Rewriting would just churn mtime; skip.
            return

    _write_atomic(py_path, compiled.source, sha=sha)
    _write_atomic(meta_path, meta_json, sha=sha)


# ---------------------------------------------------------------------------
# Purge (operator / CI tool)
# ---------------------------------------------------------------------------


def cache_purge(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    older_than_days: int | None = None,
) -> int:
    """Manual purge (D1 §6.4 — no LRU / TTL in Phase 1). Returns count of
    removed entries.

    * ``older_than_days=None`` — removes EVERY entry. Test / CI use.
    * ``older_than_days=N`` — removes entries whose ``.py`` mtime is more
      than ``N`` days ago. Operator use — the pattern the D3 CLI
      ``openbb pine cache prune`` wraps.

    An "entry" is a ``.py`` + ``.meta.json`` pair — both get removed
    together. An orphan (only ``.py`` or only ``.meta.json``) counts as
    one entry and is also removed.

    Missing ``cache_dir`` is a no-op that returns 0 (matches operator
    expectation: "purge the empty cache").
    """
    if not cache_dir.exists():
        return 0

    threshold = (
        None if older_than_days is None else time.time() - older_than_days * 86400
    )

    removed = 0
    for shard in cache_dir.iterdir():
        if not shard.is_dir():
            continue
        # Group files by <sha>-stem so we count each entry once.
        stems: set[str] = set()
        for p in shard.iterdir():
            name = p.name
            if name.endswith(".meta.json"):
                stems.add(name[: -len(".meta.json")])
            elif name.endswith(".py"):
                stems.add(name[: -len(".py")])
            # Ignore tmp-* residue (see _cleanup_tempfiles).

        for stem in stems:
            py = shard / f"{stem}.py"
            meta = shard / f"{stem}.meta.json"

            if threshold is not None:
                # Use the .py mtime as the entry's age.
                try:
                    mtime = py.stat().st_mtime if py.exists() else meta.stat().st_mtime
                except OSError:
                    continue
                if mtime > threshold:
                    continue

            gone = False
            for target in (py, meta):
                try:
                    if target.exists():
                        target.unlink()
                        gone = True
                except OSError as exc:  # pragma: no cover - defensive
                    _log.warning(
                        "compile cache: could not remove %s during purge: %s",
                        target,
                        exc,
                    )
            if gone:
                removed += 1
    return removed
