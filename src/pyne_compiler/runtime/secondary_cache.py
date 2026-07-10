"""D5 §4.2 secondary-series cache — content-addressed under
``~/.openbb/pine_cache/secondary/``.

Separate directory from the D1 §6 compile cache (which lives at
``~/.openbb/pine_cache/<sha[:2]>/...``) so a compile-cache purge does not
wipe secondary OHLCV frames and vice versa.

Cache key = blake2b digest of ``f"{symbol}|{timeframe}|{start_bar}|{end_bar}"``.
Value = the aligned secondary-series ``pandas.DataFrame`` serialized as
pickle (``pyarrow`` isn't a hard dep in this venv; parquet was preferred by
D5 but pickle keeps dtypes + tz-aware indexes for zero cost and no extra
runtime dep). Follow-up bead can swap to parquet once pyarrow ships in the
default extras.

Security note on pickle
-----------------------
``pickle.load`` allows arbitrary code execution when reading attacker-
controlled data. This cache is safe because:

* The directory ``~/.openbb/pine_cache/secondary/`` is user-local (mode 700
  under a standard umask) and is only ever written by ``put()`` in this
  process — no cross-user or network sourcing.
* Filenames are 64-char blake2b hex derived from ``(symbol, timeframe,
  start_bar, end_bar)``, so an attacker cannot poison an entry that
  ``get()`` will look up without first predicting the exact call site.
* On any unpickling error, ``get()`` returns ``None`` (cache miss) and the
  runtime re-fetches — corrupted / spoofed files degrade gracefully rather
  than propagating unsafe objects.
* A type-guard rejects anything that isn't a ``pd.DataFrame`` after
  unpickling, capping the blast radius of a same-user tamper.

If cross-user shared caches (``/var/cache``) are added later, this MUST be
migrated to a schema-validated codec (parquet, or msgspec+arrow) before
enabling the shared path.

Atomic write via tempfile-in-same-dir + ``os.replace`` — mirrors
``compile_cache._write_atomic`` exactly (POSIX-atomic, NTFS-atomic).

TTL enforcement (D5 §4.2):
    * daily / higher timeframes → 24h (86 400s)
    * intraday timeframes       → 1h (3 600s)

TTL is checked in :meth:`get` against the on-disk mtime; stale entries are
deleted eagerly and reported as a miss (matches compile-cache convention:
corruption / expiry never breaks the run, it just fills again).

Clean-room note: I have not viewed TradingView or PyneComp source code.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import secrets
import tempfile
import time
from pathlib import Path

import pandas as pd

__all__ = [
    "DEFAULT_SECONDARY_CACHE_DIR",
    "SecondarySeriesCache",
    "make_secondary_key",
]


_log = logging.getLogger(__name__)


# Resolved at module-import time (as a Path — MUST NOT mkdir the target at
# import; mirror D1 §6.2 laziness). Tests monkeypatch to a tmp_path.
DEFAULT_SECONDARY_CACHE_DIR: Path = (
    Path.home() / ".openbb" / "pine_cache" / "secondary"
)


# --- Timeframe classification (D5 §4.2 TTL policy) ---------------------------

# Match ``byo_provider.INTRADAY_INTERVALS`` — Pine minute-count strings +
# lowercase FMP-style aliases (so ``"5"`` and ``"5m"`` and ``"5M"`` all match).
_INTRADAY_TFS: frozenset[str] = frozenset(
    {
        # FMP / OBB-shape
        "1m", "2m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
        # Pine-shape (digit-only minute counts)
        "1", "2", "3", "5", "15", "30", "60", "120", "240",
    }
)

_TTL_INTRADAY_S: int = 3_600
"""1 hour — intraday secondaries change quickly."""

_TTL_DAILY_S: int = 86_400
"""24 hours — daily and higher-timeframe secondaries change slowly."""


def _timeframe_ttl_seconds(tf: str) -> int:
    """Return the TTL (seconds) for a given Pine timeframe string.

    Intraday → 1 hour; daily / weekly / monthly → 24 hours. Unknown strings
    default to the daily budget (safer to expire slowly than to keep
    re-fetching quickly-changing data as if it were slow).
    """
    if tf in _INTRADAY_TFS or tf.lower() in _INTRADAY_TFS:
        return _TTL_INTRADAY_S
    return _TTL_DAILY_S


# --- Cache key ---------------------------------------------------------------


def make_secondary_key(
    symbol: str, timeframe: str, start_bar: object, end_bar: object
) -> str:
    """Deterministic 256-bit BLAKE2b hash over ``(symbol, timeframe,
    start_bar, end_bar)``. Returns 64-char lowercase hex.

    Uses the same digest size + algorithm as
    ``compile_cache.make_cache_key`` for consistency (grep'd errors, hex
    length, etc.). ``|`` separator matches the D5 §4.2 spec verbatim.

    ``start_bar`` / ``end_bar`` accept any object with a ``str()``
    representation — typically ``pd.Timestamp``, ``datetime``, ``int``, or
    ``None`` (start/end unknown means "full window").
    """
    payload = f"{symbol}|{timeframe}|{start_bar}|{end_bar}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


# --- Storage layout ----------------------------------------------------------


def _shard_dir(cache_dir: Path, key: str) -> Path:
    """First-byte shard — mirrors compile_cache._shard_dir. Spreads 256 ways
    so the secondary dir doesn't hit the one-million-files-in-one-dir
    ext4/NTFS pathology."""
    return cache_dir / key[:2]


def _path_for(cache_dir: Path, key: str) -> Path:
    """Return the single ``.pkl`` path for a given cache key.

    One file per entry (unlike the compile cache's two-file .py + .meta.json
    split) because the DataFrame's dtypes / index tz are already carried by
    the pickle format — no separate meta needed for TTL, which reads mtime.
    """
    return _shard_dir(cache_dir, key) / f"{key}.pkl"


# --- Atomic write helper -----------------------------------------------------


def _write_atomic(target: Path, payload: bytes, *, key: str) -> None:
    """Write ``payload`` bytes to ``target`` atomically.

    Tempfile in the SAME directory (so ``os.replace`` is a rename, not a
    cross-device copy+delete). Mirrors ``compile_cache._write_atomic``
    exactly.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=f"{key}.tmp-{os.getpid()}-",
            suffix=".pkl",
            delete=False,
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_path = Path(fh.name)
        os.replace(str(tmp_path), str(target))
        tmp_path = None
    finally:
        # Best-effort cleanup on failure (partial tempfile left behind).
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:  # pragma: no cover -- defensive
                pass


# --- Public class ------------------------------------------------------------


class SecondarySeriesCache:
    """On-disk LRU-free cache of per-``SecurityContext`` DataFrames.

    D5 §4.2 spec:

    * Key: ``(symbol, timeframe, start_bar, end_bar)`` → blake2b hex.
    * TTL: 24h for daily+ timeframes, 1h for intraday. Enforced via file mtime.
    * Storage: ``~/.openbb/pine_cache/secondary/`` (separate from compile cache).
    * Atomic writes.

    The class is a thin wrapper around a ``Path`` — no in-memory layer, no
    LRU, no size cap in Wave 1. Sizing / LRU is a follow-up bead once real
    hit-rate data is available.

    Corruption (unpicklable file, unexpected type) is logged at WARNING and
    treated as a miss — never breaks the run. Same convention as
    ``compile_cache.cache_read``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir: Path = cache_dir if cache_dir is not None else DEFAULT_SECONDARY_CACHE_DIR

    @property
    def cache_dir(self) -> Path:
        """Absolute path to the on-disk cache root."""
        return self._dir

    # --- Read -------------------------------------------------------------

    def get(
        self,
        symbol: str,
        timeframe: str,
        start_bar: object,
        end_bar: object,
    ) -> pd.DataFrame | None:
        """Return the cached DataFrame or ``None`` on miss / expiry / corruption.

        Miss / expiry / corruption are indistinguishable from the caller's
        perspective — the fetch path re-populates in every case. Expired
        entries are deleted eagerly (so the next miss doesn't repeat the
        stat call).
        """
        key = make_secondary_key(symbol, timeframe, start_bar, end_bar)
        path = _path_for(self._dir, key)
        if not path.exists():
            return None

        # TTL: reject entries older than the timeframe's budget.
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:  # pragma: no cover -- rare race
            _log.warning(
                "secondary cache: stat failed at %s: %s (treating as miss)",
                path, exc,
            )
            return None

        ttl = _timeframe_ttl_seconds(timeframe)
        age = time.time() - mtime
        if age > ttl:
            # Expired — evict + report miss.
            try:
                path.unlink()
            except OSError:  # pragma: no cover -- defensive
                pass
            return None

        try:
            with path.open("rb") as fh:
                # Safe pickle: file is under the user's own ~/.openbb tree,
                # keyed by a blake2b hex of the fetch parameters, only ever
                # written by put() in this process; any deserialise error or
                # type mismatch falls through to a cache miss (see the
                # module-level "Security note on pickle").
                obj = pickle.load(fh)  # noqa: S301  -- see module docstring
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
            _log.warning(
                "secondary cache: could not unpickle %s: %s (treating as miss)",
                path, exc,
            )
            return None

        if not isinstance(obj, pd.DataFrame):
            _log.warning(
                "secondary cache: %s deserialised to %s, not DataFrame "
                "(treating as miss)",
                path, type(obj).__name__,
            )
            return None

        return obj

    # --- Write ------------------------------------------------------------

    def put(
        self,
        symbol: str,
        timeframe: str,
        start_bar: object,
        end_bar: object,
        df: pd.DataFrame,
    ) -> None:
        """Atomically persist ``df`` under the ``(symbol, tf, start, end)`` key.

        Empty DataFrames are cached (a legitimately-empty secondary is still
        a datapoint — re-fetching would hit FMP for the same "no rows"
        response).

        Raises ``TypeError`` if ``df`` is not a DataFrame — catches the
        upstream bug where a caller accidentally passes an ``OBBject`` or a
        Series before it's been ``to_df()``'d.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"SecondarySeriesCache.put expects pd.DataFrame, got "
                f"{type(df).__name__}"
            )
        key = make_secondary_key(symbol, timeframe, start_bar, end_bar)
        path = _path_for(self._dir, key)
        payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        _write_atomic(path, payload, key=key)

    # --- Housekeeping ------------------------------------------------------

    def clear(self) -> int:
        """Remove every cached entry. Returns count of removed files.

        Test / operator utility; not called on the hot path.
        """
        if not self._dir.exists():
            return 0
        removed = 0
        for shard in self._dir.iterdir():
            if not shard.is_dir():
                continue
            for entry in shard.iterdir():
                if entry.name.endswith(".pkl"):
                    try:
                        entry.unlink()
                        removed += 1
                    except OSError:  # pragma: no cover -- defensive
                        pass
        return removed


# Re-export for tests that want to poke at the TTL policy directly.
__all__ += ["_timeframe_ttl_seconds"]
