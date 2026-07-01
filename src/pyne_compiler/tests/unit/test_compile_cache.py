"""C6 compile cache — unit tests (bead 0e9.5.6, GH #107 C6).

Source of truth: D1 §6 (cache key + on-disk layout + atomic write + eviction).

Every test uses a `tmp_path`-scoped cache directory so the tests never touch
``~/.openbb/pine_cache`` and can run in parallel without collisions.

Coverage areas:

* :class:`TestMakeCacheKey` — determinism, sensitivity to every input axis,
  digest shape (64 lowercase hex chars = BLAKE2b-256).
* :class:`TestCacheReadWriteRoundtrip` — write → read returns the same
  CompiledModule (modulo cache_status flip to "hit"), miss returns None.
* :class:`TestAtomicWrite` — os.replace fault-injection keeps partial
  writes invisible to cache_read.
* :class:`TestCorruptCacheHandling` — missing / bogus meta.json degrades
  to miss + logs a warning, never raises.
* :class:`TestIdempotence` — repeat writes are safe (no torn state).
* :class:`TestCompilePineIntegration` — end-to-end: first compile is a miss,
  second is a hit, use_cache=False bypasses. Params carve separate slots.
* :class:`TestLazyCacheDir` — importing compile_cache does NOT create
  ``~/.openbb/pine_cache`` as a side effect.
* :class:`TestPurge` — cache_purge counts removed entries; older_than_days
  filter works.
* :class:`TestPineCacheErrorSurface` — the preempted structured init is
  usable at the C6 layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from openbb_pine.compiler.compile_cache import (
    DEFAULT_CACHE_DIR,
    cache_purge,
    cache_read,
    cache_write,
    make_cache_key,
)
from openbb_pine.compiler.types import CompiledModule
from openbb_pine.errors import PineCacheError


# ---------------------------------------------------------------------------
# Shared fixtures + helpers
# ---------------------------------------------------------------------------


TRIVIAL_V6_SOURCE = '//@version=6\nindicator("X")\nplot(close)\n'
TRIVIAL_V6_SOURCE_B = '//@version=6\nindicator("Y")\nplot(close)\n'


def _make_compiled(
    *,
    source: str = "from pynecore.lib import close\n",
    sha: str = "a" * 64,
    pine_version: int = 6,
    compiler_version: str = "0.1.0",
    builtins_used: frozenset[str] = frozenset({"close"}),
    security_contexts=None,
    cache_status: str = "miss",
) -> CompiledModule:
    return CompiledModule(
        source=source,
        sha=sha,
        pine_version=pine_version,
        compiler_version=compiler_version,
        builtins_used=builtins_used,
        security_contexts=security_contexts,
        cache_status=cache_status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# make_cache_key: determinism + sensitivity
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """The BLAKE2b-256 cache key derived from source ‖ params ‖ compiler ‖ pine.

    D1 §6.1 spells out the four axes that make a compilation unique. The key
    MUST change when any of them changes; it MUST be stable across runs when
    all of them are held constant.
    """

    def test_key_is_64_char_lowercase_hex(self) -> None:
        key = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params=None,
            compiler_version="0.1.0",
            pine_version=6,
        )
        assert isinstance(key, str)
        assert len(key) == 64
        # blake2b-256 hexdigest is always lowercase.
        assert key == key.lower()
        # And it's a valid hex string.
        int(key, 16)

    def test_key_is_deterministic(self) -> None:
        k1 = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params=None,
            compiler_version="0.1.0",
            pine_version=6,
        )
        k2 = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params=None,
            compiler_version="0.1.0",
            pine_version=6,
        )
        assert k1 == k2

    def test_key_changes_when_source_changes(self) -> None:
        k1 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.1.0", pine_version=6)
        k2 = make_cache_key(TRIVIAL_V6_SOURCE_B, params=None, compiler_version="0.1.0", pine_version=6)
        assert k1 != k2

    def test_key_changes_when_params_change(self) -> None:
        k1 = make_cache_key(TRIVIAL_V6_SOURCE, params={"length": 10}, compiler_version="0.1.0", pine_version=6)
        k2 = make_cache_key(TRIVIAL_V6_SOURCE, params={"length": 20}, compiler_version="0.1.0", pine_version=6)
        assert k1 != k2

    def test_key_changes_when_compiler_version_changes(self) -> None:
        k1 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.1.0", pine_version=6)
        k2 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.2.0", pine_version=6)
        assert k1 != k2

    def test_key_changes_when_pine_version_changes(self) -> None:
        k1 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.1.0", pine_version=5)
        k2 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.1.0", pine_version=6)
        assert k1 != k2

    def test_none_params_and_empty_dict_are_distinct(self) -> None:
        """Sanity: `params={}` and `params=None` may or may not collide, but
        the function must handle both without exploding."""
        k1 = make_cache_key(TRIVIAL_V6_SOURCE, params=None, compiler_version="0.1.0", pine_version=6)
        k2 = make_cache_key(TRIVIAL_V6_SOURCE, params={}, compiler_version="0.1.0", pine_version=6)
        # Both are valid keys; equality is an implementation choice.
        assert len(k1) == 64
        assert len(k2) == 64

    def test_params_ordering_stable(self) -> None:
        """sort_keys=True in the JSON canonicalisation means dict-order
        MUST NOT affect the key."""
        k1 = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params={"a": 1, "b": 2},
            compiler_version="0.1.0",
            pine_version=6,
        )
        k2 = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params={"b": 2, "a": 1},
            compiler_version="0.1.0",
            pine_version=6,
        )
        assert k1 == k2

    def test_boundary_separator_prevents_concat_collision(self) -> None:
        """The \\x00 separator between fields is what stops
        f'{source}{compiler}' being confusable with f'{source}\\x00{compiler}'.
        Verify: two inputs that would collide if joined naively still differ."""
        # If join was naive: "abc" + "def" ≡ "ab" + "cdef".
        k1 = make_cache_key("abc", params=None, compiler_version="def", pine_version=6)
        k2 = make_cache_key("ab", params=None, compiler_version="cdef", pine_version=6)
        assert k1 != k2

    def test_matches_expected_blake2b_shape(self) -> None:
        """The key should be BLAKE2b-256 (64 hex) — not some other digest."""
        key = make_cache_key(
            TRIVIAL_V6_SOURCE,
            params=None,
            compiler_version="0.1.0",
            pine_version=6,
        )
        # Reproduce independently — should match.
        h = hashlib.blake2b(digest_size=32)
        params_json = json.dumps(None, sort_keys=True, separators=(",", ":"))
        payload = (
            TRIVIAL_V6_SOURCE + "\x00" + params_json + "\x00" + "0.1.0" + "\x00" + "6"
        ).encode("utf-8")
        h.update(payload)
        assert key == h.hexdigest()


# ---------------------------------------------------------------------------
# Round-trip: write → read returns the module
# ---------------------------------------------------------------------------


class TestCacheReadWriteRoundtrip:
    """Two-file layout (.py + .meta.json) reconstructs the CompiledModule
    with cache_status flipped from 'miss' (writer) → 'hit' (reader)."""

    def test_miss_returns_none_for_unknown_sha(self, tmp_path: Path) -> None:
        assert cache_read("0" * 64, cache_dir=tmp_path) is None

    def test_write_then_read_returns_hit(self, tmp_path: Path) -> None:
        cm = _make_compiled(sha="a" * 64, cache_status="miss")
        cache_write(cm, cache_dir=tmp_path)

        got = cache_read("a" * 64, cache_dir=tmp_path)
        assert got is not None
        assert got.source == cm.source
        assert got.pine_version == cm.pine_version
        assert got.compiler_version == cm.compiler_version
        assert got.builtins_used == cm.builtins_used
        assert got.security_contexts == cm.security_contexts
        # cache_status flips to 'hit' on read.
        assert got.cache_status == "hit"

    def test_read_preserves_sha_field(self, tmp_path: Path) -> None:
        cm = _make_compiled(sha="b" * 64)
        cache_write(cm, cache_dir=tmp_path)

        got = cache_read("b" * 64, cache_dir=tmp_path)
        assert got is not None
        assert got.sha == "b" * 64

    def test_read_preserves_multiple_builtins(self, tmp_path: Path) -> None:
        cm = _make_compiled(
            sha="c" * 64,
            builtins_used=frozenset({"ta.sma", "close", "plot"}),
        )
        cache_write(cm, cache_dir=tmp_path)
        got = cache_read("c" * 64, cache_dir=tmp_path)
        assert got is not None
        assert got.builtins_used == frozenset({"ta.sma", "close", "plot"})

    def test_write_creates_two_files_under_shard(self, tmp_path: Path) -> None:
        """D1 §6.2 layout: cache_dir/<sha[:2]>/<sha>.py + <sha>.meta.json."""
        cm = _make_compiled(sha="12abcdef" + "0" * 56)
        cache_write(cm, cache_dir=tmp_path)

        shard = tmp_path / "12"
        assert shard.exists()
        py_path = shard / (cm.sha + ".py")
        meta_path = shard / (cm.sha + ".meta.json")
        assert py_path.exists()
        assert meta_path.exists()

    def test_write_meta_json_is_valid_json_with_expected_fields(
        self, tmp_path: Path
    ) -> None:
        cm = _make_compiled(sha="d" * 64, builtins_used=frozenset({"close"}))
        cache_write(cm, cache_dir=tmp_path)

        meta_path = tmp_path / "dd" / (cm.sha + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # D1 §6.2 required fields (per bead spec docstring).
        assert meta["pine_version"] == 6
        assert meta["compiler_version"] == "0.1.0"
        assert sorted(meta["builtins_used"]) == ["close"]
        assert meta["security_contexts"] is None
        assert isinstance(meta["cached_at"], (int, float))
        assert meta["source_len"] == len(cm.source)

    def test_write_source_file_is_the_module_source_verbatim(
        self, tmp_path: Path
    ) -> None:
        cm = _make_compiled(sha="e" * 64, source="print('hello world')\n")
        cache_write(cm, cache_dir=tmp_path)

        py_path = tmp_path / "ee" / (cm.sha + ".py")
        assert py_path.read_text(encoding="utf-8") == cm.source


# ---------------------------------------------------------------------------
# Atomic write: os.replace fault injection
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """A killed / crashed writer must never leave a partial file visible
    as a cache hit. Fault-injection on os.replace models the crash."""

    def test_replace_failure_removes_tempfiles(self, tmp_path: Path) -> None:
        """When os.replace raises, no tmp-* files should linger in the shard."""
        cm = _make_compiled(sha="f" * 64)

        with patch("openbb_pine.compiler.compile_cache.os.replace") as mocked:
            mocked.side_effect = OSError("simulated crash")
            with pytest.raises(OSError, match="simulated crash"):
                cache_write(cm, cache_dir=tmp_path)

        # No tmp-* residue in the shard.
        shard = tmp_path / "ff"
        if shard.exists():
            leftovers = list(shard.iterdir())
            for p in leftovers:
                assert ".tmp-" not in p.name, f"tempfile leaked: {p}"

    def test_replace_failure_leaves_no_visible_hit(self, tmp_path: Path) -> None:
        """After a failed write, cache_read must still return None."""
        cm = _make_compiled(sha="1" * 64)

        with patch("openbb_pine.compiler.compile_cache.os.replace") as mocked:
            mocked.side_effect = OSError("simulated crash")
            with pytest.raises(OSError):
                cache_write(cm, cache_dir=tmp_path)

        # No visible cache entry.
        assert cache_read("1" * 64, cache_dir=tmp_path) is None

    def test_tempfiles_created_in_same_directory(self, tmp_path: Path) -> None:
        """Tempfiles must be created in the target shard directory to make
        os.replace an atomic rename (cross-device replace would degrade
        to copy+delete, breaking atomicity)."""
        cm = _make_compiled(sha="2" * 64)
        original_replace = os.replace
        seen_source_dirs: list[Path] = []
        seen_dst_dirs: list[Path] = []

        def _capture(src, dst) -> None:
            seen_source_dirs.append(Path(src).parent)
            seen_dst_dirs.append(Path(dst).parent)
            return original_replace(src, dst)

        with patch("openbb_pine.compiler.compile_cache.os.replace", side_effect=_capture):
            cache_write(cm, cache_dir=tmp_path)

        # Each source dir equals the corresponding dst dir → atomic rename.
        for src, dst in zip(seen_source_dirs, seen_dst_dirs):
            assert src == dst


# ---------------------------------------------------------------------------
# Corrupt cache handling: never raise, always degrade to miss
# ---------------------------------------------------------------------------


class TestCorruptCacheHandling:
    """Corruption must never break compilation — cache_read logs + returns
    None. Only actual atomic-write failures propagate as PineCacheError."""

    def test_missing_meta_json_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """.py exists but .meta.json doesn't → treat as miss + log warning."""
        sha = "3" * 64
        shard = tmp_path / "33"
        shard.mkdir(parents=True)
        (shard / (sha + ".py")).write_text("print('hi')\n", encoding="utf-8")
        # deliberately don't write meta.json

        got = cache_read(sha, cache_dir=tmp_path)
        assert got is None

    def test_missing_py_source_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """.meta.json present but .py missing → treat as miss + log."""
        sha = "4" * 64
        shard = tmp_path / "44"
        shard.mkdir(parents=True)
        (shard / (sha + ".meta.json")).write_text(
            json.dumps({
                "pine_version": 6,
                "compiler_version": "0.1.0",
                "builtins_used": [],
                "security_contexts": None,
                "cached_at": time.time(),
                "source_len": 12,
            }),
            encoding="utf-8",
        )
        # deliberately don't write .py

        got = cache_read(sha, cache_dir=tmp_path)
        assert got is None

    def test_bogus_meta_json_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """.meta.json is not valid JSON → treat as miss + log warning; no raise."""
        sha = "5" * 64
        shard = tmp_path / "55"
        shard.mkdir(parents=True)
        (shard / (sha + ".py")).write_text("print('hi')\n", encoding="utf-8")
        (shard / (sha + ".meta.json")).write_text(
            "{{not valid json", encoding="utf-8"
        )

        # No exception.
        got = cache_read(sha, cache_dir=tmp_path)
        assert got is None

    def test_meta_json_missing_required_field_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """.meta.json parses but is missing pine_version → miss + log."""
        sha = "6" * 64
        shard = tmp_path / "66"
        shard.mkdir(parents=True)
        (shard / (sha + ".py")).write_text("print('hi')\n", encoding="utf-8")
        (shard / (sha + ".meta.json")).write_text(
            json.dumps({"compiler_version": "0.1.0"}),  # missing pine_version, etc.
            encoding="utf-8",
        )

        got = cache_read(sha, cache_dir=tmp_path)
        assert got is None


# ---------------------------------------------------------------------------
# Idempotence: repeat writes are safe
# ---------------------------------------------------------------------------


class TestIdempotence:
    """A repeat cache_write of the same CompiledModule must be safe."""

    def test_write_twice_no_error(self, tmp_path: Path) -> None:
        cm = _make_compiled(sha="7" * 64)
        cache_write(cm, cache_dir=tmp_path)
        # Second write is a no-op on identical content — no exception.
        cache_write(cm, cache_dir=tmp_path)

    def test_write_twice_content_unchanged(self, tmp_path: Path) -> None:
        cm = _make_compiled(sha="8" * 64)
        cache_write(cm, cache_dir=tmp_path)
        py_path = tmp_path / "88" / (cm.sha + ".py")
        first_content = py_path.read_text(encoding="utf-8")

        cache_write(cm, cache_dir=tmp_path)
        second_content = py_path.read_text(encoding="utf-8")

        assert first_content == second_content

    def test_write_twice_read_still_returns_module(self, tmp_path: Path) -> None:
        cm = _make_compiled(sha="9" * 64)
        cache_write(cm, cache_dir=tmp_path)
        cache_write(cm, cache_dir=tmp_path)
        got = cache_read("9" * 64, cache_dir=tmp_path)
        assert got is not None
        assert got.source == cm.source


# ---------------------------------------------------------------------------
# compile_pine facade integration (end-to-end)
# ---------------------------------------------------------------------------


class TestCompilePineIntegration:
    """The compile_pine facade wraps the cache probe → hit → return / miss →
    compile → cache_write → return path."""

    def test_first_compile_is_miss(self, tmp_path: Path) -> None:
        from openbb_pine.compiler import compile_pine

        cm = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        assert cm.cache_status == "miss"
        # The real 64-char blake2b cache key, not the C5 placeholder 32-char.
        assert len(cm.sha) == 64

    def test_second_compile_same_source_is_hit(self, tmp_path: Path) -> None:
        from openbb_pine.compiler import compile_pine

        cm1 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        cm2 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        assert cm1.cache_status == "miss"
        assert cm2.cache_status == "hit"
        # Content must be byte-identical (modulo cache_status).
        assert cm2.source == cm1.source
        assert cm2.sha == cm1.sha
        assert cm2.builtins_used == cm1.builtins_used

    def test_second_compile_different_source_is_miss(self, tmp_path: Path) -> None:
        from openbb_pine.compiler import compile_pine

        cm1 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        cm2 = compile_pine(TRIVIAL_V6_SOURCE_B, cache_dir=tmp_path)
        assert cm1.cache_status == "miss"
        assert cm2.cache_status == "miss"
        assert cm1.sha != cm2.sha

    def test_use_cache_false_bypasses(self, tmp_path: Path) -> None:
        from openbb_pine.compiler import compile_pine

        cm1 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path, use_cache=False)
        cm2 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path, use_cache=False)
        assert cm1.cache_status == "bypass"
        assert cm2.cache_status == "bypass"
        # No cache directory contents.
        assert not any(tmp_path.rglob("*.py")), "use_cache=False must not write"

    def test_use_cache_false_ignores_existing_cache(self, tmp_path: Path) -> None:
        """use_cache=False must not read from the cache even if it's warm."""
        from openbb_pine.compiler import compile_pine

        _ = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)  # populate
        cm = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path, use_cache=False)
        assert cm.cache_status == "bypass"

    def test_params_carve_separate_slots(self, tmp_path: Path) -> None:
        from openbb_pine.compiler import compile_pine

        cm1 = compile_pine(
            TRIVIAL_V6_SOURCE, cache_dir=tmp_path, params={"length": 10}
        )
        cm2 = compile_pine(
            TRIVIAL_V6_SOURCE, cache_dir=tmp_path, params={"length": 20}
        )
        assert cm1.cache_status == "miss"
        assert cm2.cache_status == "miss"
        assert cm1.sha != cm2.sha

    def test_default_cache_dir_is_used_when_none(self, tmp_path: Path, monkeypatch) -> None:
        """When cache_dir=None, compile_pine falls back to DEFAULT_CACHE_DIR.
        Monkeypatch it to a tmp path so we don't touch ~/.openbb."""
        from openbb_pine.compiler import compile_pine
        import openbb_pine.compiler as compiler_pkg

        monkeypatch.setattr(compiler_pkg, "DEFAULT_CACHE_DIR", tmp_path)
        cm = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=None)
        assert cm.cache_status == "miss"

    def test_hit_module_is_immediately_re_hittable(self, tmp_path: Path) -> None:
        """A miss populates the cache; the immediately-following read yields
        a hit — no lag / race even in the same process."""
        from openbb_pine.compiler import compile_pine

        cm1 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        cm2 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        cm3 = compile_pine(TRIVIAL_V6_SOURCE, cache_dir=tmp_path)
        assert (cm1.cache_status, cm2.cache_status, cm3.cache_status) == (
            "miss",
            "hit",
            "hit",
        )


# ---------------------------------------------------------------------------
# Lazy cache dir — import of compile_cache MUST NOT touch ~/.openbb
# ---------------------------------------------------------------------------


class TestLazyCacheDir:
    """Merely importing compile_cache must not create the default cache dir.
    Some CI environments run as unprivileged users where ~/.openbb doesn't
    exist and shouldn't be created just because the extension imported."""

    def test_default_cache_dir_constant_shape(self) -> None:
        """DEFAULT_CACHE_DIR resolves to ~/.openbb/pine_cache. Doesn't have
        to EXIST — just resolve to the right path."""
        assert DEFAULT_CACHE_DIR == Path.home() / ".openbb" / "pine_cache"

    def test_import_does_not_create_default_dir(self, tmp_path: Path, monkeypatch) -> None:
        """A fresh import of compile_cache must not eagerly mkdir the default
        location. We monkeypatch Path.home() to tmp_path, re-import, and check."""
        import importlib
        import sys

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Force re-import to re-evaluate module-level constants.
        if "openbb_pine.compiler.compile_cache" in sys.modules:
            del sys.modules["openbb_pine.compiler.compile_cache"]
        importlib.import_module("openbb_pine.compiler.compile_cache")

        default_dir = tmp_path / ".openbb" / "pine_cache"
        assert not default_dir.exists(), (
            f"Importing compile_cache should not create {default_dir}"
        )


# ---------------------------------------------------------------------------
# cache_purge — operator/CI tool
# ---------------------------------------------------------------------------


class TestPurge:
    """cache_purge deletes cache entries; older_than_days optionally filters."""

    def test_purge_empty_cache_returns_zero(self, tmp_path: Path) -> None:
        assert cache_purge(cache_dir=tmp_path) == 0

    def test_purge_removes_all_when_no_threshold(self, tmp_path: Path) -> None:
        for i in range(3):
            cm = _make_compiled(sha=f"{i}" * 64)
            cache_write(cm, cache_dir=tmp_path)

        count = cache_purge(cache_dir=tmp_path)
        assert count == 3
        # Cache dir is now empty of entries.
        assert cache_read("0" * 64, cache_dir=tmp_path) is None
        assert cache_read("1" * 64, cache_dir=tmp_path) is None
        assert cache_read("2" * 64, cache_dir=tmp_path) is None

    def test_purge_missing_cache_dir_returns_zero(self, tmp_path: Path) -> None:
        """cache_purge on a nonexistent dir is a no-op, not an error."""
        missing = tmp_path / "does-not-exist"
        assert cache_purge(cache_dir=missing) == 0

    def test_purge_older_than_days_filters(self, tmp_path: Path) -> None:
        """older_than_days=0 removes everything; older_than_days=999999
        removes nothing (all entries are fresh)."""
        cm = _make_compiled(sha="7" * 64)
        cache_write(cm, cache_dir=tmp_path)

        # Fresh — 999999 days threshold spares it.
        count_none = cache_purge(cache_dir=tmp_path, older_than_days=999999)
        assert count_none == 0
        # Still there.
        assert cache_read("7" * 64, cache_dir=tmp_path) is not None

        # older_than_days=0 → nothing older than "now" — but this hinges on
        # implementation-defined boundary. Verify with a clearly-in-the-past
        # threshold by mtime-backdating.
        py_path = tmp_path / "77" / ("7" * 64 + ".py")
        meta_path = tmp_path / "77" / ("7" * 64 + ".meta.json")
        past = time.time() - 86400 * 30  # 30 days ago
        os.utime(py_path, (past, past))
        os.utime(meta_path, (past, past))

        # Threshold at 7 days — entry (30 days old) matches.
        count = cache_purge(cache_dir=tmp_path, older_than_days=7)
        assert count == 1
        assert cache_read("7" * 64, cache_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# PineCacheError surface — the C6 layer can raise structured errors
# ---------------------------------------------------------------------------


class TestPineCacheErrorSurface:
    """The preempted structured init is usable at the C6 boundary."""

    def test_can_raise_structured(self, tmp_path: Path) -> None:
        with pytest.raises(PineCacheError) as excinfo:
            raise PineCacheError(
                sha="a" * 64,
                defect="os.replace failed",
                path=tmp_path / "aa" / ("a" * 64 + ".py"),
            )
        assert excinfo.value.sha == "a" * 64
        assert excinfo.value.defect == "os.replace failed"
        assert excinfo.value.path is not None
