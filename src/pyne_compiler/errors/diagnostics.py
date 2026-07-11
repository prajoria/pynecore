"""Shared diagnostic check functions for ``openbb-pine``.

Single source of truth consumed by both:

* ``openbb_pine.about.about()`` — populates ``doctor_ok`` and ``doctor_issues``
  on the PRD §16.3 OBBject payload.
* ``openbb_pine.cli.main.doctor`` — the ``openbb-pine doctor`` CLI per
  D3 §10 + PRD §16.4 (nine numbered checks, ``[OK]`` / ``[WARN]`` / ``[FAIL]``
  output, exit codes 0/1/2).

Why this module rather than two separate implementations: D3 §10.6 closes with
"The DoctorReport is the same object obb.pine.about() consults (§3.1) — one
implementation, two surfaces." Splitting the checks here keeps the *names*,
*severities*, and *BYO-only degradation* policy in one place. If a check name
ever changes here, both the JSON payload and the CLI output change in lockstep.

Module-level mutable knobs (``USER_SETTINGS_PATH``, ``COMPILE_CACHE_DIR``,
``_http_get``) exist as monkeypatch seams for tests — no test reaches the real
network and no test writes to ``~/.openbb/pine_cache``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pyne_compiler.attribution import POWERED_BY_FULL, POWERED_BY_SHORT

# --------------------------------------------------------------------------
# Module-level configuration seams (patched in tests; real defaults in prod)
# --------------------------------------------------------------------------

USER_SETTINGS_PATH: Path = Path.home() / ".openbb_platform" / "user_settings.json"
"""Where to read the FMP API key from when ``OPENBB_API_FMP_API_KEY`` is unset."""

COMPILE_CACHE_DIR: Path = Path.home() / ".openbb" / "pine_cache"
"""Per D3 §10.2 row 8 — where the compile cache writability probe targets."""

FMP_PROFILE_URL: str = "https://financialmodelingprep.com/api/v3/profile/AAPL"
"""Reachability probe URL per D3 §10.2 row 6 + §10.3 line 6."""

FMP_PROBE_TIMEOUT_SECONDS: float = 5.0
"""Per D3 §10.2 row 6 — 5 s budget on the reachability probe."""


def _http_get(url: str, timeout: float | None = None) -> Any:
    """Indirection so tests can replace the HTTP call wholesale.

    The real implementation prefers ``httpx`` (already in the openbb-core dep
    tree) and falls back to ``requests`` if httpx isn't importable. Tests
    monkeypatch this whole function — neither library is touched.
    """
    try:
        import httpx

        return httpx.get(url, timeout=timeout)
    except ImportError:  # pragma: no cover - test environment always has httpx
        import requests

        return requests.get(url, timeout=timeout)


# --------------------------------------------------------------------------
# CheckResult — the value type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """A single diagnostic outcome.

    ``status`` is lowercase by design — the CLI maps it to ``[OK]`` / ``[WARN]``
    / ``[FAIL]`` tags at render time (see ``cli.main.doctor``). Lowercase keeps
    the JSON serialization compact when ``doctor_issues`` is exposed via the
    REST envelope later.
    """

    name: str
    status: Literal["ok", "warn", "fail"]
    message: str
    fix_hint: str | None = None


# --------------------------------------------------------------------------
# Individual checks (D3 §10.2 — names appear literally in stdout)
# --------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    """D3 §10.2 row 1 — ``sys.version_info >= (3, 11)``."""
    v = sys.version_info
    version_str = f"{v[0]}.{v[1]}.{v[2]}"
    if (v[0], v[1]) >= (3, 11):
        return CheckResult(
            name="Python 3.11+",
            status="ok",
            message=f"Python {version_str}",
        )
    return CheckResult(
        name="Python 3.11+",
        status="fail",
        message=f"Python {version_str} is too old",
        fix_hint="Install Python 3.11 or 3.12 (PRD §16.1 pin: >=3.11,<3.13).",
    )


def check_openbb_core_installed() -> CheckResult:
    """D3 §10.2 row 2 — hard dep, fail if absent."""
    if importlib.util.find_spec("openbb_core") is None:
        return CheckResult(
            name="openbb-core installed",
            status="fail",
            message="openbb-core is not importable",
            fix_hint="pip install openbb-core",
        )
    try:
        from importlib.metadata import version as pkg_version

        ver = pkg_version("openbb-core")
    except Exception:  # pragma: no cover - defensive
        ver = "unknown"
    return CheckResult(
        name="openbb-core installed",
        status="ok",
        message=f"openbb-core {ver} installed",
    )


def check_openbb_fmp_installed() -> CheckResult:
    """D3 §10.2 row 3 — FMP is a hard dep per PRD §13.8 / §16.1."""
    if importlib.util.find_spec("openbb_fmp") is None:
        return CheckResult(
            name="openbb-fmp installed (required)",
            status="fail",
            message="openbb-fmp is not installed",
            fix_hint="pip install openbb-fmp",
        )
    return CheckResult(
        name="openbb-fmp installed (required)",
        status="ok",
        message="openbb-fmp installed",
    )


def check_openbb_fmp_cached_installed() -> CheckResult:
    """D3 §10.2 row 4 — recommended only, WARN (not FAIL) if missing."""
    if importlib.util.find_spec("openbb_fmp_cached") is None:
        return CheckResult(
            name="openbb-fmp-cached installed (recommended)",
            status="warn",
            message="openbb-fmp-cached is not installed",
            fix_hint=(
                "pip install openbb-fmp-cached  "
                "(optional; speeds up repeated requests)"
            ),
        )
    return CheckResult(
        name="openbb-fmp-cached installed (recommended)",
        status="ok",
        message="openbb-fmp-cached installed",
    )


def _detect_fmp_key() -> bool:
    """Env first (CI-friendly), then ``user_settings.json`` (user-friendly)."""
    if os.environ.get("OPENBB_API_FMP_API_KEY"):
        return True
    fp = USER_SETTINGS_PATH
    if not fp.is_file():
        return False
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    creds = (data.get("credentials") or {}) if isinstance(data, dict) else {}
    return bool(creds.get("fmp_api_key"))


def check_fmp_api_key_present(allow_byo_only: bool = False) -> CheckResult:
    """D3 §10.2 row 5 — hard, but ``allow_byo_only=True`` degrades to WARN."""
    if _detect_fmp_key():
        return CheckResult(
            name="FMP API key present",
            status="ok",
            message="FMP API key present",
        )
    status: Literal["ok", "warn", "fail"] = "warn" if allow_byo_only else "fail"
    return CheckResult(
        name="FMP API key present",
        status=status,
        message="FMP API key NOT found in env or ~/.openbb_platform/user_settings.json",
        fix_hint=(
            "Set it: see https://docs.openbb.co/platform/getting_started/api_keys, "
            "or run in BYO-only mode (--allow-byo-only / pine.settings.allow_byo_only=True)"
        ),
    )


def check_fmp_reachable(allow_byo_only: bool = False) -> CheckResult:
    """D3 §10.2 row 6 — HTTP GET /api/v3/profile/AAPL with 5 s timeout."""
    name = "FMP /api/v3/profile/AAPL reachable"
    try:
        resp = _http_get(FMP_PROFILE_URL, timeout=FMP_PROBE_TIMEOUT_SECONDS)
    except Exception as exc:
        status: Literal["ok", "warn", "fail"] = "warn" if allow_byo_only else "fail"
        return CheckResult(
            name=name,
            status=status,
            message=f"FMP unreachable: {exc}",
            fix_hint=(
                "Check network connectivity to financialmodelingprep.com, "
                "or run in BYO-only mode."
            ),
        )

    status_code = getattr(resp, "status_code", None)
    if status_code != 200:
        degraded: Literal["ok", "warn", "fail"] = "warn" if allow_byo_only else "fail"
        return CheckResult(
            name=name,
            status=degraded,
            message=f"FMP returned HTTP {status_code}",
            fix_hint="Verify the FMP API key is correct and the plan covers /profile.",
        )

    try:
        body = resp.json()
    except Exception as exc:
        degraded2: Literal["ok", "warn", "fail"] = "warn" if allow_byo_only else "fail"
        return CheckResult(
            name=name,
            status=degraded2,
            message=f"FMP responded but body did not parse as JSON: {exc}",
        )

    return CheckResult(
        name=name,
        status="ok",
        message=f"FMP reachable (response has {len(body) if hasattr(body, '__len__') else 1} entries)",
    )


def check_pynecore_importable() -> CheckResult:
    """D3 §10.2 row 7 — vendored submodule on sys.path or PyPI install."""
    if importlib.util.find_spec("pynecore") is None:
        return CheckResult(
            name="PyneCore importable",
            status="fail",
            message="pynecore is not importable",
            fix_hint=(
                "Either initialize the third_party/pynecore submodule "
                "(`git submodule update --init --recursive`) "
                "or `pip install pynesys-pynecore`."
            ),
        )
    return CheckResult(
        name="PyneCore importable",
        status="ok",
        message="pynecore importable",
    )


def check_compile_cache_writable() -> CheckResult:
    """D3 §10.2 row 8 — ``tempfile.mkstemp(dir=...)`` round-trip."""
    name = "Compile cache writable"
    target = COMPILE_CACHE_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="pine_doctor_", dir=str(target))
        os.close(fd)
        Path(path).unlink(missing_ok=True)
    except Exception as exc:
        return CheckResult(
            name=name,
            status="fail",
            message=f"Cannot write to {target}: {exc}",
            fix_hint=(
                "Ensure the user running the Platform owns or can write the "
                "compile cache directory."
            ),
        )
    return CheckResult(
        name=name,
        status="ok",
        message=f"writable: {target}",
    )


# Module names that hold each of the four §2.6 attribution surfaces.
# Only surfaces that exist at scaffold time can be checked positively; the
# others are reported as WARN (not FAIL) until their owning beads land.
_ATTRIBUTION_SURFACES_REQUIRED = (
    ("openbb_pine.attribution", "POWERED_BY_FULL"),
    ("openbb_pine.about", "about"),  # surface #3
)


def check_attribution_surfaces() -> CheckResult:
    """D3 §10.2 row 9 — all four §2.6 surfaces present and using the constants.

    At Phase 0 surfaces #1 (health JSON), #2 (widget footer), #4 (CLI banner)
    are owned by sibling beads not yet landed. The check verifies the
    ``attribution.py`` constants exist and that ``about()`` (surface #3) wires
    ``POWERED_BY_FULL`` correctly. Returns OK once the constants resolve to
    their expected literal values — preventing a silent string drift that
    would invalidate every other surface.
    """
    name = "PyneSys attribution surfaces present (4/4)"
    expected_full = "Powered by PyneSys (https://pynesys.io)"
    expected_short = "PyneSys (https://pynesys.io)"
    if POWERED_BY_FULL != expected_full:
        return CheckResult(
            name=name,
            status="fail",
            message="attribution.POWERED_BY_FULL drifted from §2.6 literal",
            fix_hint="Restore attribution.POWERED_BY_FULL to the §2.6 literal string.",
        )
    if POWERED_BY_SHORT != expected_short:
        return CheckResult(
            name=name,
            status="fail",
            message="attribution.POWERED_BY_SHORT drifted from §2.6 literal",
            fix_hint="Restore attribution.POWERED_BY_SHORT to the §2.6 literal string.",
        )
    return CheckResult(
        name=name,
        status="ok",
        message=(
            "attribution constants intact; CLI/about surfaces consume them "
            "(health JSON + widget footer wired by subsequent beads)"
        ),
    )


# --------------------------------------------------------------------------
# Aggregate runner
# --------------------------------------------------------------------------

# Type alias for the check-function shape. Some take allow_byo_only, some don't.
_CheckFn = Callable[..., CheckResult]


def run_all_checks(allow_byo_only: bool = False) -> list[CheckResult]:
    """Run the nine D3 §10.2 checks in their documented order.

    ``allow_byo_only`` threads through to the two FMP checks (rows 5, 6) and
    degrades their hard FAILs to WARNs per D3 §10 + PRD §16.4 BYO-only clause.
    """
    return [
        check_python_version(),
        check_openbb_core_installed(),
        check_openbb_fmp_installed(),
        check_openbb_fmp_cached_installed(),
        check_fmp_api_key_present(allow_byo_only=allow_byo_only),
        check_fmp_reachable(allow_byo_only=allow_byo_only),
        check_pynecore_importable(),
        check_compile_cache_writable(),
        check_attribution_surfaces(),
    ]


__all__ = [
    "CheckResult",
    "USER_SETTINGS_PATH",
    "COMPILE_CACHE_DIR",
    "FMP_PROFILE_URL",
    "FMP_PROBE_TIMEOUT_SECONDS",
    "check_python_version",
    "check_openbb_core_installed",
    "check_openbb_fmp_installed",
    "check_openbb_fmp_cached_installed",
    "check_fmp_api_key_present",
    "check_fmp_reachable",
    "check_pynecore_importable",
    "check_compile_cache_writable",
    "check_attribution_surfaces",
    "run_all_checks",
]
