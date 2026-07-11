"""Single source of truth for PyneCore NOTICE section 4(d) attribution strings.

Importing from here is the only approved way to mention the string. The four
section 2.6 surfaces (health, widget footer, about, CLI banner) all import from
this module so PyneCore Apache-2.0 attribution cannot silently drift.
"""

POWERED_BY_SHORT = "PyneSys (https://pynesys.io)"
"""For dict fields where the key already conveys 'powered_by'."""

POWERED_BY_FULL = "Powered by PyneSys (https://pynesys.io)"
"""For prose contexts: widget footer, OBBject extra, CLI banner."""
