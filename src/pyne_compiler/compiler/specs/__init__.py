"""Per-builtin behavioral specs — one module per Pine builtin family.

Each spec carries the signature, qualifier rules, and codegen lowering for
the builtins under its namespace. Populated incrementally during Phase 1+
builtin-implementation beads; empty marker at C4.
"""
