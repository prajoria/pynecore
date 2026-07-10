"""Hand-rolled DFA scanner for Pine Script v5 / v6 source.

Authoritative source: D1 §1.2-§1.4. Hand-rolled (not lark) because:

1. ``//@version=`` is a **column-0-only** prelude that dispatches grammar
   dialect before the body lexes — awkward in lark, trivial in a prelude.
2. Pine's **significant indentation** needs INDENT/DEDENT bookkeeping native
   to the lexer; lark's ``_INDENT`` recipe has known footguns when indent
   interacts with parenthesized expressions (D1 §1.2).
3. Token-level diagnostics want ``(file, line, col)`` attached directly.

Public surface:

* :class:`Token` — frozen, slotted dataclass.
* :func:`tokenize` — ``str -> list[Token]``; raises
  :class:`openbb_pine.errors.PineSyntaxError` on mismatched indentation or
  malformed source.

This module emits tokens only — parsing is C2's concern.
"""

from __future__ import annotations

from dataclasses import dataclass

from openbb_pine.compiler_errors import PineSyntaxError

__all__ = ["Token", "tokenize"]


@dataclass(frozen=True, slots=True)
class Token:
    """One lexical token. ``line`` and ``col`` are 1-based (D1 §2.2)."""

    kind: str
    text: str
    line: int
    col: int


# ---------------------------------------------------------------------------
# Token-class tables (D1 §1.4: ~60 distinct kinds for full Pine v6)
# ---------------------------------------------------------------------------

# Single-char operators / punctuation -> token kind.
_SINGLE_CHAR: dict[str, str] = {
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    ";": "SEMICOLON",
    ".": "DOT",
    ":": "COLON",
    "?": "QMARK",
    "+": "OP_PLUS",
    "-": "OP_MINUS",
    "*": "OP_STAR",
    "/": "OP_SLASH",
    "%": "OP_PERCENT",
    "<": "OP_LT",
    ">": "OP_GT",
    "=": "ASSIGN",
}

# Two-char operators (checked before single-char so longest match wins).
_TWO_CHAR: dict[str, str] = {
    "==": "OP_EQ",
    "!=": "OP_NEQ",
    "<=": "OP_LE",
    ">=": "OP_GE",
    ":=": "WALRUS",
    "=>": "ARROW",
}

_OPENERS = "([{"
_CLOSERS = ")]}"

_PRAGMA = "//@version="


def tokenize(src: str) -> list[Token]:
    """Tokenize a Pine source string into a list of :class:`Token` (D1 §1.2).

    The first token is ``AT_VERSION`` if and only if ``src`` starts with the
    ``//@version=`` prelude at column 0; otherwise ``//@…`` is treated as a
    regular comment.

    Always terminates with an ``EOF`` token; any open indent levels emit
    closing ``DEDENT`` tokens immediately before ``EOF``.
    """
    return _Lexer(src).run()


# ---------------------------------------------------------------------------
# DFA state machine — kept internal so consumers depend on `tokenize` only.
# ---------------------------------------------------------------------------


class _Lexer:
    """Single-pass DFA scanner with INDENT/DEDENT bookkeeping (D1 §1.2)."""

    __slots__ = (
        "src",
        "pos",
        "line",
        "col",
        "tokens",
        "indent_stack",
        "paren_depth",
        "at_line_start",
    )

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []
        # Stack of indent levels (column counts). Always non-empty; floor at 0.
        self.indent_stack: list[int] = [0]
        # Parenthesis nesting (incl. brackets / braces). > 0 suppresses
        # NEWLINE emission and indent processing (D1 §1.2).
        self.paren_depth = 0
        # True at the start of every physical line so indent handling fires.
        self.at_line_start = True

    # -------------------- main loop --------------------

    def run(self) -> list[Token]:
        self._maybe_consume_version_pragma()
        while self.pos < len(self.src):
            if self.at_line_start and self.paren_depth == 0:
                self._handle_indent()
                # _handle_indent may early-return on blank/comment-only lines;
                # we always need to fall through to _lex_one for the rest.
            self.at_line_start = False
            self._lex_one()
        # Close any open indent levels before EOF (D1 §1.2).
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token("DEDENT", "", self.line, self.col))
        self.tokens.append(Token("EOF", "", self.line, self.col))
        return self.tokens

    # -------------------- prelude --------------------

    def _maybe_consume_version_pragma(self) -> None:
        """Detect the column-0 ``//@version=N`` pragma and emit AT_VERSION."""
        if not self.src.startswith(_PRAGMA):
            return
        i = len(_PRAGMA)
        j = i
        while j < len(self.src) and self.src[j].isdigit():
            j += 1
        if j == i:
            raise PineSyntaxError(
                f"malformed //@version= pragma at line 1: missing version number"
            )
        version = self.src[i:j]
        self.tokens.append(Token("AT_VERSION", version, 1, 1))
        self.pos = j
        # Skip rest of pragma line, including the terminating \n if any.
        while self.pos < len(self.src) and self.src[self.pos] != "\n":
            self.pos += 1
        if self.pos < len(self.src) and self.src[self.pos] == "\n":
            self.pos += 1
            self.line += 1
            self.col = 1
            self.at_line_start = True

    # -------------------- indentation --------------------

    def _handle_indent(self) -> None:
        """Compute indent depth and emit INDENT/DEDENT or raise on mismatch."""
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos] in (" ", "\t"):
            self.pos += 1
        indent = self.pos - start
        self.col = 1 + indent

        # Blank or comment-only lines don't move the indent stack.
        if self.pos >= len(self.src):
            return
        ch = self.src[self.pos]
        if ch in ("\n", "\r"):
            return
        if ch == "/" and self.pos + 1 < len(self.src) and self.src[self.pos + 1] == "/":
            return

        current = self.indent_stack[-1]
        if indent > current:
            self.indent_stack.append(indent)
            self.tokens.append(Token("INDENT", "", self.line, 1))
        elif indent < current:
            while self.indent_stack and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                self.tokens.append(Token("DEDENT", "", self.line, 1))
            if not self.indent_stack or self.indent_stack[-1] != indent:
                raise PineSyntaxError(
                    f"mismatched indentation at line {self.line}: dedent to "
                    f"level {indent} doesn't match any outer level"
                )

    # -------------------- main lexer --------------------

    def _lex_one(self) -> None:
        ch = self.src[self.pos]

        # Horizontal whitespace
        if ch in (" ", "\t"):
            self.pos += 1
            self.col += 1
            return

        # CRLF / lone CR — fold to \n behavior
        if ch == "\r":
            self.pos += 1
            return

        # Newline — emit NEWLINE only when outside parens
        if ch == "\n":
            if self.paren_depth == 0 and self._last_token_was_content():
                self.tokens.append(Token("NEWLINE", "\n", self.line, self.col))
            self.pos += 1
            self.line += 1
            self.col = 1
            self.at_line_start = True
            return

        # Line comment
        if ch == "/" and self.pos + 1 < len(self.src) and self.src[self.pos + 1] == "/":
            while self.pos < len(self.src) and self.src[self.pos] != "\n":
                self.pos += 1
                self.col += 1
            return

        # String literal
        if ch == '"' or ch == "'":
            self._lex_string(ch)
            return

        # Numeric literal
        if ch.isdigit():
            self._lex_number()
            return
        if (
            ch == "."
            and self.pos + 1 < len(self.src)
            and self.src[self.pos + 1].isdigit()
        ):
            self._lex_number()
            return

        # Identifier / keyword (Pine keywords are lex'd as NAME; parser
        # classifies — keeps the lexer trivially extensible).
        if ch.isalpha() or ch == "_":
            self._lex_name()
            return

        # Two-char operators (longest match)
        rest = self.src[self.pos : self.pos + 2]
        if rest in _TWO_CHAR:
            self.tokens.append(Token(_TWO_CHAR[rest], rest, self.line, self.col))
            self.pos += 2
            self.col += 2
            return

        # Single-char operators / punctuation
        if ch in _SINGLE_CHAR:
            self.tokens.append(Token(_SINGLE_CHAR[ch], ch, self.line, self.col))
            self.pos += 1
            self.col += 1
            if ch in _OPENERS:
                self.paren_depth += 1
            elif ch in _CLOSERS:
                # Underflow is a syntax error; caught at the parser. We clamp
                # at 0 so INDENT/DEDENT logic stays sane.
                self.paren_depth = max(0, self.paren_depth - 1)
            return

        raise PineSyntaxError(
            f"unexpected character {ch!r} at line {self.line} col {self.col}"
        )

    # -------------------- helpers: literals --------------------

    def _lex_string(self, quote: str) -> None:
        start_col = self.col
        start_line = self.line
        self.pos += 1
        self.col += 1
        buf: list[str] = []
        escapes = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "\\": "\\",
            '"': '"',
            "'": "'",
        }
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c == quote:
                self.pos += 1
                self.col += 1
                self.tokens.append(
                    Token("STRING", "".join(buf), start_line, start_col)
                )
                return
            if c == "\n":
                raise PineSyntaxError(
                    f"unterminated string at line {start_line} col {start_col}"
                )
            if c == "\\":
                self.pos += 1
                self.col += 1
                if self.pos >= len(self.src):
                    raise PineSyntaxError(
                        f"unterminated string escape at line {start_line}"
                    )
                esc = self.src[self.pos]
                buf.append(escapes.get(esc, esc))
                self.pos += 1
                self.col += 1
                continue
            buf.append(c)
            self.pos += 1
            self.col += 1
        raise PineSyntaxError(
            f"unterminated string starting at line {start_line} col {start_col}"
        )

    def _lex_number(self) -> None:
        """Lex int / float / leading-dot / scientific notation literals."""
        start = self.pos
        start_col = self.col
        if self.src[self.pos] == ".":
            # leading-dot float (.5)
            self.pos += 1
            self.col += 1
            self._consume_digits()
        else:
            self._consume_digits()
            # Optional fractional part, only if followed by a digit so `x.foo`
            # tokenizes as NAME DOT NAME instead of bleeding into the number.
            if (
                self.pos < len(self.src)
                and self.src[self.pos] == "."
                and self.pos + 1 < len(self.src)
                and self.src[self.pos + 1].isdigit()
            ):
                self.pos += 1
                self.col += 1
                self._consume_digits()
        # Optional exponent
        if self.pos < len(self.src) and self.src[self.pos] in ("e", "E"):
            self.pos += 1
            self.col += 1
            if self.pos < len(self.src) and self.src[self.pos] in ("+", "-"):
                self.pos += 1
                self.col += 1
            self._consume_digits()
        text = self.src[start : self.pos]
        self.tokens.append(Token("NUMBER", text, self.line, start_col))

    def _consume_digits(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos].isdigit():
            self.pos += 1
            self.col += 1

    def _lex_name(self) -> None:
        start = self.pos
        start_col = self.col
        while self.pos < len(self.src) and (
            self.src[self.pos].isalnum() or self.src[self.pos] == "_"
        ):
            self.pos += 1
            self.col += 1
        text = self.src[start : self.pos]
        self.tokens.append(Token("NAME", text, self.line, start_col))

    # -------------------- helpers: NEWLINE suppression --------------------

    def _last_token_was_content(self) -> bool:
        """Return True if the most recent token is real content (not NEWLINE).

        Suppresses runs of duplicate NEWLINEs from blank/comment-only lines —
        the parser sees one NEWLINE per logical line.
        """
        if not self.tokens:
            return False
        last = self.tokens[-1].kind
        return last not in ("NEWLINE", "INDENT", "DEDENT")
