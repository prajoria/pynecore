"""Tests for ``openbb_pine.compiler.lexer`` — hand-rolled Pine DFA scanner.

Source of truth: D1 §1.2-§1.4. Hand-rolled (not lark) because the
``//@version=`` pragma is column-0 only and dispatches grammar dialect before
the body lexes, and Pine's significant indentation needs INDENT/DEDENT
bookkeeping.

The lexer returns ``list[Token]`` where ``Token`` is a frozen slot-ted
dataclass carrying ``kind: str``, ``text: str``, ``line: int``, ``col: int``.

Errors raise ``PineSyntaxError`` imported from ``openbb_pine.errors`` (the
af08128d3 single-class consolidation). The compiler-side ``errors.py`` is a
follow-up bead — for now lexer raises the shared root directly.
"""

from __future__ import annotations

import dataclasses

import pytest


# ---------------------------------------------------------------------------
# Token dataclass shape (D1 §1.2)
# ---------------------------------------------------------------------------


class TestTokenShape:
    def test_token_fields(self) -> None:
        from openbb_pine.compiler.lexer import Token

        t = Token(kind="NAME", text="close", line=2, col=4)
        assert t.kind == "NAME"
        assert t.text == "close"
        assert t.line == 2
        assert t.col == 4

    def test_token_frozen_and_slotted(self) -> None:
        from openbb_pine.compiler.lexer import Token

        t = Token(kind="NAME", text="x", line=1, col=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.text = "y"  # type: ignore[misc]
        assert not hasattr(t, "__dict__")

    def test_token_equality_and_hash(self) -> None:
        from openbb_pine.compiler.lexer import Token

        a = Token(kind="NUMBER", text="20", line=3, col=5)
        b = Token(kind="NUMBER", text="20", line=3, col=5)
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Pragma routing — //@version= is column-0 only, switches grammar (D1 §1.2)
# ---------------------------------------------------------------------------


class TestVersionPragma:
    def test_detects_v6_pragma(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("//@version=6\n")
        kinds = [t.kind for t in tokens]
        assert "AT_VERSION" in kinds
        v = next(t for t in tokens if t.kind == "AT_VERSION")
        assert v.text == "6"
        assert v.line == 1
        assert v.col == 1

    def test_detects_v5_pragma(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("//@version=5\nindicator(\"X\")\n")
        v = next(t for t in tokens if t.kind == "AT_VERSION")
        assert v.text == "5"

    def test_pragma_not_at_column_0_is_just_a_comment(self) -> None:
        """``//@version=`` only routes when it starts at column 0 (D1 §1.2)."""
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("  //@version=6\nx = 1\n")
        assert not any(t.kind == "AT_VERSION" for t in tokens)


# ---------------------------------------------------------------------------
# Indentation — INDENT/DEDENT bookkeeping (D1 §1.2)
# ---------------------------------------------------------------------------


class TestIndentation:
    def test_two_space_if_else_emits_indent_dedent(self) -> None:
        """A 2-space-indented if/else block emits INDENT before the body, DEDENT before
        ``else``, INDENT before the else body, DEDENT at EOF."""
        from openbb_pine.compiler.lexer import tokenize

        src = "if cond\n  x = 1\nelse\n  x = 2\n"
        tokens = tokenize(src)
        kinds = [t.kind for t in tokens]
        # Locate the INDENT around the if-body and DEDENT before else.
        assert kinds.count("INDENT") == 2
        assert kinds.count("DEDENT") == 2

    def test_mismatched_indent_raises_pine_syntax_error(self) -> None:
        """Dedenting to a level that isn't on the indent stack is a PS error."""
        from openbb_pine.errors import PineSyntaxError
        from openbb_pine.compiler.lexer import tokenize

        # `if cond:` body indented 4, then a line indented 2 (not matching any
        # outer level).
        src = "if cond\n    x = 1\n  y = 2\n"
        with pytest.raises(PineSyntaxError):
            tokenize(src)

    def test_no_indent_dedent_inside_parens(self) -> None:
        """Newlines and leading whitespace inside a paren-group are not significant
        per D1 §1.2 (the lark _INDENT silent-failure footgun mentioned)."""
        from openbb_pine.compiler.lexer import tokenize

        src = "x = f(\n  1,\n  2,\n)\n"
        tokens = tokenize(src)
        kinds = [t.kind for t in tokens]
        # No INDENT/DEDENT inside the call; only at EOF
        assert "INDENT" not in kinds


# ---------------------------------------------------------------------------
# D1 §1.4 token-count fixture (the BB snippet)
# ---------------------------------------------------------------------------


# Verbatim from D1 §1.4
BB_FIXTURE = (
    '//@version=6\n'
    'indicator("BB")\n'
    'length = input.int(20, minval=1)\n'
    'mult   = input.float(2.0)\n'
    'basis  = ta.sma(close, length)\n'
    'dev    = mult * ta.stdev(close, length)\n'
    'plot(basis); plot(basis + dev); plot(basis - dev)\n'
)


class TestBollingerFixture:
    def test_pragma_routes_first(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(BB_FIXTURE)
        # Pragma must be the very first non-EOF token (D1 §1.2 column-0 pre-lex)
        assert tokens[0].kind == "AT_VERSION"
        assert tokens[0].text == "6"

    def test_emits_all_expected_token_kinds(self) -> None:
        """The BB snippet exercises NAME, NUMBER (int+float), STRING, parens, comma,
        dot, semicolon, assign, plus/minus/star plus NEWLINE and the pragma."""
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(BB_FIXTURE)
        kinds = {t.kind for t in tokens}
        expected = {
            "AT_VERSION",
            "NAME",
            "NUMBER",
            "STRING",
            "LPAREN",
            "RPAREN",
            "COMMA",
            "DOT",
            "SEMICOLON",
            "ASSIGN",
            "OP_PLUS",
            "OP_MINUS",
            "OP_STAR",
            "NEWLINE",
            "EOF",
        }
        missing = expected - kinds
        assert not missing, f"missing kinds: {missing}"

    def test_no_indent_in_flat_program(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(BB_FIXTURE)
        kinds = [t.kind for t in tokens]
        assert "INDENT" not in kinds
        assert "DEDENT" not in kinds

    def test_string_literal_extracted(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(BB_FIXTURE)
        strs = [t for t in tokens if t.kind == "STRING"]
        # The text should preserve the quote-stripped contents (D1 §1.4 — clean
        # tokens consumable by parser).
        assert any(t.text == "BB" for t in strs)

    def test_keyword_identifiers_preserved_as_name(self) -> None:
        """``ta.sma`` and ``input.int`` come through as NAME-DOT-NAME triples."""
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(BB_FIXTURE)
        # Find `ta.sma`: NAME(ta), DOT, NAME(sma) consecutively.
        for i, t in enumerate(tokens[:-2]):
            if t.kind == "NAME" and t.text == "ta":
                assert tokens[i + 1].kind == "DOT"
                assert tokens[i + 2].kind == "NAME"
                assert tokens[i + 2].text == "sma"
                break
        else:  # pragma: no cover - belt-and-suspenders
            pytest.fail("`ta.sma` triple not found in BB fixture token stream")


# ---------------------------------------------------------------------------
# Strings & comments
# ---------------------------------------------------------------------------


class TestStrings:
    def test_double_quoted_string(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize('x = "hello"\n')
        s = [t for t in tokens if t.kind == "STRING"]
        assert len(s) == 1
        assert s[0].text == "hello"

    def test_single_quoted_string(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 'hi'\n")
        s = [t for t in tokens if t.kind == "STRING"]
        assert len(s) == 1
        assert s[0].text == "hi"

    def test_string_escape_sequence(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize('x = "a\\nb"\n')
        s = next(t for t in tokens if t.kind == "STRING")
        assert s.text == "a\nb"

    def test_unterminated_string_raises(self) -> None:
        from openbb_pine.errors import PineSyntaxError
        from openbb_pine.compiler.lexer import tokenize

        with pytest.raises(PineSyntaxError):
            tokenize('x = "abc\n')


class TestComments:
    def test_line_comment_not_emitted_as_token(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("// just a comment\nx = 1\n")
        # No tokens for the comment line (its own NEWLINE may also be skipped
        # since the comment line is "blank" for indentation purposes).
        kinds = [t.kind for t in tokens]
        assert "COMMENT" not in kinds  # we suppress regular comments
        # `x = 1` must still be there
        names = [t.text for t in tokens if t.kind == "NAME"]
        assert "x" in names

    def test_inline_comment_terminates_at_end_of_line(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 1 // inline\ny = 2\n")
        names = [t.text for t in tokens if t.kind == "NAME"]
        assert names == ["x", "y"]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


class TestNumbers:
    def test_integer(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 42\n")
        n = next(t for t in tokens if t.kind == "NUMBER")
        assert n.text == "42"

    def test_float(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 2.0\n")
        n = next(t for t in tokens if t.kind == "NUMBER")
        assert n.text == "2.0"

    def test_float_no_leading_zero(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = .5\n")
        n = next(t for t in tokens if t.kind == "NUMBER")
        assert n.text == ".5"

    def test_scientific_notation(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 1e6\n")
        n = next(t for t in tokens if t.kind == "NUMBER")
        assert n.text == "1e6"


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestOperators:
    @pytest.mark.parametrize(
        "src, kind",
        [
            ("a == b\n", "OP_EQ"),
            ("a != b\n", "OP_NEQ"),
            ("a < b\n", "OP_LT"),
            ("a <= b\n", "OP_LE"),
            ("a > b\n", "OP_GT"),
            ("a >= b\n", "OP_GE"),
            ("a + b\n", "OP_PLUS"),
            ("a - b\n", "OP_MINUS"),
            ("a * b\n", "OP_STAR"),
            ("a / b\n", "OP_SLASH"),
            ("a % b\n", "OP_PERCENT"),
        ],
    )
    def test_binary_operators(self, src: str, kind: str) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize(src)
        assert any(t.kind == kind for t in tokens), f"expected {kind} in {tokens}"

    def test_walrus_reassign(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x := 1\n")
        assert any(t.kind == "WALRUS" for t in tokens)

    def test_arrow_in_switch(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("k => 1\n")
        assert any(t.kind == "ARROW" for t in tokens)

    def test_qmark_ternary(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("a ? b : c\n")
        kinds = [t.kind for t in tokens]
        assert "QMARK" in kinds
        assert "COLON" in kinds

    def test_brackets(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = arr[i]\n")
        kinds = [t.kind for t in tokens]
        assert "LBRACKET" in kinds
        assert "RBRACKET" in kinds


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


class TestPositions:
    def test_line_and_col_are_1_based(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 1\n")
        x = next(t for t in tokens if t.kind == "NAME")
        assert x.line == 1
        assert x.col == 1

    def test_line_advances_on_newline(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("a = 1\nb = 2\n")
        b = next(t for t in tokens if t.kind == "NAME" and t.text == "b")
        assert b.line == 2
        assert b.col == 1


# ---------------------------------------------------------------------------
# EOF
# ---------------------------------------------------------------------------


class TestEOF:
    def test_eof_emitted_last(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("x = 1\n")
        assert tokens[-1].kind == "EOF"

    def test_empty_source_emits_only_eof(self) -> None:
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("")
        assert [t.kind for t in tokens] == ["EOF"]

    def test_dedent_flushed_before_eof(self) -> None:
        """Indent levels still open at EOF must emit closing DEDENTs."""
        from openbb_pine.compiler.lexer import tokenize

        tokens = tokenize("if c\n  x = 1\n")
        kinds = [t.kind for t in tokens]
        # There should be one INDENT and one corresponding DEDENT before EOF.
        assert kinds.count("INDENT") == kinds.count("DEDENT") == 1
        assert kinds[-1] == "EOF"
