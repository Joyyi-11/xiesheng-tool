"""Tests for quote normalization (project house style)."""

from src.processor.normalize import normalize_quotes


class TestNormalizeQuotes:
    def test_pairs_straight_quotes(self):
        assert normalize_quotes('他说"你好"，她说"再见"。') == "他说「你好」，她说「再见」。"

    def test_single_pair(self):
        assert normalize_quotes('"第一段话"') == "「第一段话」"

    def test_unpaired_dangling_quote_is_left_unchanged(self):
        # 不成对的直引号保留原样，避免被错误配成「…
        assert normalize_quotes('只说了一个"引号') == '只说了一个"引号'

    def test_curly_quotes_are_normalized(self):
        assert normalize_quotes('“你好”') == "「你好」"

    def test_empty_input(self):
        assert normalize_quotes("") == ""
