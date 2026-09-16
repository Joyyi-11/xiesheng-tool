"""Tests for deterministic text normalization (project house style).

覆盖三组确定性规则：引号规范化（转录落盘时用）、外文词后半角标点转全角、
≥3 连重复字合并（后两组为成稿级，2026-09-15 起由 `python -m src.processor.normalize`
兜底，源自公开仓 issue #1 / #2）。
"""

from src.processor.normalize import (
    count_changes,
    count_text,
    merge_repeat_chars,
    normalize_en_punct,
    normalize_markdown,
    normalize_quotes,
)


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


class TestEnPunct:
    """外文词后的半角标点转全角（rules.EDIT_EN_PUNCT 的确定性落地）。"""

    def test_converts_comma_before_chinese(self):
        # issue #1 实测样本：标点紧贴中文
        assert normalize_en_punct("有一个 gap,那个 gap 的填充") == "有一个 gap，那个 gap 的填充"

    def test_converts_after_space_before_chinese(self):
        # issue #1 实测样本：标点后空格再中文
        assert normalize_en_punct("买了 token, 它并不会真的被消耗掉") == "买了 token， 它并不会真的被消耗掉"

    def test_converts_split_letter_words(self):
        # ASR 把英文词切成单字母（c e o. ）——最常见的漏项形态
        assert normalize_en_punct("一个企业的 c e o.他说不可能") == "一个企业的 c e o。他说不可能"

    def test_converts_at_line_end_when_line_has_chinese(self):
        assert normalize_en_punct("我们会给每个人尽可能多的 Token.") == "我们会给每个人尽可能多的 Token。"

    def test_keeps_english_word_internal_punct(self):
        # Node.js / v1.5 的标点属词内，不是中文标点
        src = "数据科学家自己安装 Node.js，版本 v1.5 可用"
        assert normalize_en_punct(src) == src

    def test_keeps_pure_english_line(self):
        # 整行英文（嘉宾口述英文）不转
        src = "lead on taste, curation, and building"
        assert normalize_en_punct(src) == src

    def test_converts_when_bold_marker_sits_between_word_and_punct(self):
        """回标加粗常把标点一起包进 `**`，此时标点前是 `*` 而非字母。

        回归背景：用 `(?<=[A-Za-z])` 做前置断言时整片漏掉（实测 13 份稿 38 处），
        故正则改为「字母 + `*{0,2}` + 标点」三组捕获。
        """
        assert normalize_en_punct("那个 **Genspark**,他就是一个想法") == "那个 **Genspark**，他就是一个想法"
        assert normalize_en_punct("**全员 marketing**,就是 marketing") == "**全员 marketing**，就是 marketing"

    def test_keeps_english_sentence_punct_inside_chinese_line(self):
        """整句英文引用常夹在含中文的行里，其句内标点不转。"""
        src = "他说 Talk is cheap, show me the code 这句"
        assert normalize_en_punct(src) == src

    def test_keeps_latin_abbreviations(self):
        src = "（e.g. 这种写法）和在 U.S. 的公司"
        assert normalize_en_punct(src) == src

    def test_empty_input(self):
        assert normalize_en_punct("") == ""


class TestMergeRepeatChars:
    """≥3 连相同汉字合并为一个（rules.EDIT_REDO 的确定性部分）。"""

    def test_merges_three_or_more(self):
        assert merge_repeat_chars("你你你培训") == "你培训"
        assert merge_repeat_chars("最最最最重") == "最重"
        assert merge_repeat_chars("又又又又又用") == "又用"

    def test_keeps_double_repeats(self):
        # 刚刚 / 看看 / 慢慢 是正常叠词，2 连一律不动
        for s in ("刚刚", "看看", "慢慢", "好好"):
            assert merge_repeat_chars(s) == s

    def test_keeps_onomatopoeia(self):
        for s in ("哈哈哈", "咚咚咚", "哗哗哗", "吱吱吱"):
            assert merge_repeat_chars(s) == s

    def test_keeps_interjection_repeats(self):
        # 对对对是强烈认同、好好好是应答，删成单字会改变语气
        for s in ("对对对", "好好好", "是是是"):
            assert merge_repeat_chars(s) == s

    def test_empty_input(self):
        assert merge_repeat_chars("") == ""


class TestNormalizeMarkdown:
    """成稿级入口：Show Notes 整节透传。"""

    MD = (
        "# 标题\n\n> 来源：播客 | 标题 | 2026-09-15\n\n"
        "## Show Notes\n\n嘉宾是 OpenAI 的 Sam, 他聊了 GPT.\n\n"
        "## 摘要\n\n本期聊 agent, 很有意思。\n\n"
        "## 原文转录\n\n【主播】你你你好，这个 agent, 我们聊聊。\n"
    )

    def test_converts_outside_show_notes(self):
        out = normalize_markdown(self.MD)
        assert "本期聊 agent， 很有意思。" in out
        assert "你好，这个 agent， 我们聊聊。" in out

    def test_show_notes_is_passed_through(self):
        # Show Notes 是节目源简介透传，不归校订管
        out = normalize_markdown(self.MD)
        assert "嘉宾是 OpenAI 的 Sam, 他聊了 GPT." in out

    def test_count_changes_matches_conversion(self):
        hits_en, hits_rp = count_changes(self.MD)
        assert hits_en == 2          # 摘要 1 处 + 转录 1 处
        assert hits_rp == 1          # 你你你
        # 干跑计数应与实际改动一致
        assert count_text(self.MD) != (hits_en, hits_rp)  # 未分节时含 Show Notes

    def test_idempotent(self):
        once = normalize_markdown(self.MD)
        assert normalize_markdown(once) == once
