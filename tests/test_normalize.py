"""Tests for deterministic text normalization (project house style).

覆盖四组确定性规则：引号规范化（转录落盘时用）、外文词后半角标点转全角、
≥3 连重复字合并、中文读数日期转阿拉伯数字、核心观点句末引语补句末标点
（后四组为成稿级，2026-09-15 起由 `python -m src.processor.normalize`
兜底，源自公开仓 issue #1 / #2）。
"""

from src.processor.normalize import (
    count_all,
    count_changes,
    count_cn_date,
    count_quote_period,
    count_text,
    merge_repeat_chars,
    normalize_cn_date,
    normalize_en_punct,
    normalize_markdown,
    normalize_quote_period,
    normalize_quotes,
    scan_quote_period,
    sync_handoff_quotes,
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


class TestCnDate:
    """中文读数日期 → 阿拉伯数字（2026年9月28日）。

    ASR 的逆文本正则化常把日期落成中文读数，扫读困难也不利于检索。
    已知边界：只有「年+月」无「日」的形态（二零二四年五月）目前保持原样——
    这类写法在口语里也可能只是随口带过，先不覆盖以避免误伤。
    """

    def test_converts_full_date(self):
        assert normalize_cn_date("二零二六年九月二十八日发布") == "2026年9月28日发布"

    def test_converts_with_spoken_comma_between_year_and_month(self):
        # 口语常在年与月之间停顿，ASR 落成逗号
        assert normalize_cn_date("二零二六年，九月二十八日星期一发布") == "2026年9月28日星期一发布"

    def test_converts_year_only(self):
        assert normalize_cn_date("二零二三年那篇") == "2023年那篇"

    def test_converts_inside_sentence(self):
        assert normalize_cn_date("mrbeast 在二零二五年说") == "mrbeast 在2025年说"

    def test_converts_multiple_dates(self):
        src = "从二零一四年二月十八日到二零二六年六月二日"
        assert normalize_cn_date(src) == "从2014年2月18日到2026年6月2日"

    def test_keeps_cn_numerals_in_enumerations(self):
        # 一是……二是……三是…… 是列举，不是日期
        src = "一是省钱，二是省力，三是省心"
        assert normalize_cn_date(src) == src

    def test_keeps_out_of_range_values(self):
        # 听错的越界值改了反而是 misinformation，保持原样留给校订环节
        assert normalize_cn_date("十三月三日") == "十三月三日"

    def test_keeps_year_month_without_day(self):
        assert normalize_cn_date("二零二四年五月") == "二零二四年五月"

    def test_count_matches_conversion(self):
        src = "二零二六年，九月二十八日与二零一四年二月十八日"
        assert count_cn_date(src) == 2
        assert normalize_cn_date(src).count("年") == 2

    def test_idempotent(self):
        once = normalize_cn_date("二零二六年，九月二十八日发布")
        assert normalize_cn_date(once) == once

    def test_empty_input(self):
        assert normalize_cn_date("") == ""


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


class TestQuotePeriod:
    """核心观点：句末引用原话须自带句末标点（句号在引号内，疑问句用问号）。"""

    def test_adds_period_inside_closing_bracket(self):
        src = "……因为流量会真的涌进来。「互联网上的大多数操作都会由 Agent 来完成」\n"
        assert normalize_quote_period(src) == (
            "……因为流量会真的涌进来。「互联网上的大多数操作都会由 Agent 来完成。」\n"
        )

    def test_question_mark_for_interrogative(self):
        # 只补句号会把「……要我试着修一下吗」压成陈述句
        src = "他拒绝了修复的请求。「他说要我试着修一下吗」\n"
        assert normalize_quote_period(src).endswith("吗？」\n")

    def test_embedded_phrase_is_untouched(self):
        # 嵌在句中的短语引述本就不带句号，补了就是病句
        src = "工作的基本单位从「职位」变为「任务」，这才是重点。\n"
        assert normalize_quote_period(src) == src

    def test_already_punctuated_is_untouched(self):
        src = "……因为流量会真的涌进来。「……由 Agent 来完成。」\n"
        assert normalize_quote_period(src) == src

    def test_period_after_bracket_is_left_to_the_validator(self):
        # 「……」后接句号：位置错了，但已有标点，机械层不动（交给校验器报）
        src = "……因为流量会真的涌进来。「……由 Agent 来完成」。\n"
        assert normalize_quote_period(src) == src

    def test_leading_connector_fragment_needs_human(self):
        # 「而……」是承接残句：补句号仍不是完整句，脚本不自动改
        src = "……每秒多发一个 prompt。「而不只是每秒多发一个 prompt」\n"
        assert normalize_quote_period(src) == src
        fixable, manual = scan_quote_period(src)
        assert fixable == []
        assert manual == ["而不只是每秒多发一个 prompt"]

    def test_count_covers_fragments_too(self):
        # 残句不改，但要报出来给人看，所以计数里算
        src = "「而是一个残句」\n「这是一个完整句」\n"
        assert count_quote_period(src) == 2

    def test_idempotent(self):
        src = "……因为流量会真的涌进来。「……由 Agent 来完成」\n"
        once = normalize_quote_period(src)
        assert normalize_quote_period(once) == once

    def test_only_applies_to_key_point_section(self):
        md = (
            "## 摘要\n\n关键是「留下来用」\n\n"
            "## 核心观点\n\n- **观点。**展开。「关键是使用」\n\n"
            "## 原文转录\n\n【主播】他说「留下来用」\n"
        )
        out = normalize_markdown(md)
        assert "关键是「留下来用」" in out          # 摘要不动
        assert "他说「留下来用」" in out            # 转录不动
        assert "- **观点。**展开。「关键是使用。」" in out
        assert count_all(md)["quote_period"] == 1

    def test_sync_handoff_quote_keeps_anchor(self):
        doc = {
            "key_points": [
                {"quote": "互联网上的大多数操作都会由 Agent 来完成",
                 "anchor": "互联网上的大多数操作都会由 Agent 来完成"}
            ]
        }
        n = sync_handoff_quotes(doc, ["互联网上的大多数操作都会由 Agent 来完成"])
        assert n == 1
        kp = doc["key_points"][0]
        assert kp["quote"].endswith("完成。")
        assert kp["anchor"].endswith("完成")  # 锚点须与转录逐字一致，不能动
