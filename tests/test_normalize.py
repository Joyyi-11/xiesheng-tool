"""Tests for deterministic text normalization (project house style).

覆盖确定性规则：引号规范化（转录落盘时用）、外文词后半角标点转全角、
≥3 连重复字合并、中文读数日期转阿拉伯数字、中文读数数量转阿拉伯数字、
原文转录话轮末逗号转句号、核心观点句末引语补句末标点
（成稿级规则由 `python -m src.processor.normalize` 兜底，源自公开仓 issue #1 / #2）。
"""

from src.processor.normalize import (
    count_all,
    count_changes,
    count_cn_date,
    count_cn_number,
    count_punct,
    count_quote_period,
    count_text,
    merge_repeat_chars,
    normalize_cn_date,
    normalize_cn_number,
    normalize_en_punct,
    normalize_cjk_halfwidth,
    normalize_markdown,
    normalize_orphan_punct,
    normalize_punct,
    normalize_punct_cluster,
    normalize_punct_space,
    normalize_quote_period,
    normalize_quotes,
    normalize_turn_terminal,
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
        # 原来的「半角逗号转全角」之后，停顿空格也一并清掉（见 TestPunctSpace）
        assert "本期聊 agent，很有意思。" in out
        assert "你好，这个 agent，我们聊聊。" in out

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


class TestCnNumber:
    """中文读数数量 → 阿拉伯数字（保留万/亿单位，避开日期与成语）。"""

    def test_two_digit_with_measure(self):
        assert normalize_cn_number("十六家合作伙伴") == "16家合作伙伴"

    def test_scale_unit_preserved(self):
        assert normalize_cn_number("多达十二亿用户") == "多达12亿用户"

    def test_full_integer(self):
        assert normalize_cn_number("三百六十五天") == "365天"

    def test_two_wan(self):
        assert normalize_cn_number("两万用户") == "2万用户"

    def test_thousands_resolved(self):
        assert normalize_cn_number("三万五千人") == "35000人"

    def test_idiom_untouched(self):
        # 单字读数不单独转，成语天然避开
        assert normalize_cn_number("他一心一意地做") == "他一心一意地做"
        assert normalize_cn_number("五颜六色的灯") == "五颜六色的灯"

    def test_blocklist_words(self):
        # 万一＝倘若、千万＝务必，不是数量
        assert normalize_cn_number("万一来不及") == "万一来不及"
        assert normalize_cn_number("千万不要这样") == "千万不要这样"

    def test_date_guard_not_touched(self):
        # 后接 月/日/号/年 的读数归日期规则管
        assert normalize_cn_number("十三月") == "十三月"
        assert normalize_cn_number("二零二六年发布") == "二零二六年发布"

    def test_applies_to_all_sections(self):
        md = (
            "## 核心观点\n\n- **亮点。**已开放给十六家合作伙伴。\n\n"
            "## 原文转录\n\n【Tibo】我们有一十六家伙伴。\n"
        )
        out = normalize_markdown(md)
        assert "16家合作伙伴" in out
        assert "16家伙伴" in out

    def test_count_and_idempotent(self):
        src = "十六家、十二亿用户、三百六十五"
        assert count_cn_number(src) == 3
        once = normalize_cn_number(src)
        assert count_cn_number(once) == 0
        assert normalize_cn_number(once) == once

    def test_skips_ambiguous_and_merged_runs(self):
        # 宁可漏转也不错转：下列读数实为「X 或 Y」概数范围，或两个相邻数量被
        # 贪心正则黏在一起，一律保留中文，绝不腐蚀成单值。
        # 省略千位的口语（一万五 = 1万5千）与跨数量边界（一亿两 = 1亿 + 两周）也跳过。
        cases = [
            "一万五",          # 万后裸数字、空后缀 → 跳过（曾腐蚀成 10005）
            "一亿两周",         # 亿后裸数字、空后缀 → 跳过（曾腐蚀成 100000002周）
            "五十五十",         # 重复进位字「十」→ 两个数量黏连，跳过
            "四十五十",         # 同上
            "十一二十",         # 重复进位字「十」→ 跳过
            "十八十九",         # 同上
            "三十万二十万",       # 重复进位字「万」→ 跳过（曾腐蚀成 300020万）
            "七八十",           # 连续单字读数 → 概数范围，跳过
            "两三百",           # 同上
            "四五十",           # 同上
            "五六百",           # 同上
            "一两千",           # 同上
            "三四万",           # 同上（曾错转成 4万）
            "二三十万",         # 同上（曾错转成 30万）
            "三六万",           # 同上（曾错转成 6万）
            "一二一百",         # 连续单字 → 跳过
            "十一五",           # 连续单字 + 非数量词，跳过
            "一二",            # 连续单字（概数）→ 跳过
            "三五",            # 同上
            "亿万富翁",         # 「亿万」是词不是数量 → 跳过
            "数百万人",         # 概数前缀「数」→ 跳过
            "数百",            # 同上
        ]
        for s in cases:
            assert normalize_cn_number(s) == s, f"应保留中文: {s!r} -> {normalize_cn_number(s)!r}"

    def test_still_converts_well_formed(self):
        # 上述守卫不得误伤真正规范的数量读数
        assert normalize_cn_number("五百") == "500"
        assert normalize_cn_number("十一万两千") == "112000"
        assert normalize_cn_number("十一万五千") == "115000"
        assert normalize_cn_number("七十八") == "78"
        assert normalize_cn_number("五十五") == "55"


class TestTurnTerminal:
    """原文转录：话轮末逗号须转句号（话轮边界即句末）。"""

    def test_speaker_turn_comma_to_period(self):
        src = "【Tibo】你唯一需要配置的就是想通过哪些渠道跟他交流，\n"
        assert normalize_turn_terminal(src) == "【Tibo】你唯一需要配置的就是想通过哪些渠道跟他交流。\n"

    def test_multiple_turns(self):
        src = (
            "【Tibo】它还会继续发生相当根本的变化。直到有一天他变得顺手了，\n"
            "【Lenny】谢谢你来参加这次访谈，\n"
            "【Tibo】也谢谢你邀请我。当然这房间不错，\n"
        )
        out = normalize_turn_terminal(src)
        assert out.count("，\n") == 0
        assert out.count("。\n") == 3

    def test_non_speaker_line_untouched(self):
        # 无 【说话人】 前缀的行（结构性/续行）不误改
        src = "这是一句没说完的话，\n"
        assert normalize_turn_terminal(src) == src

    def test_already_period_untouched(self):
        src = "【Tibo】他说完了。\n"
        assert normalize_turn_terminal(src) == src

    def test_only_applies_to_transcript_section(self):
        md = (
            "## 摘要\n\n他提到一个未说完的观点，\n\n"
            "## 原文转录\n\n【Lenny】谢谢你来参加这次访谈，\n"
        )
        out = normalize_markdown(md)
        assert "未说完的观点，" in out          # 摘要不动
        assert "谢谢你来参加这次访谈。" in out  # 转录转句号

    def test_idempotent(self):
        src = "【Tibo】讲到一半，\n"
        once = normalize_turn_terminal(src)
        assert normalize_turn_terminal(once) == once


class TestPunctSpace:
    """① 停顿空格：全角标点两侧不得有多余空格。"""

    def test_space_after_punct_removed(self):
        # ASR 在停顿处先留空格后补标点，空格就夹在了标点与下文之间
        assert normalize_punct_space("你好啊，我是嘉宾， 欢迎来到节目") == (
            "你好啊，我是嘉宾，欢迎来到节目"
        )

    def test_space_after_punct_before_latin_removed(self):
        # 中文—西文之间按库内 93.7% 的无空格惯例（1970 : 132）
        assert normalize_punct_space("他有数据， what 以及为什么") == "他有数据，what 以及为什么"

    def test_space_before_punct_removed(self):
        assert normalize_punct_space("包括种种困惑 。") == "包括种种困惑。"
        assert normalize_punct_space("3 分钟 ：商业策划") == "3 分钟：商业策划"

    def test_multiple_spaces_collapsed(self):
        assert normalize_punct_space("排列，   然后") == "排列，然后"

    def test_english_context_space_kept(self):
        # 半角标点不属于清空格的字符集，西文句内的正常空格必须原样保留
        assert normalize_punct_space("Talk is cheap, show me the code") == (
            "Talk is cheap, show me the code"
        )

    def test_no_false_positive_on_cjk_latin_space(self):
        # 中文与西文之间的空格是正确排版，不能误删
        assert normalize_punct_space("我在用 Claude 写代码") == "我在用 Claude 写代码"

    def test_source_line_separator_kept(self):
        # 来源行用 ` | ` 分字段，双向空格是结构的一部分（干跑实测曾被吃掉）
        src = "> 来源：播客 | 你的职业价值只剩下野心与判断力？ | 2026-08-31"
        assert normalize_punct_space(src) == src

    def test_latin_initials_kept(self):
        # 西文人名缩写：句点间的空格删了会挤成一坨（R。F。Kuang）
        src = "**匡灵秀（R。 F。 Kuang）**：作家、教师"
        assert normalize_punct_space(src) == src

    def test_english_clause_space_kept(self):
        # 英文从句/歌词整句被转成全角标点后，空格是英文书写所必须
        src = "You're on your own kid， you always have been"
        assert normalize_punct_space(src) == src

    def test_space_removed_before_cjk_after_latin(self):
        # 英文术语后接中文：空格一律删（这是 ASR 留空格的重灾区）
        assert normalize_punct_space("就是一个编程 Agent， 我们做出来了") == (
            "就是一个编程 Agent，我们做出来了"
        )


class TestPunctCluster:
    """② 堆叠标点：冗余停顿删除、同类合并取首个。"""

    def test_pause_before_terminal_removed(self):
        assert normalize_punct_cluster("你要水电煤工人，。建筑工人") == "你要水电煤工人。建筑工人"
        assert normalize_punct_cluster("什么技能、？什么软技能") == "什么技能？什么软技能"

    def test_pause_after_terminal_removed(self):
        assert normalize_punct_cluster("业务就是kpi。，这时候") == "业务就是kpi。这时候"

    def test_adjacent_pause_keeps_first(self):
        # 保留首个标点：前一个停顿决定了此处该用什么停顿
        assert normalize_punct_cluster("包括愤怒、，悲伤") == "包括愤怒、悲伤"
        assert normalize_punct_cluster("蛋白粉啊，，鱼油啊") == "蛋白粉啊，鱼油啊"

    def test_stacked_terminal_collapsed(self):
        assert normalize_punct_cluster("选择来念nba。。小的团体") == "选择来念nba。小的团体"

    def test_mixed_mood_punct_kept(self):
        # 「？！」「！？」是合法语气叠加，不是堆叠错误
        assert normalize_punct_cluster("你确定吗？！") == "你确定吗？！"

    def test_normal_sequence_untouched(self):
        assert normalize_punct_cluster("第一，第二，第三。") == "第一，第二，第三。"


class TestCjkHalfwidth:
    """③ 汉字后紧跟的半角标点（normalize_en_punct 漏网的那一半）。"""

    def test_comma_after_cjk(self):
        assert normalize_cjk_halfwidth("循环在跑,并且自己判断") == "循环在跑，并且自己判断"

    def test_comma_with_space_after_cjk(self):
        assert normalize_cjk_halfwidth("发推特, 他们员工也鼓励") == "发推特， 他们员工也鼓励"

    def test_colon_semicolon_after_cjk(self):
        assert normalize_cjk_halfwidth("听听例子:塞利米耶清真寺") == "听听例子：塞利米耶清真寺"
        assert normalize_cjk_halfwidth("社群教育代表; 用AI升职") == "社群教育代表； 用AI升职"

    def test_terminal_halfwidth_at_line_end(self):
        # 行内以汉字为主体时才转：时间戳、英文标题一类的行不动
        assert normalize_cjk_halfwidth("感谢听众的耐心聆听!\n") == "感谢听众的耐心聆听！\n"

    def test_english_enumeration_untouched(self):
        # 西文列举句：下一个非空格字符是拉丁字母，整句保持原样
        src = "Special Guests: Kolento, 林桐宇, and 饶杰武."
        assert normalize_cjk_halfwidth(src) == src

    def test_latin_comma_untouched(self):
        assert normalize_cjk_halfwidth("Talk is cheap, show me the code") == (
            "Talk is cheap, show me the code"
        )

    def test_ellipsis_becomes_chinese(self):
        assert normalize_cjk_halfwidth("但对人类文明有...") == "但对人类文明有……"

    def test_version_number_untouched(self):
        assert normalize_cjk_halfwidth("版本号 v1.5 起") == "版本号 v1.5 起"


class TestOrphanPunct:
    """④ 句首错位：换行把标点甩到下一行行首。"""

    def test_leading_pause_removed(self):
        assert normalize_orphan_punct("上句完。\n\n，包括其实线下也有活动\n") == (
            "上句完。\n\n包括其实线下也有活动\n"
        )

    def test_leading_pause_with_quote_marker(self):
        assert normalize_orphan_punct("> ，以及线下的活动\n") == "> 以及线下的活动\n"

    def test_wrapped_terminal_merged(self):
        # 软换行导致句号掉到下一行首，应与前一行合并
        assert normalize_orphan_punct("会出现生存困境\n。相比之下好一些\n") == (
            "会出现生存困境。相比之下好一些\n"
        )

    def test_not_merged_when_prev_is_complete(self):
        # 前一行已是完整句时合并会产生双句号，留给人工看
        assert normalize_orphan_punct("前句已收尾。\n。后句也有句号\n") == (
            "前句已收尾。\n。后句也有句号\n"
        )

    def test_not_merged_into_list_item(self):
        assert normalize_orphan_punct("- 列表项没有结尾\n。接上去会错吗\n") == (
            "- 列表项没有结尾\n。接上去会错吗\n"
        )

    def test_trailing_newline_preserved(self):
        # 空行不能被误判为「以句末标点开头」而与上一行合并吃掉文末换行
        assert normalize_orphan_punct("段落甲。\n") == "段落甲。\n"


class TestPunctIntegration:
    """四组合一的口径：计数与转换一致、幂等、Show Notes 不动。"""

    def test_count_matches_conversion(self):
        for src in (
            "甲说这话， 然后走了",
            "工人，。建筑工人",
            "Special Guests: Kolento, and 饶杰武.",
            "干净的一句话。",
        ):
            assert count_punct(src) == (0 if normalize_punct(src) == src else 1), src

    def test_idempotent(self):
        src = "他说完， 然后又说，、再补一句， 结束了。 "
        once = normalize_punct(src)
        assert normalize_punct(once) == once

    def test_show_notes_skipped(self):
        md = (
            "## Show Notes\n\n"
            "主持人说 hello， 这里有停顿空格， 别动它\n\n"
            "## 摘要\n\n"
            "这同一类是 witty， 应该被清掉\n"
        )
        out = normalize_markdown(md)
        assert "主持人说 hello， 这里有停顿空格， 别动它" in out  # 源简介透传
        assert "这同一类是 witty，应该被清掉" in out              # 校订范围内的标点被清

    def test_section_punct_unified(self):
        # 同一段里四类缺陷叠在一起时，都能清干净（数量与标点一并收敛）
        md = (
            "## 摘要\n\n"
            "嘉宾提到 ABC， 然后说明 CD。、最后举了个例,收尾在十一分钟。\n"
        )
        out = normalize_markdown(md)
        assert "ABC，然后说明 CD。" in out
        assert "CD。、最后" not in out
        assert "举了个例，收尾在11分钟。" in out

