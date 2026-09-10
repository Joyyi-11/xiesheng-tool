"""Tests for the Markdown renderer."""

from src.models.schemas import KeyPoint, Keyword, OutputDoc, QuestionItem, TermDef
from src.renderer.markdown import (
    _apply_bold_anchors,
    _collect_bold_anchors,
    build_output_markdown,
    fmt_ts,
)


def _make_doc(**kw):
    defaults = {
        "title": "标题",
        "podcast_name": "播客",
        "pub_date": "2026-01-01",
        "show_notes": "节目简介",
        "key_points": [KeyPoint("要点", "证据", "这是一句原话。")],
        "speaker_intro": "**主播**：某人",
        "full_text": "正文内容。",
        "costs": {},
        "timings": {},
    }
    defaults.update(kw)
    return OutputDoc(**defaults)


class TestFmtTs:
    def test_mm_ss(self):
        assert fmt_ts(5 * 60 + 49) == "05:49"

    def test_h_mm_ss(self):
        assert fmt_ts(3600 + 5 * 60 + 49) == "1:05:49"

    def test_none(self):
        assert fmt_ts(None) == "??:??"


class TestBuildOutputMarkdown:
    def test_renders_source_with_three_parts(self):
        md = build_output_markdown(_make_doc())
        assert "> 来源：播客 | 标题 | 2026-01-01" in md

    def test_renders_core_viewpoints_and_quotes(self):
        md = build_output_markdown(_make_doc())
        assert "## 核心观点" in md
        # 三段式：观点句、展开论述句、引用句顺次相接，不用冒号
        assert "**要点。** 证据。「这是一句原话。」" in md

    def test_renders_transcript_heading_from_shared_constant(self):
        # 小节标题由 src.utils.TRANSCRIPT_HEADING 统一定义，渲染器不得硬编码
        md = build_output_markdown(_make_doc())
        assert "## 原文转录" in md
        assert md.rstrip().splitlines()[-1] == "正文内容。"

    def test_renders_footer_when_costs_present(self):
        doc = _make_doc(costs={"llm": 0.02}, timings={"process": 60})
        md = build_output_markdown(doc)
        assert "耗时 校订 1分0秒" in md
        assert "LLM 费用约 0.0200 元" in md

    def test_no_footer_without_metadata(self):
        md = build_output_markdown(_make_doc())
        assert "---" not in md

    def test_renders_inline_quote_not_blockquote(self):
        doc = _make_doc(
            key_points=[
                KeyPoint("有原话的要点", "证据", "这是值得单独收录的原话。"),
                KeyPoint("无原话的要点", "证据"),
                KeyPoint("只有原话的要点", "", "原话单独出现。"),
            ],
        )
        md = build_output_markdown(doc)
        assert "## 核心观点" in md
        assert "## 闪光语句" not in md
        # 原话用直角引号行内化，不再单独成 > 引用块
        assert "**有原话的要点。** 证据。「这是值得单独收录的原话。」" in md
        assert "**无原话的要点。** 证据。" in md
        assert "**只有原话的要点。** 「原话单独出现。」" in md
        assert "> 这是值得单独收录的原话。" not in md

    def test_renders_terms_as_indented_subitems(self):
        doc = _make_doc(
            key_points=[
                KeyPoint(
                    "用第一性原理找工作。",
                    "把求职当完整项目拆解",
                    "Offer 不是终点。",
                    terms=[
                        TermDef("第一性原理", "把问题拆到最底层要素，从本质出发规划"),
                        TermDef("第二曲线", "主业之外的新方向，应反哺主业"),
                    ],
                ),
                KeyPoint("普通观点", "证据"),
            ],
        )
        md = build_output_markdown(doc)
        assert "**用第一性原理找工作。** 把求职当完整项目拆解。「Offer 不是终点。」" in md
        assert "  - **第一性原理**：把问题拆到最底层要素，从本质出发规划。" in md
        assert "  - **第二曲线**：主业之外的新方向，应反哺主业。" in md
        assert "**普通观点。** 证据。" in md

    def test_key_point_colon_tail_is_downgraded_to_period(self):
        # 观点句若以冒号收尾（旧「主题句：展开」写法），脚本确定性降级为句号，不输出冒号
        doc = _make_doc(
            key_points=[KeyPoint("AI 产品的第三个时代：常驻同事：", "Tara 把演进划为三段")]
        )
        md = build_output_markdown(doc)
        kp_block = md.split("## 核心观点")[1].split("## 问题与思考")[0]
        assert "**AI 产品的第三个时代：常驻同事。** Tara 把演进划为三段。" in kp_block
        assert "**：" not in kp_block

    def test_renders_terms_glossary_section(self):
        doc = _make_doc(
            keywords=[Keyword(key="术语", desc="解释"), Keyword(key="无解释")]
        )
        md = build_output_markdown(doc)
        assert "## 术语表" in md
        assert "## 关键词" not in md
        assert "- **术语**：解释。" in md
        assert "- **无解释**" in md

    def test_renders_summary_before_key_points(self):
        doc = _make_doc(summary="本期聊了求职。")
        md = build_output_markdown(doc)
        assert "## 摘要" in md
        assert "本期聊了求职。" in md
        # 摘要位于 Show Notes 之后、核心观点之前
        assert md.index("## Show Notes") < md.index("## 摘要") < md.index("## 核心观点")

    def test_renders_show_notes_even_when_empty(self):
        doc = _make_doc(show_notes="", summary="本期聊了求职。")
        md = build_output_markdown(doc)
        assert "## Show Notes" in md
        assert "（无）" in md
        # 空时仍稳定位于摘要之前
        assert md.index("## Show Notes") < md.index("## 摘要")

    def test_renders_questions_after_highlights(self):
        doc = _make_doc(
            questions=[
                QuestionItem(question="找不到工作是谁的问题？", answer="策略错位。"),
                QuestionItem(question="副业该不该搞？", answer="反哺主业才好。"),
            ]
        )
        md = build_output_markdown(doc)
        assert "## 问题与思考" in md
        # 问题整句加粗、序号留在加粗外；答案不加粗
        assert "1. **找不到工作是谁的问题？**" in md
        assert "   策略错位。" in md
        assert "2. **副业该不该搞？**" in md
        assert "   反哺主业才好。" in md

    def test_question_bold_is_idempotent(self):
        # question 自带 ** 时不得双重加粗
        doc = _make_doc(questions=[QuestionItem(question="**已经加粗的问题？**", answer="答案。")])
        md = build_output_markdown(doc)
        assert "1. **已经加粗的问题？**" in md
        assert "****" not in md
        # 问题与思考位于核心观点之后、人物简介之前
        assert md.index("## 核心观点") < md.index("## 问题与思考") < md.index("## 人物简介")

    def test_keeps_required_sections_when_summary_and_questions_empty(self):
        md = build_output_markdown(_make_doc(summary="", questions=[]))
        assert "## 摘要" in md
        assert "## 问题与思考" in md
        assert "## 术语表" in md
        assert "## 人物简介" in md

    def test_renders_none_for_empty_speaker_intro(self):
        md = build_output_markdown(_make_doc(speaker_intro=""))
        assert "## 人物简介\n\n（无）" in md

    def test_renders_quality_warnings(self):
        md = build_output_markdown(_make_doc(warnings=["分块 1 校订失败，已回退原稿"]))
        assert "> [WARN] 分块 1 校订失败，已回退原稿" in md


class TestTranscriptAnchorBolding:
    """原文转录回标锚点加粗（连漪：核心观点句/问题句/术语在原文首次出现处加粗）。"""

    def _doc(self, full_text, **kw):
        defaults = {
            "title": "标题",
            "podcast_name": "播客",
            "pub_date": "2026-01-01",
            "show_notes": "简介",
            "speaker_intro": "**主播**：某人",
            "key_points": [],
            "questions": [],
            "keywords": [],
        }
        defaults.update(kw)
        return OutputDoc(full_text=full_text, **defaults)

    def test_bolds_quote_anchor_at_first_occurrence(self):
        doc = self._doc(
            "【主播】大家好，先说一句：模型塌缩不可避免。\n\n【嘉宾】对，模型塌缩不可避免，关键是筛选数据。",
            key_points=[
                KeyPoint(
                    point="模型塌缩不可避免。",
                    evidence="关键在于筛选",
                    quote="模型塌缩不可避免",
                )
            ],
        )
        md = build_output_markdown(doc)
        transcript = md.split("## 原文转录", 1)[1]
        # 首次出现处加粗；第二次出现（嘉宾重复）不再加粗
        assert "【主播】大家好，先说一句：**模型塌缩不可避免**。" in transcript
        assert "【嘉宾】对，模型塌缩不可避免，关键是筛选数据。" in transcript

    def test_bolds_explicit_keypoint_anchor(self):
        doc = self._doc(
            "【主播】欢迎收听。\n\n【嘉宾】我们准备了这次聊天的问题清单。",
            key_points=[
                KeyPoint(point="准备比临场重要。", evidence="证据", anchor="准备了这次聊天的问题清单")
            ],
        )
        md = build_output_markdown(doc)
        assert "**准备了这次聊天的问题清单**" in md

    def test_bolds_question_anchor(self):
        doc = self._doc(
            "【主播】那我想先问，开源模型最大的结构性缺陷是什么？\n\n【嘉宾】数据飞轮转不起来。",
            questions=[
                QuestionItem(
                    question="开源模型最大的结构性缺陷是什么？",
                    answer="数据飞轮转不起来。",
                    anchor="开源模型最大的结构性缺陷是什么",
                )
            ],
        )
        md = build_output_markdown(doc)
        assert "**开源模型最大的结构性缺陷是什么**？" in md

    def test_bolds_glossary_keyword(self):
        doc = self._doc(
            "【嘉宾】数据飞轮转不起来，这是开源模型的结构性难题。",
            keywords=[Keyword(key="数据飞轮", desc="开源模型无法回流用户数据的缺陷")],
        )
        md = build_output_markdown(doc)
        assert "**数据飞轮**转不起来" in md

    def test_speaker_label_never_bolded(self):
        # 锚点若与【标签】同名/同子串，不得把说话人标签加粗
        doc = self._doc(
            "【主播】这里是主播开场。\n\n【嘉宾】我是嘉宾。",
            keywords=[Keyword(key="主播", desc="节目主持人")],
        )
        md = build_output_markdown(doc)
        assert "【**主播**】" not in md

    def test_already_bolded_span_not_double_bolded(self):
        # 长锚先加粗后，其内部的短锚（可能是长锚子串）不得二次加粗
        doc = self._doc(
            "【嘉宾】模型塌缩不可避免，关键在于对生成数据的筛选。",
            key_points=[
                KeyPoint(point="A。", evidence="e", anchor="模型塌缩不可避免，关键在于对生成数据的筛选"),
                KeyPoint(point="B。", evidence="e", anchor="模型塌缩不可避免"),
            ],
        )
        md = build_output_markdown(doc)
        assert "**模型塌缩不可避免，关键在于对生成数据的筛选**。" in md
        assert "****" not in md

    def test_overlong_anchor_skipped(self):
        # 显式 anchor 限 ≤40 字、quote 限 ≤24 字：超长视为噪音跳过，不加粗
        doc = self._doc(
            "【嘉宾】这是一段很长的原文，用来测试超长锚点不会触发加粗行为。",
            key_points=[
                KeyPoint(
                    point="P。",
                    evidence="e",
                    anchor="这是一段很长的原文，用来测试超长锚点不会触发加粗行为（超过四十字）",
                    quote="这是一段很长的原文，用来测试超长锚点不会触发加粗行为（超过二十四字）",
                )
            ],
        )
        md = build_output_markdown(doc)
        assert "**" not in md.split("## 原文转录", 1)[1]

    def test_anchor_missing_in_transcript_silently_skipped(self):
        # 锚点未在转录中出现（校订后措辞变化）时不报错、不改字
        doc = self._doc(
            "【嘉宾】实际转录里是另一句话。",
            key_points=[KeyPoint(point="P。", evidence="e", anchor="原文里并不存在的句子")],
        )
        md = build_output_markdown(doc)
        assert "**" not in md.split("## 原文转录", 1)[1]

    def test_bolds_within_speaker_labeled_transcript_line(self):
        # 标签与正文同行时，只加粗正文部分、标签保留
        md = _apply_bold_anchors("【主播】今天聊模型塌缩。\n【嘉宾】模型塌缩确实存在。", ["模型塌缩"])
        assert md == "【主播】今天聊**模型塌缩**。\n【嘉宾】模型塌缩确实存在。"

    def test_anchor_not_rebolded_across_lines(self):
        # 同一锚点全局只在首次出现处加粗一次
        md = _apply_bold_anchors(
            "【A】第一次出现锚点。\n【B】第二次出现锚点。\n【C】第三次出现锚点。",
            ["出现锚点"],
        )
        assert md.count("**出现锚点**") == 1

    def test_collect_anchors_dedup_and_sort(self):
        # 收集锚点：quote/terms/keywords/anchor 合并、去重、按长度降序
        doc = self._doc(
            "",
            key_points=[
                KeyPoint(point="P。", evidence="e", quote="短引", anchor="一个较长的显式锚点"),
            ],
            keywords=[Keyword(key="短引", desc="d")],
        )
        anchors = _collect_bold_anchors(doc)
        assert anchors == ["一个较长的显式锚点", "短引"]
