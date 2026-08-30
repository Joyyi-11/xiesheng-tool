"""Tests for the Markdown renderer."""

from src.models.schemas import KeyPoint, Keyword, OutputDoc, QuestionItem
from src.renderer.markdown import build_output_markdown, fmt_ts


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
        assert "**要点**：证据" in md
        assert "> 这是一句原话。" in md

    def test_renders_footer_when_costs_present(self):
        doc = _make_doc(costs={"llm": 0.02}, timings={"process": 60})
        md = build_output_markdown(doc)
        assert "耗时 校订 1分0秒" in md
        assert "LLM 费用约 0.0200 元" in md

    def test_no_footer_without_metadata(self):
        md = build_output_markdown(_make_doc())
        assert "---" not in md

    def test_renders_optional_quote_as_blockquote(self):
        doc = _make_doc(
            key_points=[
                KeyPoint("有原话的要点", "证据", "这是值得单独收录的原话。"),
                KeyPoint("无原话的要点", "证据"),
            ],
        )
        md = build_output_markdown(doc)
        assert "## 核心观点" in md
        assert "## 闪光语句" not in md
        assert "> 这是值得单独收录的原话。" in md

    def test_renders_keywords_section(self):
        doc = _make_doc(
            keywords=[Keyword(key="术语", desc="解释"), Keyword(key="无解释")]
        )
        md = build_output_markdown(doc)
        assert "## 关键词" in md
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
        assert "1. 找不到工作是谁的问题？" in md
        assert "   策略错位。" in md
        assert "2. 副业该不该搞？" in md
        assert "   反哺主业才好。" in md
        # 问题与思考位于核心观点之后、人物简介之前
        assert md.index("## 核心观点") < md.index("## 问题与思考") < md.index("## 人物简介")

    def test_no_summary_or_questions_section_when_empty(self):
        md = build_output_markdown(_make_doc())
        assert "## 摘要" not in md
        assert "## 问题与思考" not in md
