"""Tests for the Markdown renderer."""

from src.models.schemas import Highlight, KeyPoint, Keyword, OutlineItem, OutputDoc
from src.renderer.markdown import build_output_markdown, fmt_ts


def _make_doc(**kw):
    defaults = {
        "title": "标题",
        "podcast_name": "播客",
        "pub_date": "2026-01-01",
        "show_notes": "节目简介",
        "key_points": [KeyPoint("要点", "证据")],
        "highlight_quotes": ["闪光语句。"],
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

    def test_renders_key_points_and_quotes(self):
        md = build_output_markdown(_make_doc())
        assert "**要点**：证据" in md
        assert "- 闪光语句。" in md

    def test_renders_footer_when_costs_present(self):
        doc = _make_doc(costs={"llm": 0.02}, timings={"process": 60})
        md = build_output_markdown(doc)
        assert "耗时 校订 1分0秒" in md
        assert "LLM 费用约 0.0200 元" in md

    def test_no_footer_without_metadata(self):
        md = build_output_markdown(_make_doc())
        assert "---" not in md

    def test_renders_enriched_highlights_with_ts_and_speaker(self):
        doc = _make_doc(
            highlight_quotes=["旧格式"],
            highlights=[
                Highlight(content="这句话很精彩。", start_sec=125, speaker="主持人连漪"),
                Highlight(content="没有元数据。"),
            ],
        )
        md = build_output_markdown(doc)
        assert "## 闪光语句" in md
        assert "[02:05]（主持人连漪） 「这句话很精彩。」" in md
        assert "- 「没有元数据。」" in md
        assert "旧格式" not in md  # 有新版 highlights 时不再回退旧字段

    def test_renders_keywords_section(self):
        doc = _make_doc(
            keywords=[Keyword(key="术语", desc="解释"), Keyword(key="无解释")]
        )
        md = build_output_markdown(doc)
        assert "## 关键词" in md
        assert "- **术语**：解释" in md
        assert "- **无解释**" in md

    def test_renders_mermaid_outline_after_highlights(self):
        doc = _make_doc(
            outline=[
                OutlineItem(title="主题一", points=["子点 A", "子点 B"], start_sec=0),
                OutlineItem(title="主题二", start_sec=300),
            ]
        )
        md = build_output_markdown(doc)
        assert "## 大纲" in md
        assert "```mermaid" in md
        assert "mindmap" in md
        assert "root((标题))" in md
        assert '"00:00 主题一"' in md
        assert "      子点 A" in md
        assert '"05:00 主题二"' in md
        # 大纲位于闪光语句之后
        assert md.index("## 闪光语句") < md.index("## 大纲")

    def test_outline_without_time_uses_plain_title(self):
        doc = _make_doc(outline=[OutlineItem(title="无时间主题")])
        md = build_output_markdown(doc)
        assert "    无时间主题" in md

    def test_no_outline_section_when_empty(self):
        md = build_output_markdown(_make_doc())
        assert "## 大纲" not in md

    def test_skips_outline_nodes_without_title(self):
        doc = _make_doc(outline=[OutlineItem(title="", points=["孤儿子点"])])
        md = build_output_markdown(doc)
        assert "## 大纲" not in md
