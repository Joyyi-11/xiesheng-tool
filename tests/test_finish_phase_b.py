"""Tests for Phase B idempotent continuation (finish_phase_b)."""

from pathlib import Path

from src.processor.finish_phase_b import (
    PLACEHOLDER_MARK,
    analyze_one,
    build_draft,
    find_diarized,
    md_path_for,
    source_line_fix,
)

SAMPLE_PACKAGE = """\
# 例话：AI 协作实践

> 来源：小宇宙 | 例话 | 2026-08-01

# Show Notes

这期聊人机协作。

# 转录全文
[SPEAKER_00] 大家好，欢迎收听。
[SPEAKER_01] 今天聊 AI。
[SPEAKER_00] 我们开始吧。
"""


class TestFindDiarized:
    def test_glob_and_sort(self, tmp_path):
        (tmp_path / "b_diarized.txt").write_text("x", encoding="utf-8")
        (tmp_path / "a_diarized.txt").write_text("y", encoding="utf-8")
        (tmp_path / "not_package.txt").write_text("z", encoding="utf-8")
        found = [p.name for p in find_diarized(tmp_path)]
        assert found == ["a_diarized.txt", "b_diarized.txt"]


class TestMdPathFor:
    def test_converts_diarized_suffix(self):
        p = md_path_for(Path("output/节目_diarized.txt"))
        assert p.name == "节目.md"


class TestSourceLineFix:
    def test_keeps_valid_line_untouched(self):
        md = "# 标题\n\n> 来源：播客 | 标题 | 2026-01-01\n"
        assert source_line_fix(md, "标题") == md

    def test_inserts_title_into_two_part_line(self):
        md = "# 标题\n\n> 来源：播客 | 2026-01-01\n"
        fixed = source_line_fix(md, "标题")
        assert "> 来源：播客  |  标题  |  2026-01-01" in fixed

    def test_leaves_untouched_when_irreparable(self):
        md = "# 标题\n\n> 来源：只有一段\n"
        assert source_line_fix(md, "标题") == md


class TestBuildDraft:
    def test_v6_shape_with_placeholders_and_full_transcript(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        draft = build_draft(diarized, {})
        assert draft.startswith("# 例话：AI 协作实践\n")
        for heading in ("## 摘要", "## 内容提要", "## 闪光语句",
                        "## 问题与思考", "## 关键词", "## 人物简介"):
            assert heading in draft and PLACEHOLDER_MARK in draft
        assert "## 全文转录" in draft
        assert "[SPEAKER_00] 大家好，欢迎收听。" in draft

    def test_repeat_run_identical(self, tmp_path):
        """幂等：同输入重复生成结果完全一致。"""
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        first = build_draft(diarized, {})
        second = build_draft(diarized, {})
        assert first == second

    def test_speaker_map_applied(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        draft = build_draft(diarized, {"SPEAKER_00": "主播"})
        assert "**主播**：大家好，欢迎收听。" in draft


class TestAnalyzeOne:
    def test_reports_missing_md(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        r = analyze_one(diarized, md_path_for(diarized), fix_source=False)
        assert r["exists"] is False

    def test_detects_leftover_labels(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        md = md_path_for(diarized)
        md.write_text("# 标题\n\n## 全文转录\n\n[SPEAKER_00] 残留\n", encoding="utf-8")
        r = analyze_one(diarized, md, fix_source=False)
        assert r["labels"] == 1

    def test_detects_placeholders(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        md = md_path_for(diarized)
        md.write_text(
            "# 标题\n\n## 摘要\n\n（待校订：占位）\n\n## 全文转录\n\n内容\n",
            encoding="utf-8",
        )
        r = analyze_one(diarized, md, fix_source=False)
        assert "## 摘要" in r["placeholders"]

    def test_fix_source_writes_back(self, tmp_path):
        diarized = tmp_path / "节目_diarized.txt"
        diarized.write_text(SAMPLE_PACKAGE, encoding="utf-8")
        md = md_path_for(diarized)
        md.write_text("# 标题\n\n> 来源：播客 | 2026-01-01\n\n## 全文转录\n\n内容\n",
                      encoding="utf-8")
        r = analyze_one(diarized, md, fix_source=True)
        assert r.get("source_fixed") is True
        assert "播客  |  标题  |  2026-01-01" in md.read_text(encoding="utf-8")
