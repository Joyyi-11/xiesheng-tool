"""Tests for faithful transcript rebuild (rebuild_transcript)."""

import pytest

from src.processor.rebuild_transcript import (
    build_markdown,
    build_transcript,
    parse_package,
    parse_speaker_map,
    replace_transcript_section,
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


class TestParsePackage:
    def test_extracts_title_source_notes_and_segments(self):
        title, source_line, show_notes, transcript, segs = parse_package(SAMPLE_PACKAGE)
        assert title == "例话：AI 协作实践"
        assert source_line == "> 来源：小宇宙 | 例话 | 2026-08-01"
        assert show_notes == "这期聊人机协作。"
        assert "大家好，欢迎收听。" in transcript
        assert segs == [
            ("SPEAKER_00", "大家好，欢迎收听。"),
            ("SPEAKER_01", "今天聊 AI。"),
            ("SPEAKER_00", "我们开始吧。"),
        ]

    def test_no_transcript_returns_empty_segments(self):
        _, _, _, _, segs = parse_package("# 标题\n\n# Show Notes\n\n内容\n")
        assert segs == []


class TestBuildTranscript:
    def test_keeps_every_segment_in_order(self):
        segs = [
            ("SPEAKER_00", "第一段"),
            ("SPEAKER_01", "第二段"),
            ("SPEAKER_00", "第三段"),
        ]
        body = build_transcript(segs, {})
        assert "[SPEAKER_00] 第一段" in body
        assert "[SPEAKER_01] 第二段" in body
        assert "[SPEAKER_00] 第三段" in body

    def test_merges_consecutive_same_label(self):
        segs = [
            ("SPEAKER_00", "第一句"),
            ("SPEAKER_00", "第二句"),
            ("SPEAKER_01", "别人的"),
        ]
        body = build_transcript(segs, {})
        assert "[SPEAKER_00] 第一句第二句" in body
        assert body.count("SPEAKER_00") == 1

    def test_skips_empty_text_segments(self):
        body = build_transcript([("SPEAKER_00", ""), ("SPEAKER_01", "正文")], {})
        assert "[SPEAKER_00]" not in body
        assert "[SPEAKER_01] 正文" in body

    def test_speaker_map_renames_labels(self):
        body = build_transcript([("SPEAKER_00", "你好")], {"SPEAKER_00": "主播"})
        assert "【主播】你好" in body
        assert "[SPEAKER_00]" not in body

    def test_no_trailing_blank_line(self):
        body = build_transcript([("SPEAKER_00", "内容")], {})
        assert not body.endswith("\n")


class TestBuildMarkdown:
    def test_v5_shape(self):
        md = build_markdown("标题", "> 来源：播客 | 标题 | 2026-01-01", "简介", "正文")
        assert md.startswith("# 标题\n")
        assert "## 全文转录" in md
        assert "正文" in md
        assert md.endswith("\n")


class TestReplaceTranscriptSection:
    def test_replaces_from_transcript_heading_keeps_earlier_sections(self):
        old = "# 标题\n\n## 摘要\n\n已有摘要\n\n## 全文转录\n\n旧内容\n"
        new = replace_transcript_section(old, "新正文")
        assert "已有摘要" in new
        assert "旧内容" not in new
        assert "## 全文转录\n\n新正文" in new

    def test_raises_when_heading_missing(self):
        with pytest.raises(ValueError):
            replace_transcript_section("# 只有标题\n", "正文")


class TestParseSpeakerMap:
    def test_parses_pairs(self):
        assert parse_speaker_map("SPEAKER_00:主播 Jean,SPEAKER_01:嘉宾姨姨") == {
            "SPEAKER_00": "主播 Jean",
            "SPEAKER_01": "嘉宾姨姨",
        }

    def test_empty_returns_empty(self):
        assert parse_speaker_map("") == {}

    def test_raises_on_bad_label(self):
        with pytest.raises(ValueError):
            parse_speaker_map("HOST:主播")

    def test_raises_on_missing_name(self):
        with pytest.raises(ValueError):
            parse_speaker_map("SPEAKER_00:")
