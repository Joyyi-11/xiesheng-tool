"""Tests for the cost tracker utility."""

from src.utils import CostTracker, merge_same_speaker_blocks, safe_filename


class TestCostTracker:
    def test_empty(self):
        t = CostTracker()
        assert t.total_yuan == 0.0

    def test_transcription_cost(self):
        t = CostTracker()
        t.add_transcription(0.66)
        assert t.transcription_yuan == 0.66
        assert t.total_yuan == 0.66

    def test_llm_cost(self):
        t = CostTracker()
        t.add_llm_usage(20000, 30000)
        assert t.llm_input_tokens == 20000
        assert t.llm_output_tokens == 30000
        assert not t.llm_cost_known
        assert "供应商账单" in t.summary()

    def test_total(self):
        t = CostTracker()
        t.add_transcription(0.66)
        t.add_llm_usage(20000, 30000, cost_yuan=0.07)
        assert abs(t.total_yuan - 0.73) < 0.001

    def test_safe_filename(self):
        assert safe_filename('节目: "AI/未来"?') == "节目_ _AI_未来__"


class TestMergeSameSpeakerBlocks:
    def test_merges_adjacent_same_speaker(self):
        body = "【湫湫】第一句。\n\n【湫湫】第二句。\n\n【小朱】别人的。\n\n【湫湫】第三段。"
        out = merge_same_speaker_blocks(body)
        assert out.count("【湫湫】") == 2  # 两段湫湫（中间被小朱隔开）
        assert "【湫湫】第一句。第二句。" in out
        assert "【小朱】别人的。" in out

    def test_merges_lines_without_blank_separator(self):
        body = "【湫湫】第一句。\n【湫湫】第二句。\n【湫湫】第三句。"
        out = merge_same_speaker_blocks(body)
        assert out.count("【湫湫】") == 1
        assert out == "【湫湫】第一句。第二句。第三句。"

    def test_inserts_period_when_no_tail_punctuation(self):
        out = merge_same_speaker_blocks("【甲】没标点\n【甲】接着说。")
        assert out == "【甲】没标点。接着说。"

    def test_continuation_lines_join_previous(self):
        out = merge_same_speaker_blocks("【甲】开头。\n\n续行内容。")
        assert out == "【甲】开头。续行内容。"

    def test_empty_input_returns_as_is(self):
        assert merge_same_speaker_blocks("") == ""
        assert merge_same_speaker_blocks("   ") == "   "

    def test_horizontal_rule_not_merged_into_speech(self):
        out = merge_same_speaker_blocks("【甲】正文。\n\n---\n\n来源：某播客")
        assert "【甲】正文。" in out
        assert "---" in out
        assert "正文。---" not in out

    def test_alternating_speakers_untouched(self):
        body = "【甲】一。\n\n【乙】二。\n\n【甲】三。"
        assert merge_same_speaker_blocks(body) == body
