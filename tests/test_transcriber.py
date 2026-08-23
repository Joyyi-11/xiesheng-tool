"""Tests for the FunASR transcriber (tag stripping, cache, segment building).

Model loading is exercised lazily and skipped here; these tests cover the
pure helpers and the whole-audio cache so they run without downloading models.
"""

from src.transcriber.funasr_transcriber import (
    FunASRTranscriber,
    _strip_sensevoice_tags,
)


class TestStripSenseVoiceTags:
    def test_removes_ascii_language_tags(self):
        assert _strip_sensevoice_tags("<|zh|><|NEUTRAL|>大家好") == "大家好"

    def test_removes_chinese_angle_emoji_tags(self):
        assert _strip_sensevoice_tags("今天天气好〈HAPPY〉我们出去") == "今天天气好我们出去"

    def test_removes_closing_emoji_tag(self):
        assert _strip_sensevoice_tags("段落结束〈/emoji〉后续") == "段落结束后续"

    def test_normalizes_whitespace(self):
        assert _strip_sensevoice_tags("  多个   空格  \n 这里  ") == "多个 空格 这里"

    def test_passthrough_clean_text(self):
        assert _strip_sensevoice_tags("普通中文文本，没有标签。") == "普通中文文本，没有标签。"


class TestModelValidation:
    def test_rejects_unknown_model(self):
        try:
            FunASRTranscriber(model_name="whisper-medium")
        except ValueError as exc:
            assert "whisper-medium" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown model")

    def test_accepts_supported_models(self):
        assert FunASRTranscriber(model_name="sensevoice-small").model_name == "sensevoice-small"
        assert FunASRTranscriber(model_name="paraformer-large").model_name == "paraformer-large"


class TestWholeAudioCache:
    def test_save_then_load_roundtrip(self, tmp_path):
        tr = FunASRTranscriber(model_name="sensevoice-small")
        audio = tmp_path / "ep.wav"
        audio.write_bytes(b"x" * 2048)
        work = tmp_path / "work"

        result = {"raw_text": "测试文本", "segments": [{"text": "测试文本", "start_time": 0.0, "end_time": 1.0}], "duration_sec": 1.0}
        tr._save_cache(work, audio, "sensevoice-small", result)
        loaded = tr._load_cache(work, audio, "sensevoice-small")
        assert loaded == result

    def test_cache_invalidated_on_model_change(self, tmp_path):
        tr = FunASRTranscriber(model_name="sensevoice-small")
        audio = tmp_path / "ep.wav"
        audio.write_bytes(b"x" * 2048)
        work = tmp_path / "work"
        tr._save_cache(work, audio, "sensevoice-small", {"raw_text": "a", "segments": [], "duration_sec": 0.0})
        # 换模型名读取应视为失效
        assert tr._load_cache(work, audio, "paraformer-large") is None

    def test_cache_invalidated_on_spk_max_seg_change(self, tmp_path):
        """段粒度参数纳入缓存指纹：改 spk_max_seg_ms 后旧段结果必须作废，
        否则重跑会静默复用旧粒度的说话人标签（Vol.11 源头修复不生效）。"""
        audio = tmp_path / "ep.wav"
        audio.write_bytes(b"x" * 2048)
        work = tmp_path / "work"
        tr8k = FunASRTranscriber(model_name="sensevoice-small", spk_max_seg_ms=8000)
        tr8k._save_cache(work, audio, "sensevoice-small", {"raw_text": "a", "segments": [], "duration_sec": 0.0})
        # 同参数读取应命中
        assert tr8k._load_cache(work, audio, "sensevoice-small") is not None
        # 换段粒度读取应视为失效
        tr4k = FunASRTranscriber(model_name="sensevoice-small", spk_max_seg_ms=4000)
        assert tr4k._load_cache(work, audio, "sensevoice-small") is None


class TestBuildSegments:
    def test_builds_from_sentence_info(self):
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None  # 跳过真实 ct-punc，避免加载模型
        res = [{
            "sentence_info": [
                {"start": 0.0, "end": 1.5, "text": "<|zh|>第一句〈HAPPY〉"},
                {"start": 1.5, "end": 3.0, "text": "第二句"},
            ]
        }]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=True)
        assert [s["text"] for s in segs] == ["第一句", "第二句"]
        assert segs[0]["start_time"] == 0.0 and segs[1]["end_time"] == 3.0

    def test_fallback_to_single_segment_when_no_timestamps(self):
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None
        res = [{"text": "<|zh|>整段文本〈NEUTRAL〉"}]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)
        assert len(segs) == 1
        assert segs[0]["text"] == "整段文本"

    def test_spk_model_sentence_info_with_words_timestamp(self):
        """spk_model 模式：sentence_info 无 text，按词级 timestamp 归组并带 speaker。"""
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None
        res = [{
            "sentence_info": [
                {"start": 0, "end": 3000, "text": None, "spk": 1},
                {"start": 3000, "end": 6000, "text": None, "spk": 2},
            ],
            "words": ["你", "好", "世", "界"],
            "timestamp": [[100, 800], [900, 1600], [3100, 3800], [3900, 4600]],
        }]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPEAKER_01"
        assert segs[0]["text"] == "你好"
        assert segs[0]["start_time"] == 0.0 and segs[0]["end_time"] == 3.0
        assert segs[1]["speaker"] == "SPEAKER_02"
        assert segs[1]["text"] == "世界"

    def test_spk_model_merges_adjacent_same_speaker(self):
        """相邻同 speaker 段按 0.2s 间隔合并为一段（gap 收紧后）。"""
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None
        res = [{
            "sentence_info": [
                {"start": 0, "end": 1000, "text": None, "spk": 0},
                {"start": 1150, "end": 2000, "text": None, "spk": 0},
                {"start": 2400, "end": 3000, "text": None, "spk": 1},
            ],
            "words": ["一", "二", "三", "四", "五", "六"],
            "timestamp": [[100, 300], [400, 600], [1200, 1400], [1500, 1700], [2500, 2700], [2800, 2900]],
        }]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPEAKER_00"
        assert segs[0]["text"] == "一二 三四"
        assert segs[1]["speaker"] == "SPEAKER_01"
        assert segs[1]["text"] == "五六"

    def test_spk_model_gap_0_2_does_not_merge_400ms(self):
        """gap 收紧到 0.2s：间隔 400ms 的同 speaker 段不再合并（0.5s 旧口径会合并）。"""
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None
        res = [{
            "sentence_info": [
                {"start": 0, "end": 1000, "text": None, "spk": 0},
                {"start": 1400, "end": 2000, "text": None, "spk": 0},
            ],
            "words": ["一", "二", "三", "四"],
            "timestamp": [[100, 300], [400, 600], [1500, 1700], [1800, 1900]],
        }]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)
        assert len(segs) == 2
        assert segs[0]["text"] == "一二"
        assert segs[1]["text"] == "三四"

    def test_spk_model_never_merges_across_speakers(self):
        """跨说话人绝不合并：即使间隔小于 gap 也不拼接（Vol.11 错乱根因防护）。"""
        tr = FunASRTranscriber(model_name="sensevoice-small")
        tr._punc_model = None
        res = [{
            "sentence_info": [
                {"start": 0, "end": 1000, "text": None, "spk": 0},
                {"start": 1050, "end": 2000, "text": None, "spk": 1},
            ],
            "words": ["一", "二", "三", "四"],
            "timestamp": [[100, 300], [400, 600], [1100, 1300], [1400, 1600]],
        }]
        segs = tr._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "SPEAKER_00"
        assert segs[1]["speaker"] == "SPEAKER_01"
