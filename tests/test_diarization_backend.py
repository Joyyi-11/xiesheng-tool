"""WeSpeaker 可选 diarize 后端的路由与透传单测。

覆盖 docs/wespeaker-diarize-backend.md §4：
- run_diarization 把 num_speakers 透传给 diarize()；
- diarize 缺失时给出清晰报错（而非静默 UNKNOWN）；
- main.py Step 4：campplus 走 spk 标签跳过外部 diarize；
  wespeaker 强制走外部 diarize（忽略 has_spk）；缺失依赖抛清晰错误。
"""
import argparse
import logging
import sys

import pytest

from src.config import LLMConfig
from src.main import process_episode
from src.models.schemas import EpisodeInfo, TranscriptResult
from src.utils import CostTracker


# ---------------------------------------------------------------------------
# run_diarization 单测（mock diarize 包，无需真实模型）
# ---------------------------------------------------------------------------

class _FakeSeg:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker


class _FakeResult:
    num_speakers = 2
    segments = [
        _FakeSeg(0.0, 1.0, "SPEAKER_00"),
        _FakeSeg(1.0, 2.0, "SPEAKER_01"),
    ]


def _make_fake_diarize(captured):
    def fake_diarize(audio_path, *, num_speakers=None, **kwargs):
        captured["num_speakers"] = num_speakers
        captured["audio"] = audio_path
        return _FakeResult()
    return fake_diarize


def test_run_diarization_passes_num_speakers(monkeypatch):
    from src.diarization import speaker_diarization as sd

    captured = {}
    monkeypatch.setattr(sd, "_get_diarize", lambda: _make_fake_diarize(captured))
    segs = sd.run_diarization("dummy.wav", num_speakers=2)
    assert captured["num_speakers"] == 2
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[1]["speaker"] == "SPEAKER_01"
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.0


def test_run_diarization_import_error_clear_message(monkeypatch):
    from src.diarization import speaker_diarization as sd

    # 让真正的 `from diarize import diarize` 失败（而非 mock 掉 _get_diarize 本身），
    # 才能验证 _get_diarize 把 ImportError 包装成清晰 RuntimeError 的链路。
    monkeypatch.setitem(sys.modules, "diarize", None)
    with pytest.raises(RuntimeError) as exc:
        sd.run_diarization("dummy.wav", num_speakers=2)
    assert "pip install 'xiesheng[diarize]'" in str(exc.value)


# ---------------------------------------------------------------------------
# main.py Step 4 路由单测（mock transcriber 与 diarize，复用会话链路提前返回）
# ---------------------------------------------------------------------------

def _episode():
    return EpisodeInfo(
        url="https://www.xiaoyuzhoufm.com/episode/test",
        title="测试节目",
        podcast_name="测试播客",
        pub_date="2026-01-01",
        show_notes="",
        audio_url="https://example.com/test.mp3",
    )


class _FakeTranscriberWithSpeaker:
    """转写结果自带 spk 标签（has_spk=True），用于验证 wespeaker 是否忽略它。"""
    model_name = "sensevoice-small"

    def can_force_num_speakers(self, duration_sec):
        return False

    def transcribe(self, audio_path, duration_sec=None, *, work_dir=None, jobs=1,
                   num_speakers=None, no_diarization=False, **kwargs):
        return TranscriptResult(
            raw_text="正文。",
            segments=[{"text": "正文。", "start_time": 0.0, "end_time": 1.0, "speaker": "SPEAKER_00"}],
            duration_sec=duration_sec,
        )

    def shutdown(self):
        return None


def _run_pipeline(diarization, monkeypatch, tmp_path):
    """驱动 process_episode 走完 Step 1-4（session 模式提前返回），返回外部 diarize 调用次数。"""
    episode = _episode()
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")

    calls = {"run": 0, "assign": 0}

    def fake_run_diarization(audio_path, num_speakers=None):
        calls["run"] += 1
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]

    def fake_assign_speakers(asr_segments, diarization_segments):
        calls["assign"] += 1
        return [{"speaker": "SPEAKER_00", "text": "正文。", "start": 0.0, "end": 1.0}]

    from src.diarization import speaker_diarization as sd
    monkeypatch.setattr(sd, "run_diarization", fake_run_diarization)
    monkeypatch.setattr(sd, "assign_speakers", fake_assign_speakers)

    monkeypatch.setattr("src.main.scrape_episode", lambda url: episode)
    monkeypatch.setattr("src.main.find_cached_markdown", lambda *a: None)
    monkeypatch.setattr("src.main.find_cached_handoff", lambda *a: None)
    monkeypatch.setattr("src.main.get_duration_seconds", lambda path: 100.0)

    args = argparse.Namespace(
        refresh=True,
        no_diarization=False,
        llm_mode="session",
        speakers=None,
        jobs=1,
        model="sensevoice-small",
        diarization=diarization,
    )
    llm_config = LLMConfig(
        api_key="key", base_url="https://example.test/v1", provider="qwen", model="qwen3.7-plus"
    )
    process_episode(
        episode.url, args, tmp_path, CostTracker(),
        _FakeTranscriberWithSpeaker(), audio, llm_config,
    )
    return calls


def test_campplus_uses_spk_labels_skips_external(monkeypatch, tmp_path):
    calls = _run_pipeline("campplus", monkeypatch, tmp_path)
    # has_spk=True → campplus 直接用转写自带标签，不应调外部 diarize
    assert calls["run"] == 0
    assert calls["assign"] == 0


def test_wespeaker_forces_external_diarize(monkeypatch, tmp_path):
    calls = _run_pipeline("wespeaker", monkeypatch, tmp_path)
    # wespeaker 忽略 has_spk，强制走外部 run_diarization + assign_speakers
    assert calls["run"] == 1
    assert calls["assign"] == 1


def test_wespeaker_missing_diarize_raises_clear_error(monkeypatch, tmp_path, caplog):
    from src.diarization import speaker_diarization as sd
    monkeypatch.setattr(sd, "run_diarization", lambda *a, **k: [])
    monkeypatch.setattr(sd, "assign_speakers", lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "diarize", None)  # 模拟未安装

    episode = _episode()
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr("src.main.scrape_episode", lambda url: episode)
    monkeypatch.setattr("src.main.find_cached_markdown", lambda *a: None)
    monkeypatch.setattr("src.main.find_cached_handoff", lambda *a: None)
    monkeypatch.setattr("src.main.get_duration_seconds", lambda path: 100.0)

    args = argparse.Namespace(
        refresh=True, no_diarization=False, llm_mode="session", speakers=None,
        jobs=1, model="sensevoice-small", diarization="wespeaker",
    )
    llm_config = LLMConfig(
        api_key="key", base_url="https://example.test/v1", provider="qwen", model="qwen3.7-plus"
    )
    # process_episode 设计为优雅降级（批处理不中断）：缺失依赖时跳过该集并返回 False，
    # 但必须把清晰报错（含安装提示）打到日志。断言该行为而非要求抛异常。
    with caplog.at_level(logging.ERROR, logger="xiesheng"):
        result = process_episode(
            episode.url, args, tmp_path, CostTracker(),
            _FakeTranscriberWithSpeaker(), audio, llm_config,
        )
    assert result is False
    assert "pip install 'xiesheng[diarize]'" in caplog.text
