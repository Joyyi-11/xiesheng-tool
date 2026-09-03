import argparse

import pytest

from src.config import LLMConfig
from src.main import main, process_episode
from src.models.schemas import EpisodeInfo, KeyPoint, OutputDoc, QuestionItem, TranscriptResult


def _episode():
    return EpisodeInfo(
        url="https://www.xiaoyuzhoufm.com/episode/test",
        title="测试节目",
        podcast_name="测试播客",
        pub_date="2026-01-01",
        show_notes="",
        audio_url="https://example.com/test.mp3",
    )


class FakeTranscriber:
    model_name = "sensevoice-small"

    def can_force_num_speakers(self, duration_sec):
        return False

    def transcribe(
        self, audio_path, duration_sec=None, *, work_dir=None, jobs=1, num_speakers=None,
        no_diarization=False, **kwargs,
    ):
        return TranscriptResult(
            raw_text="正文。",
            segments=[{"text": "正文。", "start_time": 0.0, "end_time": 1.0}],
            duration_sec=duration_sec,
        )

    def shutdown(self):
        return None


def test_api_degraded_output_is_not_cached_as_final(tmp_path, monkeypatch, capsys):
    from src.utils import CostTracker

    episode = _episode()
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"audio")
    doc = OutputDoc(
        title=episode.title,
        podcast_name=episode.podcast_name,
        pub_date=episode.pub_date,
        show_notes=episode.show_notes,
        key_points=[KeyPoint("要点。", "证据。")],
        full_text="正文。",
        summary="摘要。",
        questions=[QuestionItem("问题？", "回答。")],
        warnings=["分块 1 校订失败，已回退原稿，需人工复核"],
    )
    args = argparse.Namespace(
        refresh=True,
        no_diarization=True,
        llm_mode="api",
        speakers=None,
        jobs=1,
        model="sensevoice-small",
    )
    llm_config = LLMConfig(
        api_key="key", base_url="https://example.test/v1", provider="qwen", model="qwen3.7-plus"
    )

    monkeypatch.setattr("src.main.scrape_episode", lambda url: episode)
    monkeypatch.setattr("src.main.find_cached_markdown", lambda *args: None)
    monkeypatch.setattr("src.main.find_cached_handoff", lambda *args: None)
    monkeypatch.setattr("src.main.get_duration_seconds", lambda path: 1.0)
    monkeypatch.setattr(
        "src.main.llm_processor.process",
        lambda *args, **kwargs: (doc, 10, 20),
    )

    assert process_episode(
        episode.url,
        args,
        tmp_path,
        CostTracker(),
        FakeTranscriber(),
        audio,
        llm_config,
    )
    output = tmp_path / "测试节目.degraded.md"
    assert output.is_file()
    assert "> [WARN] 分块 1 校订失败" in output.read_text(encoding="utf-8")
    assert "[WARN] 已产出降级文稿" in capsys.readouterr().out
    assert not (tmp_path / ".work" / "result_cache.json").exists()


def test_server_mode_does_not_accept_episode_urls(monkeypatch):
    monkeypatch.setattr("sys.argv", ["xiesheng", "--server", "https://example.com/episode"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_non_server_mode_requires_url(monkeypatch):
    monkeypatch.setattr("sys.argv", ["xiesheng"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_server_mode_can_start_without_url(monkeypatch):
    calls = []
    monkeypatch.setattr("src.server.start_server", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("sys.argv", ["xiesheng", "--server", "--server-port", "8900"])
    main()
    assert calls == [{"port": 8900, "model_name": "sensevoice-small", "spk_max_seg_ms": 4000, "batch_size_s": 60}]
