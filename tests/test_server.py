import pytest

from src.models.schemas import TranscriptResult
from src.server import HttpTranscriber, ModelMismatchError, validate_model_name


def test_validate_model_name_requires_model():
    with pytest.raises(ValueError, match="model_name is required"):
        validate_model_name("", "sensevoice-small")


def test_validate_model_name_rejects_mismatch():
    with pytest.raises(ModelMismatchError, match="requested paraformer-large"):
        validate_model_name("paraformer-large", "sensevoice-small")


def test_validate_model_name_accepts_match():
    validate_model_name("sensevoice-small", "sensevoice-small")


def test_http_client_sends_requested_model(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "raw_text": "正文",
                "segments": [{"text": "正文", "start_time": 0.0, "end_time": 1.0}],
                "duration_sec": 1.0,
                "cost_yuan": 0.0,
            }

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    client = HttpTranscriber(base_url="http://127.0.0.1:8765/", model_name="paraformer-large")
    result = client.transcribe(tmp_path / "a.wav", 1.0, work_dir=tmp_path, num_speakers=2)

    assert captured["url"] == "http://127.0.0.1:8765/transcribe"
    assert captured["json"]["model_name"] == "paraformer-large"
    assert captured["json"]["num_speakers"] == 2
    assert isinstance(result, TranscriptResult)


def test_http_client_ensures_loaded_model(monkeypatch):
    client = HttpTranscriber(model_name="sensevoice-small")
    monkeypatch.setattr(client, "health", lambda: {"status": "ok", "model": "sensevoice-small"})
    assert client.ensure_model() == {"status": "ok", "model": "sensevoice-small"}


def test_http_client_rejects_model_mismatch_before_transcription(monkeypatch):
    client = HttpTranscriber(model_name="sensevoice-small")
    monkeypatch.setattr(client, "health", lambda: {"status": "ok", "model": "paraformer-large"})
    with pytest.raises(ModelMismatchError):
        client.ensure_model()
