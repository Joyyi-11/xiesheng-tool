from src.audio import download_audio


def test_existing_readable_audio_is_reused(tmp_path, monkeypatch):
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"existing audio")
    monkeypatch.setattr("src.audio.get_duration_seconds", lambda path: 123.0)

    def fail_download(*args, **kwargs):
        raise AssertionError("download should be skipped")

    monkeypatch.setattr("src.audio.requests.get", fail_download)
    assert download_audio("https://example.com/episode.mp3", tmp_path, "episode") == audio


def test_corrupt_existing_audio_is_redownloaded(tmp_path, monkeypatch):
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"not audio")

    def unreadable_duration(path):
        raise RuntimeError("Invalid data")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            return iter([b"new audio"])

    monkeypatch.setattr("src.audio.get_duration_seconds", unreadable_duration)
    monkeypatch.setattr("src.audio.requests.get", lambda *args, **kwargs: FakeResponse())
    assert download_audio("https://example.com/episode.mp3", tmp_path, "episode") == audio
    assert audio.read_bytes() == b"new audio"
