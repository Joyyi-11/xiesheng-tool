"""Tests for the idempotent result cache."""

from src.result_cache import (
    episode_key,
    find_cached_markdown,
    load_cache,
    record_result,
)


class TestEpisodeKey:
    def test_stable_and_consistent(self):
        assert episode_key("https://a/1", "2026-01-01") == episode_key("https://a/1", "2026-01-01")
        assert episode_key("https://a/1", "2026-01-01") != episode_key("https://a/2", "2026-01-01")
        assert len(episode_key("https://a/1", "2026-01-01")) == 16

    def test_url_alone_is_stable(self):
        assert episode_key("https://a/1") == episode_key("https://a/1", "")


class TestResultCache:
    def test_record_then_find(self, tmp_path):
        md = tmp_path / "out.md"
        md.write_text("# 内容", encoding="utf-8")
        record_result(tmp_path, "https://a/1", "2026-01-01", md, provider="qwen", model="m")
        found = find_cached_markdown(tmp_path, "https://a/1", "2026-01-01")
        assert found is not None
        assert found == md

    def test_different_episode_not_cached(self, tmp_path):
        md = tmp_path / "out.md"
        md.write_text("# 内容", encoding="utf-8")
        record_result(tmp_path, "https://a/1", "2026-01-01", md)
        assert find_cached_markdown(tmp_path, "https://a/999", "2026-01-01") is None

    def test_missing_markdown_returns_none(self, tmp_path):
        md = tmp_path / "out.md"  # 不存在
        record_result(tmp_path, "https://a/1", "2026-01-01", md)
        assert find_cached_markdown(tmp_path, "https://a/1", "2026-01-01") is None

    def test_pub_date_part_of_identity(self, tmp_path):
        md = tmp_path / "out.md"
        md.write_text("# 内容", encoding="utf-8")
        record_result(tmp_path, "https://a/1", "2026-01-01", md)
        assert find_cached_markdown(tmp_path, "https://a/1", "2026-05-05") is None

    def test_cache_survives_reload(self, tmp_path):
        md = tmp_path / "out.md"
        md.write_text("# 内容", encoding="utf-8")
        record_result(tmp_path, "https://a/1", "2026-01-01", md)
        cache = load_cache(tmp_path)
        assert len(cache) == 1
        assert cache[episode_key("https://a/1", "2026-01-01")]["markdown_path"] == str(md)

    def test_corrupt_cache_is_ignored(self, tmp_path):
        cache_path = tmp_path / ".work" / "result_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{{{ 不是json", encoding="utf-8")
        assert load_cache(tmp_path) == {}
