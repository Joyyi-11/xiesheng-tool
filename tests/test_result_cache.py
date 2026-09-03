"""Tests for the idempotent result cache."""

from src.result_cache import (
    episode_key,
    find_cached_handoff,
    find_cached_markdown,
    load_cache,
    record_handoff,
    record_result,
)


class TestEpisodeKey:
    def test_stable_and_consistent(self):
        assert episode_key("https://a/1", "2026-01-01") == episode_key("https://a/1", "2026-01-01")
        assert episode_key("https://a/1", "2026-01-01") != episode_key("https://a/2", "2026-01-01")
        assert len(episode_key("https://a/1", "2026-01-01")) == 16

    def test_url_alone_is_stable(self):
        assert episode_key("https://a/1") == episode_key("https://a/1", "")
        assert episode_key("https://a/1") == episode_key("https://a/1", stage="final")
        assert episode_key("https://a/1") != episode_key("https://a/1", stage="session_handoff")


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
        entry = cache[episode_key("https://a/1", "2026-01-01")]
        assert entry["path"] == "out.md"
        assert entry["stage"] == "final"

    def test_handoff_is_a_separate_stage_from_final(self, tmp_path):
        package = tmp_path / "out_diarized.txt"
        package.write_text("会话包", encoding="utf-8")
        record_handoff(tmp_path, "https://a/1", "2026-01-01", package)

        assert find_cached_handoff(tmp_path, "https://a/1", "2026-01-01") == package
        assert find_cached_markdown(tmp_path, "https://a/1", "2026-01-01") is None

    def test_final_cache_rejects_handoff_text_file(self, tmp_path):
        package = tmp_path / "out_diarized.txt"
        package.write_text("会话包", encoding="utf-8")
        try:
            record_result(tmp_path, "https://a/1", "2026-01-01", package)
        except ValueError as exc:
            assert ".md" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-Markdown final result")

    def test_corrupt_cache_is_ignored(self, tmp_path):
        cache_path = tmp_path / ".work" / "result_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{{{ 不是json", encoding="utf-8")
        assert load_cache(tmp_path) == {}

    def test_write_failure_degrades_to_noop(self, tmp_path):
        """B1：缓存写失败（如 Windows 下 tmp.replace 被占用）不得抛异常，
        否则会把已成功产出的整期误判为失败（vol.231 实测 WinError 5）。"""
        md = tmp_path / "out.md"
        md.write_text("# 内容", encoding="utf-8")
        # 让 .work 目录不可写：用文件占住目录名，save_cache 的 mkdir 会失败
        blocker = tmp_path / ".work"
        blocker.write_text("占用", encoding="utf-8")
        # 不应抛异常
        record_result(tmp_path, "https://a/1", "2026-01-01", md)
        # 缓存未写成，find 返回 None，但流程不中断
        assert find_cached_markdown(tmp_path, "https://a/1", "2026-01-01") is None
