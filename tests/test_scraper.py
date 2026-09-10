"""Tests for the scraper module."""

import pytest
from bs4 import BeautifulSoup

from src.scraper.xiaoyuzhou import _find_json_ld, parse_episode_id


class TestParseEpisodeId:
    def test_standard_url(self):
        eid = parse_episode_id("https://www.xiaoyuzhoufm.com/episode/60b030ef3104c523cd6d7eef")
        assert eid == "60b030ef3104c523cd6d7eef"

    def test_without_www(self):
        eid = parse_episode_id("https://xiaoyuzhoufm.com/episode/abc123")
        assert eid == "abc123"

    def test_invalid_url(self):
        with pytest.raises(ValueError):
            parse_episode_id("https://example.com/not-a-podcast")


def test_json_ld_episode_title_is_used():
    """回归：节目标题必须取 PodcastEpisode.name，不能沿用旧缓存标题。"""
    html = """
    <script type="application/ld+json">
    {"@type":"PodcastEpisode","name":"昆山杜克大学周忆粟：AI 来了，年轻人的梯子被抽掉了",
     "partOfSeries":{"name":"AI炼金术"}}
    </script>
    """
    data = _find_json_ld(BeautifulSoup(html, "html.parser"))
    assert data is not None
    assert data["name"] == "昆山杜克大学周忆粟：AI 来了，年轻人的梯子被抽掉了"
    assert data["partOfSeries"]["name"] == "AI炼金术"
