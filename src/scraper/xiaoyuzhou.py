"""Scrape Xiaoyuzhou FM episode page for audio URL and metadata."""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from src.models.schemas import EpisodeInfo

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

EPISODE_URL_PATTERN = re.compile(r"https?://(?:www\.)?xiaoyuzhoufm\.com/episode/([a-zA-Z0-9]+)")


def _get_with_retry(url, headers=None, timeout=30, max_attempts=3, base_delay=1.0):
    """GET with exponential backoff for transient network errors.

    小宇宙偶发 SSL/连接瞬时错误（如 (generated) UNEXPECTED_EOF / 连接重置），
    重试即可，避免一次白跑整期。仅对网络层异常重试；HTTP 4xx/5xx 由调用方
    的 raise_for_status 处理（不重试，属页面结构问题而非瞬时故障）。
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.SSLError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "抓取瞬时失败（第 %d/%d 次）：%s；%.1fs 后重试",
                    attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error("抓取重试 %d 次仍失败: %s", max_attempts, exc)
    raise last_exc


def parse_episode_id(url: str) -> str:
    """Extract episode ID from a Xiaoyuzhou URL."""
    m = EPISODE_URL_PATTERN.search(url)
    if not m:
        raise ValueError(f"Not a valid Xiaoyuzhou episode URL: {url}")
    return m.group(1)


def scrape_episode(url: str) -> EpisodeInfo:
    """Scrape episode page for audio URL and metadata."""
    eid = parse_episode_id(url)
    logger.info("Scraping episode %s from %s", eid, url)

    resp = _get_with_retry(url, headers=HEADERS, timeout=30)
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Try JSON-LD first (richest data source) ---
    ld_data = _find_json_ld(soup)

    # --- audio URL ---
    audio_url = ""
    if ld_data:
        audio_url = _extract_audio_from_ld(ld_data)
    if not audio_url:
        audio_url = _get_meta_content(soup, "og:audio")
    if not audio_url:
        raise RuntimeError("Could not find audio URL in page. The page structure may have changed.")

    # --- title ---
    title = ""
    if ld_data:
        title = ld_data.get("name", "")
    if not title:
        title = _get_meta_content(soup, "og:title") or ""
    if not title:
        t = soup.find("title")
        if t:
            title = t.get_text(strip=True)

    # --- podcast name from JSON-LD partOfSeries ---
    podcast_name = ""
    if ld_data:
        series = ld_data.get("partOfSeries", {})
        podcast_name = series.get("name", "")
    if not podcast_name:
        podcast_name = _get_meta_content(soup, "og:site_name") or ""
    if not podcast_name:
        podcast_name = _guess_podcast_name(soup, title)

    # --- show notes: JSON-LD description is the full show notes ---
    show_notes = ""
    if ld_data:
        show_notes = ld_data.get("description", "")
    if not show_notes:
        show_notes = _get_meta_content(soup, "og:description") or ""
    if not show_notes:
        sn = soup.find(class_=re.compile(r"show.?notes|description|content", re.I))
        if sn:
            show_notes = sn.get_text(strip=True)

    # --- pub date ---
    pub_date = ""
    if ld_data:
        pub_date = _extract_date_from_ld(ld_data) or ""
    if not pub_date:
        pub_date = _find_pub_date_from_json_ld(soup) or ""

    # --- cover image ---
    cover_url = _get_meta_content(soup, "og:image")

    logger.info("Found: [%s] %s (podcast: %s)", eid, title, podcast_name)
    return EpisodeInfo(
        url=url,
        title=title,
        podcast_name=podcast_name,
        pub_date=pub_date,
        show_notes=show_notes,
        audio_url=audio_url,
        cover_url=cover_url,
    )


def _find_json_ld(soup: BeautifulSoup) -> dict | None:
    """Find the first JSON-LD script block with PodcastEpisode type."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "PodcastEpisode":
                        return item
            elif data.get("@type") == "PodcastEpisode":
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _get_meta_content(soup: BeautifulSoup, property_name: str) -> str | None:
    """Extract content from a meta tag by property or name."""
    for attr in ("property", "name"):
        tag = soup.find("meta", attrs={attr: property_name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _extract_audio_from_ld(data: dict) -> str | None:
    """Walk JSON-LD structure looking for audio URLs."""
    # Check associatedMedia.contentUrl (PodcastEpisode pattern)
    media = data.get("associatedMedia")
    if isinstance(media, dict):
        url = media.get("contentUrl") or media.get("url") or ""
        if isinstance(url, str) and _looks_like_audio_url(url):
            return url
    # Check direct keys
    for key in ("audio", "contentUrl", "url"):
        val = data.get(key)
        if isinstance(val, str) and _looks_like_audio_url(val):
            return val
    # Recurse into @graph
    for item in data.get("@graph", []):
        url = _extract_audio_from_ld(item)
        if url:
            return url
    return None


def _looks_like_audio_url(url: str) -> bool:
    """Rough check if a URL points to an audio file or CDN."""
    return bool(re.search(r"\.(mp3|m4a|wav|ogg)", url, re.I)) or "media.xyzcdn.net" in url


def _find_pub_date_from_json_ld(soup: BeautifulSoup) -> str | None:
    """Find publication date from JSON-LD."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    d = _extract_date_from_ld(item)
                    if d:
                        return d
            else:
                d = _extract_date_from_ld(data)
                if d:
                    return d
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _extract_date_from_ld(data: dict) -> str | None:
    """Extract pub date from JSON-LD.

    仅对带时区的 aware 时间做 UTC→中国时间（UTC+8）转换；naive 时间来源未给时区，
    不臆测成 UTC 后再按本地时区偏移（否则非中国时区/naive 日期可能被移动一天），
    原样返回日期。
    """
    for key in ("datePublished", "dateCreated", "pubDate"):
        val = data.get(key)
        if not val:
            continue
        try:
            dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except ValueError:
            return str(val)[:10]
        if dt.tzinfo is None:
            # naive：来源未标注时区，原样返回，不偏移。
            return str(val)[:10]
        china_tz = timezone(timedelta(hours=8))
        return dt.astimezone(china_tz).strftime("%Y-%m-%d")
    for item in data.get("@graph", []):
        d = _extract_date_from_ld(item)
        if d:
            return d
    return None


def _guess_podcast_name(soup: BeautifulSoup, title: str) -> str:
    """Guess podcast name from title (format often: 'ep_title - podcast_name')."""
    if " - " in title:
        parts = title.split(" - ", 1)
        return parts[-1].strip()
    # try site name
    site = _get_meta_content(soup, "og:site_name")
    if site:
        return site
    return ""
