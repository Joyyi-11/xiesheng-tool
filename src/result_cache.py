"""Idempotent result cache: re-running the same episode reuses prior output.

Transcription and LLM post-processing are the expensive steps. If an episode was
already processed, we record its identity (URL hash + publish date + provider)
and the produced Markdown path. Re-running the same URL returns the existing
result instead of redoing everything, unless the caller forces a refresh.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILE = "result_cache.json"


def episode_key(url: str, pub_date: str = "") -> str:
    """Stable key for an episode identity."""
    raw = f"{url}|{pub_date}".strip("|")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(output_dir: Path) -> Path:
    return output_dir / ".work" / CACHE_FILE


def load_cache(output_dir: Path) -> dict:
    path = _cache_path(output_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Result cache unreadable, starting fresh: %s", path)
        return {}


def save_cache(output_dir: Path, cache: dict) -> None:
    """Persist the cache. Failures degrade to a no-op (never raise).

    Windows 下 ``tmp.replace`` 可能因目标被占用（如 Defender 实时扫描、
    其他进程并发写）抛 ``PermissionError``；此前该异常会向上传播，把
    已成功产出的整期误判为失败（vol.231 实测 WinError 5）。缓存只是幂等
    记录，写失败不应翻转转录结果——降级为 warning，下次运行可重建。
    """
    path = _cache_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Result cache write failed, degraded to no-op: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def find_cached_markdown(output_dir: Path, url: str, pub_date: str = "") -> Path | None:
    """Return the cached .md path for this episode, if it still exists."""
    key = episode_key(url, pub_date)
    entry = load_cache(output_dir).get(key)
    if not entry:
        return None
    md_path = Path(entry.get("markdown_path", ""))
    if not md_path.exists():
        return None
    return md_path


def record_result(
    output_dir: Path,
    url: str,
    pub_date: str,
    markdown_path: Path,
    provider: str = "",
    model: str = "",
) -> None:
    """Remember that this episode was successfully processed."""
    cache = load_cache(output_dir)
    cache[episode_key(url, pub_date)] = {
        "url": url,
        "pub_date": pub_date,
        "markdown_path": str(markdown_path),
        "provider": provider,
        "model": model,
    }
    save_cache(output_dir, cache)
    logger.info("Result cached: %s -> %s", url, markdown_path)
