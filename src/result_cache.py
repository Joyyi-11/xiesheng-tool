"""Idempotent result cache: re-running the same episode reuses prior output.

Transcription and LLM post-processing are the expensive steps. If an episode was
already processed, we record its identity (URL hash + publish date + provider)
and the produced Markdown path. Re-running the same URL returns the existing
result instead of redoing everything, unless the caller forces a refresh.
"""

import hashlib
import logging
from pathlib import Path

from src.utils import read_json, write_json

logger = logging.getLogger(__name__)

CACHE_FILE = "result_cache.json"


def episode_key(url: str, pub_date: str = "", stage: str = "final") -> str:
    """Stable key for an episode processing stage."""
    raw = f"{url}|{pub_date}|{stage}".strip("|")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(output_dir: Path) -> Path:
    return output_dir / ".work" / CACHE_FILE


def load_cache(output_dir: Path) -> dict:
    path = _cache_path(output_dir)
    data = read_json(path)
    return data if isinstance(data, dict) else {}


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
        write_json(tmp, cache)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Result cache write failed, degraded to no-op: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _stored_path(output_dir: Path, path: Path) -> str:
    """Store outputs inside output_dir portably; keep external paths absolute."""
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _entry_path(output_dir: Path, entry: dict) -> Path:
    stored = str(entry.get("path", ""))
    path = Path(stored)
    return path if path.is_absolute() else output_dir / path


def find_cached_markdown(output_dir: Path, url: str, pub_date: str = "") -> Path | None:
    """Return the validated final .md path for this episode, if it exists."""
    key = episode_key(url, pub_date, stage="final")
    entry = load_cache(output_dir).get(key)
    if not entry:
        return None
    md_path = _entry_path(output_dir, entry)
    if md_path.suffix.lower() != ".md" or not md_path.is_file():
        return None
    return md_path


def record_handoff(
    output_dir: Path,
    url: str,
    pub_date: str,
    handoff_path: Path,
) -> None:
    """Remember a session handoff without presenting it as a final result."""
    key = episode_key(url, pub_date, stage="session_handoff")
    cache = load_cache(output_dir)
    cache[key] = {
        "url": url,
        "pub_date": pub_date,
        "path": _stored_path(output_dir, handoff_path),
        "stage": "session_handoff",
    }
    save_cache(output_dir, cache)
    logger.info("Session handoff cached: %s -> %s", url, handoff_path)


def find_cached_handoff(output_dir: Path, url: str, pub_date: str = "") -> Path | None:
    """Return an existing session handoff package, if it still exists."""
    key = episode_key(url, pub_date, stage="session_handoff")
    entry = load_cache(output_dir).get(key)
    if not entry:
        return None
    path = _entry_path(output_dir, entry)
    return path if path.is_file() else None


def record_result(
    output_dir: Path,
    url: str,
    pub_date: str,
    markdown_path: Path,
    provider: str = "",
    model: str = "",
) -> None:
    """Remember that a validated final Markdown result was produced."""
    if markdown_path.suffix.lower() != ".md":
        raise ValueError(f"Final result cache only accepts .md files: {markdown_path}")
    cache = load_cache(output_dir)
    cache[episode_key(url, pub_date, stage="final")] = {
        "url": url,
        "pub_date": pub_date,
        "path": _stored_path(output_dir, markdown_path),
        "stage": "final",
        "provider": provider,
        "model": model,
    }
    save_cache(output_dir, cache)
    logger.info("Result cached: %s -> %s", url, markdown_path)
