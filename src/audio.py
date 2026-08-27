"""Audio download and conversion utilities."""

import logging
import subprocess
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 600  # 单段下载最长 10 分钟
CONNECT_TIMEOUT = 15

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def download_audio(url: str, output_dir: Path | None = None, stem: str | None = None) -> Path:
    """Download audio file from URL. Returns path to downloaded file.

    ``stem`` 必须显式传入（通常为节目标题 safe_filename），禁止默认成
    ``podcast_audio``——否则 output/ 会堆满无法辨认的音频。仅在未传入时
    用 ``podcast`` 占位并打警告。
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "xiesheng-audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not stem:
        logger.warning("download_audio 未传 stem（节目名），产出通用名 podcast 文件")
        stem = "podcast"
    ext = _guess_extension(url)
    dest = output_dir / f"{stem}{ext}"

    # 跳过已存在的音频：批量重跑时避免重复下载大文件（如 EP80 252MB）。
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("音频已存在，跳过下载：%s", dest)
        return dest

    logger.info("Downloading audio from %s ...", url)
    with requests.get(
        url,
        headers={"User-Agent": _UA},
        stream=True,
        timeout=(CONNECT_TIMEOUT, DOWNLOAD_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    logger.info("Downloaded to %s (%.1f MB)", dest, downloaded / 1_000_000)
    return dest


def convert_to_wav(input_path: Path, output_dir: Path | None = None, stem: str | None = None) -> Path:
    """Convert audio to WAV 16kHz mono using ffmpeg.

    ``stem`` 必须显式传入（通常为节目标题 safe_filename），见 download_audio。
    """
    if output_dir is None:
        output_dir = input_path.parent
    if not stem:
        logger.warning("convert_to_wav 未传 stem（节目名），产出通用名 podcast.wav")
        stem = "podcast"
    output_path = output_dir / f"{stem}.wav"

    logger.info("Converting to WAV 16kHz mono...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        check=True, capture_output=True, timeout=600,
    )
    logger.info("Converted to %s", output_path)
    return output_path


def get_duration_seconds(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _guess_extension(url: str) -> str:
    """Guess file extension from URL."""
    path = url.split("?")[0].rstrip("/")
    for ext in (".mp3", ".m4a", ".wav", ".ogg", ".aac"):
        if path.endswith(ext):
            return ext
    return ".mp3"  # default
