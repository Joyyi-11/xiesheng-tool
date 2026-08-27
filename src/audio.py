"""Audio download and conversion utilities."""

import logging
import subprocess
import tempfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 600  # 单段下载最长 10 分钟
CONNECT_TIMEOUT = 15
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_BASE_S = 1.0

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 瞬时网络错误：重试有意义；HTTP 4xx/5xx 与本地写盘错误不属于此类，直接抛出。
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.SSLError,
    requests.exceptions.ChunkedEncodingError,
)


def download_audio(url: str, output_dir: Path | None = None, stem: str | None = None) -> Path:
    """Download audio file from URL. Returns path to downloaded file.

    ``stem`` 必须显式传入（通常为节目标题 safe_filename），禁止默认成
    ``podcast_audio``——否则 output/ 会堆满无法辨认的音频。仅在未传入时
    用 ``podcast`` 占位并打警告。

    下载是原子的：先写 ``<dest>.part`` 临时文件，成功后 ``replace`` 到最终
    路径。这样中途失败（断连/超时）留下的半截文件不会在下次运行时被当作
    完整产物复用，避免把截断音频喂给转写。瞬时网络错误会指数退避重试。
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "xiesheng-audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not stem:
        logger.warning("download_audio 未传 stem（节目名），产出通用名 podcast 文件")
        stem = "podcast"
    ext = _guess_extension(url)
    dest = output_dir / f"{stem}{ext}"

    # 跳过已存在的完整音频：原子下载保证 dest 若非空即为完整成功产物。
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("音频已存在，跳过下载：%s", dest)
        return dest

    part = output_dir / f"{stem}{ext}.part"
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                url,
                headers={"User-Agent": _UA},
                stream=True,
                timeout=(CONNECT_TIMEOUT, DOWNLOAD_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                downloaded = 0
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
        except Exception as exc:
            # 任何失败都清掉半截文件，避免下次误当完整产物复用。
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, _TRANSIENT) and attempt < DOWNLOAD_RETRIES:
                last_exc = exc
                logger.warning(
                    "下载瞬时失败（第 %d/%d 次）：%s；%.1fs 后重试",
                    attempt, DOWNLOAD_RETRIES, exc,
                    DOWNLOAD_RETRY_BASE_S * (2 ** (attempt - 1)),
                )
                time.sleep(DOWNLOAD_RETRY_BASE_S * (2 ** (attempt - 1)))
                continue
            raise
        else:
            part.replace(dest)  # 原子落盘：只有完整下载才成为最终产物
            logger.info("Downloaded to %s (%.1f MB)", dest, downloaded / 1_000_000)
            return dest
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"下载失败：{url}")


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
