r"""Quick re-run from existing audio (thin wrapper over the main pipeline).

DEPRECATED: 完全委托 ``src.main`` 的 ``--audio`` 参数，直接使用

    xiesheng <url> --audio <existing.wav> [extra flags...]

即可达到同等效果；本入口仅保留兼容，不再新增功能。

This exists for convenience during development/iteration: given an episode URL
and an already-converted 16k mono WAV, it re-runs the whole ``xiesheng``
pipeline while skipping download and re-conversion.

It now delegates entirely to ``src.main`` with ``--audio``, so the two entry
points never drift. Anything this script could express is covered by:

    xiesheng <url> --audio <existing.wav> [extra flags...]

Usage:
    python -m src.quick_reprocess <xiaoyuzhou-url> --audio <existing.wav>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Windows terminal encoding fix
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick-reprocess")

OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEFAULT_AUDIO_WAV = OUTPUT_DIR / "podcast_audio.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已有音频重新转录和整理")
    parser.add_argument("url", help="小宇宙播客单集链接")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_WAV, help="已有 WAV 音频路径")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--llm-provider", default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--llm-model", help="覆盖提供方的默认模型")
    parser.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 4) // 4)), help="并行转录进程数")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"错误：找不到音频文件 {args.audio}", file=sys.stderr)
        sys.exit(1)

    # 复用主流程，避免两个入口逻辑漂移
    from src.main import main as xiesheng_main

    logger.info("委托主流程 xiesheng 处理（--audio 模式）...")
    sys.argv = [
        "xiesheng",
        args.url,
        "--audio", str(args.audio),
        "--model", args.model,
        "--llm-provider", args.llm_provider,
    ]
    if args.llm_model:
        sys.argv += ["--llm-model", args.llm_model]
    sys.argv += ["--jobs", str(args.jobs)]
    xiesheng_main()


if __name__ == "__main__":
    main()
