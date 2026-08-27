"""撷声 CLI.

Usage:
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx -o output/
    xiesheng url1 url2 url3 -o output/   # 多期：模型只加载一次，逐期转录
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

from src.audio import convert_to_wav, download_audio, get_duration_seconds
from src.config import DEFAULT_LLM_PROVIDER, LLM_MODELS, get_llm_config
from src.models.schemas import EpisodeInfo
from src.processor import llm_processor
from src.processor.normalize import normalize_quotes
from src.renderer.markdown import build_output_markdown
from src.result_cache import find_cached_markdown, record_result
from src.scraper.xiaoyuzhou import scrape_episode
from src.utils import CostTracker, Timer, fmt_time, safe_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xiesheng")


def build_session_doc(episode: EpisodeInfo, transcript_text: str) -> str:
    """Build a self-contained, speaker-labeled transcript for in-session post-processing.

    The --no-llm path produces this instead of the API path. It embeds episode
    metadata and Show Notes so the whole file can be pasted directly into an AI
    chat session for cleaning and structuring (speaker->name mapping, key points...).
    """
    lines = [
        f"# {episode.title}",
        "",
        f"> 来源：{episode.podcast_name}  |  {episode.pub_date}",
        "",
        "# Show Notes",
        "",
        episode.show_notes or "（无）",
        "",
        "# 转录全文（带说话人标签）",
        "",
        transcript_text,
    ]
    return "\n".join(lines)


def process_episode(url, args, output_dir, tracker, transcriber, audio_override, llm_config) -> bool:
    """处理单个单集：抓取→下载→转录→整理。

    transcriber 在批处理中复用同一实例（模型只加载一次）。audio_override 仅单
    链接时有效（音频与单集一一对应）。返回 True 表示该期已处理（含降级到会话内
    链路），False 表示该期失败（批处理会继续下一期，不中断整体）。
    """
    timers: dict[str, float] = {}
    try:
        # --- Step 1: Scrape ---
        with Timer() as t:
            logger.info("Step 1/6: 爬取节目信息...")
            episode = scrape_episode(url)
            print(f"  → {episode.title}")
            print(f"  播客: {episode.podcast_name}")
        timers["scrape"] = t.elapsed

        # --- Cache short-circuit: same episode already processed ---
        if not args.refresh:
            cached_md = find_cached_markdown(output_dir, url, episode.pub_date)
            if cached_md is not None:
                print(f"\n[缓存命中] 该单集已处理过，直接返回既有结果: {cached_md}")
                print("如需重新转录/整理，请加 --refresh")
                return True

        # --- Step 2: Download audio (or reuse existing WAV) ---
        safe_name = safe_filename(episode.title)
        with Timer() as t:
            if audio_override is not None:
                if not audio_override.exists():
                    raise FileNotFoundError(f"找不到音频文件: {audio_override}")
                logger.info("Step 2/6: 复用已有音频 %s", audio_override)
                wav_file = audio_override
            else:
                logger.info("Step 2/6: 下载音频...")
                audio_file = download_audio(episode.audio_url, output_dir, stem=safe_name)
                wav_file = convert_to_wav(audio_file, output_dir, stem=safe_name)
            duration_sec = get_duration_seconds(wav_file)
            print(f"  → 音频时长: {fmt_time(duration_sec)}")
        timers["download"] = t.elapsed

        # --- Step 3: Transcribe (复用 transcriber，模型只加载一次) ---
        with Timer() as t:
            logger.info("Step 3/6: 本地转录中（FunASR %s, CPU）...", args.model)
            transcript = transcriber.transcribe(
                wav_file,
                duration_sec,
                work_dir=output_dir / ".work" / safe_name,
                jobs=args.jobs,
            )
            tracker.add_transcription(transcript.cost_yuan)
        timers["transcribe"] = t.elapsed
        char_count = len(transcript.raw_text)
        rtf = t.elapsed / duration_sec if duration_sec else 0
        est_2h = int(7200 * rtf)
        print(f"  → 转录完成：{char_count} 字, {fmt_time(t.elapsed)}, RTF={rtf:.2f}")
        print(f"  → [估算] 2小时节目约需 {fmt_time(est_2h)}（当前模型: {args.model}）")

        raw_path = output_dir / f"{safe_name}_raw.txt"
        raw_path.write_text(transcript.raw_text, encoding="utf-8")
        logger.info("原始转录已保存到 %s", raw_path)

        # --- Step 4: Speaker Diarization (本地，独立于 LLM) ---
        # SenseVoice 转写自带说话人标签（spk_model 一站式输出，B 方案）时直接使用，
        # 无需再跑独立 diarize；Paraformer 路径（无 spk 标签）才回退到独立 diarize。
        transcript_text = transcript.raw_text
        labeled_segments: list[dict] | None = None
        labeled_path = None
        if not args.no_diarization:
            with Timer() as t:
                from src.diarization.speaker_diarization import format_labeled_segments

                has_spk = any(seg.get("speaker") for seg in transcript.segments)
                if has_spk:
                    logger.info("Step 4/6: 使用转写自带说话人标签（spk_model）...")
                    labeled_segments = transcript.segments
                else:
                    from src.diarization.speaker_diarization import (
                        assign_speakers,
                        run_diarization,
                    )

                    logger.info("Step 4/6: 说话人日志（Speaker Diarization，独立模块）...")
                    diarization_segments = run_diarization(wav_file, num_speakers=args.speakers)
                    labeled_segments = assign_speakers(transcript.segments, diarization_segments)
                try:
                    transcript_text = format_labeled_segments(labeled_segments)
                    labeled_path = output_dir / f"{safe_name}_diarized.txt"
                    labeled_path.write_text(
                        build_session_doc(episode, normalize_quotes(transcript_text)),
                        encoding="utf-8",
                    )
                    logger.info("带说话人标签的转录已保存到 %s", labeled_path)
                except Exception as e:
                    logger.warning("说话人标签处理失败，跳过: %s", e)
                    transcript_text = transcript.raw_text
                    labeled_segments = None
            timers["diarization"] = t.elapsed
        else:
            logger.info("已跳过说话人分离（--no-diarization）")

        if args.no_llm:
            # 无 LLM 链路：除会话输入包外，一并产出固定的会话内校订规范提示词，
            # 使"会话内校订"可直接复制执行，规则保持一致（与自动化链路同口径）。
            print(f"\n原始转录已保存到: {raw_path}")
            if labeled_path is not None:
                print(f"会话输入包（含 Show Notes 与说话人标签）已保存到: {labeled_path}")
                from src.processor.session_edit import (
                    build_session_prompt,
                    SESSION_SPEC_VERSION,
                )

                prompt_path = output_dir / f"{safe_name}_session_prompt.txt"
                prompt_path.write_text(
                    build_session_prompt(labeled_path.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )
                print(f"会话内校订提示词（固定规范 v{SESSION_SPEC_VERSION}）已保存到: {prompt_path}")
                print("（未启用 LLM；将提示词与包内容一起粘贴到 AI 会话，即可按固定规范完成校订与结构化）")
                # 会话包也算一次完整结果：缓存命中后重跑同样直接返回，避免重复转录
                record_result(
                    output_dir,
                    url,
                    episode.pub_date,
                    labeled_path,
                    provider="",
                    model="",
                )
            else:
                print("（未启用 LLM 且未做说话人分离；如需说话人标签，请提供 HF_TOKEN 后重跑）")
        else:
            # --- Step 5: LLM Process ---
            with Timer() as t:
                # 先探测网关可用模型并固定，避免 404 白白消耗配额；失败则降级为会话内链路
                try:
                    resolved_llm = llm_processor.resolve_llm_config(llm_config)
                except Exception as exc:
                    logger.warning("LLM 模型探测失败，降级为会话内链路: %s", exc)
                    print(
                        f"\n[降级] 无法确定可用的 LLM 模型（{exc}），"
                        "本次改走会话内处理链路。",
                        file=sys.stderr,
                    )
                    print(f"原始转录已保存到: {raw_path}")
                    if labeled_path is not None:
                        print(f"会话输入包（含 Show Notes 与说话人标签）已保存到: {labeled_path}")
                        record_result(
                            output_dir,
                            url,
                            episode.pub_date,
                            labeled_path,
                            provider="",
                            model="",
                        )
                    return True
                logger.info("Step 5/6: %s/%s 分块校订与整理中...", resolved_llm.provider, resolved_llm.model)
                try:
                    doc, inp_tok, out_tok = llm_processor.process(
                        episode.title,
                        episode.podcast_name,
                        episode.pub_date,
                        episode.show_notes,
                        transcript_text,
                        llm_config=resolved_llm,
                        work_dir=output_dir / ".work" / safe_name,
                        segments=labeled_segments,
                    )
                except Exception as exc:
                    logger.warning("LLM 后处理失败，降级为会话内链路: %s", exc)
                    print(
                        f"\n[降级] LLM 后处理失败（{exc}），本次改走会话内处理链路。",
                        file=sys.stderr,
                    )
                    print(f"原始转录已保存到: {raw_path}")
                    if labeled_path is not None:
                        print(f"会话输入包（含 Show Notes 与说话人标签）已保存到: {labeled_path}")
                        record_result(
                            output_dir,
                            url,
                            episode.pub_date,
                            labeled_path,
                            provider="",
                            model="",
                        )
                    return True
                tracker.add_llm_usage(inp_tok, out_tok)
            timers["process"] = t.elapsed

            # --- Step 6: Write output ---
            with Timer() as t:
                logger.info("Step 6/6: 写入 Markdown...")
                doc.costs = {
                    "transcription": tracker.transcription_yuan,
                    "llm": tracker.llm_cost_yuan if tracker.llm_cost_known else None,
                }
                doc.timings = timers
                md_content = build_output_markdown(doc)
                output_path = output_dir / f"{safe_name}.md"
                output_path.write_text(md_content, encoding="utf-8")
                record_result(
                    output_dir,
                    url,
                    episode.pub_date,
                    output_path,
                    provider=resolved_llm.provider,
                    model=resolved_llm.model,
                )
            timers["write"] = t.elapsed

            total_time = sum(timers.values())
            print(f"\n{'='*50}")
            print("[OK] 完成！")
            print(f"  输出文件: {output_path}")
            print(f"  总耗时: {fmt_time(total_time)}")
            print(f"  费用: {tracker.summary()}")
            print(f"  要点数: {len(doc.key_points)}")
            print(f"  闪光语句: {len(doc.highlight_quotes)}")
            print(f"{'='*50}")

        return True

    except Exception as e:
        logger.exception("单集处理失败: %s", url)
        print(f"\n错误（跳过该单集）: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="撷声 - 输入小宇宙播客单集链接，输出结构化 Markdown 文稿"
    )
    parser.add_argument("urls", nargs="*", help="小宇宙播客单集链接（可传多个；模型只加载一次，逐期转录；--server 模式可省略）")
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认: output/)")
    parser.add_argument("--model", default="sensevoice-small", choices=["sensevoice-small", "paraformer-large"],
                        help="ASR 模型（默认 sensevoice-small；paraformer-large 支持热词与字级时间戳）")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=sorted(LLM_MODELS),
                        help="LLM 后处理提供方 (默认: qwen)")
    parser.add_argument("--llm-model",
                        help="覆盖提供方的默认模型；可用逗号分隔指定多个候选，按顺序回退（如 qwen3.7-plus,gpt-5.2）")
    parser.add_argument("--no-llm", action="store_true", help="仅转录，不进行 LLM 后处理")
    parser.add_argument("--no-diarization", action="store_true", help="跳过说话人日志，不区分说话人")
    parser.add_argument("--speakers", type=int, help="已知说话人数；默认自动检测")
    parser.add_argument(
        "--spk-max-seg-ms", type=int, default=4000,
        help="说话人区分（spk_model）的 VAD 段上限（毫秒）。段越短，两人问答被并成一段的概率越低；"
             "实验验证档位 8000（882 段 6 人核验正确）；若聚类质量下降可回退 --spk-max-seg-ms 8000",
    )
    parser.add_argument("--audio", type=Path, help="复用已存在的 16k mono WAV，跳过下载与转换（仅单链接时有效）")
    parser.add_argument(
        "--batch-size-s", type=int, default=60,
        help="FunASR VAD 批切段时长（秒，默认 60）。越大单次送入音频越长、调用次数越少但峰值内存越高；内存紧张默认保守，可在 90/120 间 A/B 验证后上调",
    )
    parser.add_argument(
        "--jobs", type=int, default=2,
        help="长音频分块并行转写的 worker 进程数（默认 2；内存紧张时自动降档，可用 --jobs 1 回退串行）",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="忽略结果缓存，强制重新转录与整理",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="启动常驻转写服务（模型只加载一次，常驻内存；HTTP 接口见 src/server.py）",
    )
    parser.add_argument(
        "--server-port", type=int, default=8765,
        help="常驻转写服务端口（默认 8765）",
    )
    parser.add_argument(
        "--use-server", metavar="URL",
        help="走常驻转写服务（如 http://127.0.0.1:8765），不在本进程加载模型",
    )
    args = parser.parse_args()

    if args.server:
        from src.server import start_server

        start_server(
            port=args.server_port,
            model_name=args.model,
            spk_max_seg_ms=args.spk_max_seg_ms,
            batch_size_s=args.batch_size_s,
        )
        return

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    if not args.no_llm and not llm_config.api_key:
        print(
            "未检测到 LLM API Key（LLM_API_KEY / LLM_BASE_URL 未设置），"
            "自动进入免费会话内处理链路。\n"
            "如需自动化校订，请配置这两个环境变量后再运行；当前以 --no-llm 继续。",
            file=sys.stderr,
        )
        args.no_llm = True

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行日志留痕：每次运行写入 output/_run_ep_<url-hash>.log（多期共用首期 hash），
    # 便于事后核对各阶段耗时与失败原因（此前后台运行无日志，只能靠文件时间戳反推）。
    if args.urls:
        import hashlib

        log_hash = hashlib.sha1(args.urls[0].encode("utf-8")).hexdigest()[:8]
        run_log = output_dir / f"_run_ep_{log_hash}.log"
        _fh = logging.FileHandler(run_log, encoding="utf-8")
        _fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(_fh)

    tracker = CostTracker()

    # 批处理：模型只加载一次，逐期复用同一转写器实例（根治并发争抢 + 省去重复加载）。
    # --use-server 时改用常驻服务（HTTP 代理），本进程不加载模型。
    if args.use_server:
        from src.server import HttpTranscriber

        transcriber = HttpTranscriber(base_url=args.use_server, model_name=args.model)
    else:
        from src.transcriber.funasr_transcriber import FunASRTranscriber

        transcriber = FunASRTranscriber(
            model_name=args.model,
            spk_max_seg_ms=args.spk_max_seg_ms,
            batch_size_s=args.batch_size_s,
        )

    n = len(args.urls)
    single = n == 1
    audio_override = args.audio if single else None
    if args.audio is not None and not single:
        print("提示：--audio 仅对单链接有效，多链接时忽略。", file=sys.stderr)

    ok_count = 0
    for i, url in enumerate(args.urls, 1):
        print(f"\n{'='*60}")
        print(f"# 第 {i}/{n} 期: {url}")
        print(f"{'='*60}")
        if process_episode(url, args, output_dir, tracker, transcriber, audio_override, llm_config):
            ok_count += 1

    # 释放并行转写进程池（worker 内的常驻模型随之退出）
    transcriber.shutdown()

    print(f"\n{'='*60}")
    print(f"[完成] 共 {n} 期，成功 {ok_count} 期，失败 {n - ok_count} 期。")
    print(f"  总费用: {tracker.summary()}")
    print(f"{'='*60}")
    if ok_count < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
