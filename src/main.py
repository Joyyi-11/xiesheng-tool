"""撷声 CLI.

Usage:
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx -o output/
    xiesheng url1 url2 url3 -o output/   # 多期：模型只加载一次，逐期转录
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Windows terminal encoding fix
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.audio import convert_to_wav, download_audio, get_duration_seconds
from src.config import DEFAULT_LLM_PROVIDER, LLM_MODELS, get_llm_config, llm_configured
from src.models.schemas import EpisodeInfo
from src.processor import llm_processor
from src.processor.normalize import normalize_quotes
from src.processor.output_quality import assert_valid_output_doc
from src.renderer.markdown import build_output_markdown
from src.result_cache import find_cached_handoff, find_cached_markdown, record_handoff, record_result
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

    The session-link (--llm-mode session) produces this instead of the API path. It embeds episode
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


def emit_session_handoff(
    output_dir, url, episode, raw_path, labeled_path, safe_name,
    spk_to_label=None, transcript_text="",
) -> None:
    """会话内校对链路交付：产出 OutputDoc JSON 骨架，交会话内 agent 填内容后跑渲染 CLI。

    会话链路与 API 链路唯一区别是「LLM 在哪跑」——前者由会话内的 agent 执行校对，
    后者由流水线调 LLM API。两者成品结构应等价，且显示标签由脚本（to_display_label）
    统一落地，不依赖 LLM 是否「遵守约定」。

    本函数只产出骨架 JSON（转写保留原始 [SPEAKER_XX]，说话人映射归一到 canonical
    姓名 + show notes/roster），会话内 agent 补全 summary/key_points/... 后运行：
        python -m src.renderer.markdown <name>_handoff.json -o <name>.md
    渲染脚本按 to_display_label 输出【短名】标签，与 API 链路等价。
    """
    print(f"\n原始转录已保存到: {raw_path}")
    if labeled_path is None:
        print("（未做说话人分离；如需说话人标签，请提供 HF_TOKEN 后重跑）")
        record_handoff(output_dir, url, episode.pub_date, raw_path)
        return
    print(f"会话输入包（含 Show Notes 与说话人标签）已保存到: {labeled_path}")
    from src.processor.session_edit import SESSION_SPEC_VERSION

    skeleton = {
        "title": episode.title,
        "podcast_name": episode.podcast_name,
        "pub_date": episode.pub_date,
        "show_notes": episode.show_notes,
        "full_text": normalize_quotes(transcript_text),
        "speaker_mapping": dict(spk_to_label or {}),
        "speaker_intro": "",
        "summary": "",
        "key_points": [],
        "questions": [],
        "keywords": [],
    }
    json_path = output_dir / f"{safe_name}_handoff.json"
    json_path.write_text(
        json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"会话校订骨架（OutputDoc JSON，v{SESSION_SPEC_VERSION}）已保存到: {json_path}")
    print(
        f"→ 在会话内补全 summary / 核心观点 / 问题与思考 / 术语表 / 人物简介"
        f"（可修正 speaker_mapping），然后运行："
    )
    print(f"    python -m src.renderer.markdown {json_path.name} -o {safe_name}.md")
    print(f"  渲染脚本按 to_display_label 统一输出【短名】标签，与 API 链路等价。")
    record_handoff(output_dir, url, episode.pub_date, json_path)


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
            if args.llm_mode == "session":
                cached_handoff = find_cached_handoff(output_dir, url, episode.pub_date)
                if cached_handoff is not None:
                    print(f"\n[会话包缓存命中] 会话包已生成，等待人工校订: {cached_handoff}")
                    print("完成后请直接校订该输入包；如需重新转录，请加 --refresh")
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

        # --- Step 3 + 4: 转录 + 说话人解析（会话链路产出命名输入包，由会话内 agent 校对）---
        # 从 Show Notes 解析预期说话人数，作为 diarization 的 K 锚点
        # （常规两人播客：主播 + 单嘉宾；显式 --speakers 优先覆盖）。
        from src.diarization.speaker_resolver import (
            detect_diarization_issues,
            parse_roster_with_roles,
            resolve_and_label,
        )

        roster = parse_roster_with_roles(episode.show_notes)
        roster_names = [name for _, name in roster]
        base_k = args.speakers if args.speakers else (len(roster) if len(roster) >= 2 else None)
        if roster:
            logger.info(
                "Step 3: Show Notes 识别到 %d 位说话人（%s），diarization K 锚点=%s",
                len(roster), "、".join(roster_names), base_k,
            )

        # K-retry 质量门：先按 base_k 强制聚类；若检测到「同簇多人自称」合并矛盾
        # （声线相近者被并成一类，本集失败根因），自动升 K 重对齐（整段重转录约
        # 数十秒，ASR+对齐缓存按 K 隔离），直到无合并矛盾或试到 base_k+3。
        transcript = None
        chosen_k = None
        merge_issues: list[str] = []
        can_force_k = transcriber.can_force_num_speakers(duration_sec)
        transcribe_elapsed = 0.0
        if base_k and not args.no_diarization and can_force_k:
            for attempt_k in range(base_k, base_k + 4):
                logger.info("Step 3/6: 本地转录中（FunASR %s, CPU, K=%d）...", args.model, attempt_k)
                with Timer() as t:
                    transcript = transcriber.transcribe(
                        wav_file, duration_sec,
                        work_dir=output_dir / ".work" / safe_name,
                        jobs=args.jobs, num_speakers=attempt_k,
                        no_diarization=args.no_diarization,
                    )
                transcribe_elapsed += t.elapsed
                tracker.add_transcription(transcript.cost_yuan)
                segs = transcript.segments
                issues = detect_diarization_issues(segs, roster)
                merge_issues = [i for i in issues if "建议提高 K" in i]
                if merge_issues:
                    logger.warning("K=%d 质量门：合并矛盾 → %s（升 K 重试）", attempt_k, "；".join(merge_issues))
                else:
                    chosen_k = attempt_k
                    logger.info("K=%d 质量门通过（无合并矛盾）", attempt_k)
                    break
                chosen_k = attempt_k
            if merge_issues:
                logger.warning("K-retry 至 %d 仍检测到合并矛盾，使用最后一次对齐结果（说话人或需人工复核）", chosen_k)
                logger.warning(
                    "人工复核建议：可尝试 --speakers N 手动指定说话人数，"
                    "或用 --no-diarization 跳过自动分离后人工标注。"
                )
        else:
            if base_k and not args.no_diarization:
                logger.info("短音频走整段转写路径，强制 K 不生效，跳过 K-retry")
            logger.info("Step 3/6: 本地转录中（FunASR %s, CPU）...", args.model)
            with Timer() as t:
                transcript = transcriber.transcribe(
                    wav_file, duration_sec,
                    work_dir=output_dir / ".work" / safe_name,
                    jobs=args.jobs, num_speakers=base_k,
                    no_diarization=args.no_diarization,
                )
            transcribe_elapsed += t.elapsed
            tracker.add_transcription(transcript.cost_yuan)
            chosen_k = base_k

        timers["transcribe"] = transcribe_elapsed
        char_count = len(transcript.raw_text)
        rtf = transcribe_elapsed / duration_sec if duration_sec else 0
        est_2h = int(7200 * rtf)
        print(f"  → 转录完成：{char_count} 字, {fmt_time(transcribe_elapsed)}, RTF={rtf:.2f}")
        print(f"  → [估算] 2小时节目约需 {fmt_time(est_2h)}（当前模型: {args.model}）")
        gate_note = ""
        if base_k and not args.no_diarization:
            gate_note = "（质量门通过）" if not merge_issues else "（质量门告警·见上）"
        print(f"  → 说话人聚类 K={chosen_k}{gate_note}")

        raw_path = output_dir / f"{safe_name}_raw.txt"
        raw_path.write_text(transcript.raw_text, encoding="utf-8")
        logger.info("原始转录已保存到 %s", raw_path)

        # --- Step 4: 说话人标签 + 命名解析（独立于 LLM）---
        # SenseVoice 转写自带说话人标签（spk_model 一站式输出）时直接使用；
        # Paraformer 路径（无 spk 标签）才回退到独立 diarize。
        transcript_text = transcript.raw_text
        labeled_segments: list[dict] | None = None
        labeled_path = None
        resolve_issues: list[str] = []
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
                    diarization_segments = run_diarization(wav_file, num_speakers=chosen_k)
                    labeled_segments = assign_speakers(transcript.segments, diarization_segments)
                try:
                    # 原始 [SPEAKER_XX] 版（保持与历史/LLM 链路一致，供校订脚本逐段贴名）
                    transcript_text = format_labeled_segments(labeled_segments)
                    # 只做确定性身份核验；无证据的簇保留标签，交给人工或 LLM 校订。
                    labeled_segments, spk_to_label, resolve_issues = resolve_and_label(labeled_segments, episode.show_notes)
                    labeled_path = output_dir / f"{safe_name}_diarized.txt"
                    labeled_path.write_text(
                        build_session_doc(episode, normalize_quotes(transcript_text)),
                        encoding="utf-8",
                    )
                    logger.info("带说话人标签的转录已保存到 %s", labeled_path)
                    if resolve_issues:
                        logger.warning("说话人命名告警：%s", "；".join(resolve_issues))
                except Exception as e:
                    logger.warning("说话人标签处理失败，跳过: %s", e)
                    transcript_text = transcript.raw_text
                    labeled_segments = None
            timers["diarization"] = t.elapsed
        else:
            logger.info("已跳过说话人分离（--no-diarization）")

        if args.llm_mode == "session":
            # 会话内校对链路：流水线产出「会话输入包」+ 固定 CLEAN+STRUCT 校订规范，
            # 由会话内的 agent 按当前规范校对并输出与 API 链路等价结构的 .md。
            emit_session_handoff(output_dir, url, episode, raw_path, labeled_path, safe_name, spk_to_label=spk_to_label, transcript_text=transcript_text)
            return True
        else:
            # --- Step 5: LLM Process ---
            with Timer() as t:
                logger.info("Step 5/6: %s/%s 分块校订与整理中...", llm_config.provider, llm_config.model)
                try:
                    doc, inp_tok, out_tok = llm_processor.process(
                        episode.title,
                        episode.podcast_name,
                        episode.pub_date,
                        episode.show_notes,
                        transcript_text,
                        llm_config=llm_config,
                        work_dir=output_dir / ".work" / safe_name,
                        segments=labeled_segments,
                    )
                except Exception as exc:
                    logger.warning("LLM 后处理失败，降级为会话内链路: %s", exc)
                    print(
                        f"\n[降级] LLM 后处理失败（{exc}），本次改走会话内处理链路。",
                        file=sys.stderr,
                    )
                    emit_session_handoff(output_dir, url, episode, raw_path, labeled_path, safe_name, spk_to_label=spk_to_label, transcript_text=transcript_text)
                    return True
                tracker.add_llm_usage(inp_tok, out_tok)
                if merge_issues:
                    doc.warnings.append(
                        f"K-retry 在 K={base_k}..K={chosen_k} 均检测到合并矛盾，"
                        f"说话人映射可能不准确，请人工复核文稿中的【SPEAKER_XX】残留标签。"
                    )
                try:
                    assert_valid_output_doc(doc)
                except ValueError as exc:
                    logger.warning("LLM 成品校验失败，降级为会话内链路: %s", exc)
                    print(f"\n[降级] {exc}，本次改走会话内处理链路。", file=sys.stderr)
                    emit_session_handoff(output_dir, url, episode, raw_path, labeled_path, safe_name, spk_to_label=spk_to_label, transcript_text=transcript_text)
                    return True
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
                degraded = bool(doc.warnings)
                output_path = output_dir / f"{safe_name}{'.degraded' if degraded else ''}.md"
                output_path.write_text(md_content, encoding="utf-8")
                if not degraded:
                    record_result(
                        output_dir,
                        url,
                        episode.pub_date,
                        output_path,
                        provider=llm_config.provider,
                        model=llm_config.model,
                    )
            timers["write"] = t.elapsed

            total_time = sum(timers.values())
            print(f"\n{'='*50}")
            if degraded:
                print("[WARN] 已产出降级文稿，未写入最终结果缓存！")
                for warning in doc.warnings:
                    print(f"  - {warning}")
            else:
                print("[OK] 完成！")
            print(f"  输出文件: {output_path}")
            print(f"  总耗时: {fmt_time(total_time)}")
            print(f"  费用: {tracker.summary()}")
            print(f"  要点数: {len(doc.key_points)}")
            print(f"  核心观点（含原话）: {len([kp for kp in doc.key_points if kp.quote])}")
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
    parser.add_argument(
        "urls", nargs="*",
        help="小宇宙播客单集链接（可传多个；模型只加载一次，逐期转录；--server 模式可省略）",
    )
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认: output/)")
    parser.add_argument("--model", default="sensevoice-small", choices=["sensevoice-small", "paraformer-large"],
                        help="ASR 模型（默认 sensevoice-small；paraformer-large 支持热词与字级时间戳）")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=sorted(LLM_MODELS),
                        help="LLM 后处理提供方 (默认: qwen)")
    parser.add_argument("--llm-model",
                        help="覆盖提供方的默认模型；可用逗号分隔指定多个候选，按顺序回退（如 qwen3.7-plus,gpt-5.2）")
    parser.add_argument("--llm-mode", choices=["api", "session"], default=None,
                        help="LLM 校对方式：api=流水线调用 LLM API（需 API Key）；"
                             "session=在会话内由 agent 校对（无需 Key）。省略时自动检测（有 Key→api，无→session）")
    parser.add_argument("--no-llm", action="store_true",
                        help="[弃用] 等价于 --llm-mode session，后续版本移除")
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
        help="FunASR VAD 批切段时长（秒，默认 60）。越大单次送入音频越长、调用次数越少但峰值内存越高；"
             "内存紧张默认保守，可在 90/120 间 A/B 验证后上调",
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
        if args.urls:
            parser.error("--server 只启动常驻转写服务，不能同时传入单集 URL")
        from src.server import start_server

        start_server(
            port=args.server_port,
            model_name=args.model,
            spk_max_seg_ms=args.spk_max_seg_ms,
            batch_size_s=args.batch_size_s,
        )
        return

    if not args.urls:
        parser.error("缺少小宇宙播客单集链接（--server 模式可省略 URL）")

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    # 解析有效 LLM 模式：显式 --llm-mode > 弃用 --no-llm > 自动检测（有 Key→api，无→session）
    if args.no_llm:
        print("警告：--no-llm 已弃用，等价于 --llm-mode session，后续版本移除。", file=sys.stderr)
        args.llm_mode = "session"
    if args.llm_mode is None:
        args.llm_mode = "api" if llm_configured() else "session"
        if args.llm_mode == "session":
            print("未完整检测到 LLM_API_KEY 与 LLM_BASE_URL，自动进入会话内校对链路（session）。", file=sys.stderr)
    elif args.llm_mode == "api" and not llm_configured():
        print(
            "警告：已指定 --llm-mode api 但未同时检测到 LLM_API_KEY 与 LLM_BASE_URL，"
            "降级为会话内链路（session）。",
            file=sys.stderr,
        )
        args.llm_mode = "session"

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
        try:
            transcriber.ensure_model()
        except Exception as exc:
            print(f"错误：常驻转写服务不可用或模型不匹配: {exc}", file=sys.stderr)
            sys.exit(1)
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
