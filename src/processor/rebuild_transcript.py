"""保真重建会话内校订的原文转录（防「过度删减」返工）。

会话内校订（会话链路 / --llm-mode session）最常见的问题：AI 校订时把口语长叙述压缩成概要，
导致 ``validate_session_output`` 报「原文转录长度仅为原始转录的 <50%（疑似
过度删减）」。本工具从 ``<节目>_diarized.txt`` 会话包**保真重建**原文转录：

* 100% 保留原始转录内容（不改写、不缩写、不省略）；
* 每个 ``[SPEAKER_XX]`` 段按序保留，合并同标签连续段；
* 可选的 ``--speaker-map`` 把标签映射为真实身份（不提供则保留原标签，
  留给校订阶段按语义处理）；
* 输出初稿 ``.md``（标题/来源/Show Notes 取自会话包）或仅替换既有 ``.md``
  的 ``## 原文转录`` 段（前面已校订好的章节原样保留；旧标题 ``## 全文转录``
  同样可识别，替换后统一写回新标题）。

用法::

    # 生成完整初稿 .md
    python -m src.processor.rebuild_transcript output/<节目>_diarized.txt -o output/<节目>.md

    # 已知说话人时直接映射
    python -m src.processor.rebuild_transcript output/<节目>_diarized.txt \\
        --speaker-map "SPEAKER_00:主播 Jean,SPEAKER_01:嘉宾姨姨" -o output/<节目>.md

    # 只替换既有 .md 的原文转录段（保留已校订的摘要/核心观点等章节）
    python -m src.processor.rebuild_transcript output/<节目>_diarized.txt -o output/<节目>.md --replace-transcript

说明：本工具只做保真重建，不做 ASR 纠错（错字、专名、填充词删除仍由校订阶段
完成），因此重建后的初稿信息量必然 ≥50% 阈值；在此基础上人工校订只会进一步
清理，不会再触发「过度删减」。
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from src.utils import (
    LEGACY_TRANSCRIPT_HEADING,
    SPEAKER_LINE_RE,
    TRANSCRIPT_HEADING,
)

logger = logging.getLogger(__name__)


def parse_package(package_text: str) -> tuple[str, str, str, str, list[tuple[str, str]]]:
    """Split a session package into (title, source_line, show_notes, transcript, segments).

    ``segments`` is ``[(label, text)]`` in file order.
    """
    lines = package_text.splitlines()
    title = ""
    source_line = ""
    show_notes: list[str] = []
    transcript: list[str] = []
    segs: list[tuple[str, str]] = []

    in_notes = False
    in_transcript = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# 转录全文"):
            in_transcript = True
            in_notes = False
            continue
        if in_transcript:
            m = SPEAKER_LINE_RE.match(stripped)
            if m:
                segs.append((m.group(1), m.group(2).strip()))
            transcript.append(line)
            continue
        if not title and stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped.startswith("> 来源："):
            source_line = stripped
            continue
        if stripped == "# Show Notes":
            in_notes = True
            continue
        if in_notes:
            show_notes.append(line)
    return title, source_line, "\n".join(show_notes).strip(), "\n".join(transcript).strip(), segs


def build_transcript(segs: list[tuple[str, str]], speaker_map: dict[str, str]) -> str:
    """Faithful rebuild: keep every segment, merge consecutive same-label runs.

    Returns the transcript body (no trailing blank line). Without a mapping the
    original ``[SPEAKER_XX]`` labels are kept so the editor can still attribute
    by content; with a mapping they become ``【名字】`` markers (v9 方括号格式，
    无冒号、无加粗，与 ``session_edit.SESSION_RULES`` 一致）。
    """
    # 合并键取「映射后的姓名」而非原始标签：diarization 常把同一人切成多个簇
    # （如 SPEAKER_01 与 SPEAKER_03 都映射到「湫湫」），只按原标签合并会留下
    # 同人相邻多段——正是「明明一个人讲却贴了好多标签」的根因。
    merged: list[tuple[str, str, bool]] = []  # (key, text, is_name)
    for label, text in segs:
        if not text:
            continue
        name = speaker_map.get(label, "")
        key = name or label
        if merged and merged[-1][0] == key:
            prev_key, prev_text, is_name = merged[-1]
            merged[-1] = (prev_key, prev_text + text, is_name)
        else:
            merged.append((key, text, bool(name)))

    out: list[str] = []
    for key, text, is_name in merged:
        if is_name:
            out.append(f"【{key}】{text}")
        else:
            out.append(f"[{key}] {text}")
    return "\n\n".join(out)


def build_markdown(
    title: str,
    source_line: str,
    show_notes: str,
    transcript_body: str,
) -> str:
    """Assemble a minimal draft .md (title/source/Show Notes/原文转录)."""
    lines = [
        f"# {title}",
        "",
        source_line or "> 来源：",
        "",
        "## Show Notes",
        "",
        show_notes or "（无）",
        "",
        TRANSCRIPT_HEADING,
        "",
        transcript_body,
    ]
    return "\n".join(lines) + "\n"


def replace_transcript_section(md: str, transcript_body: str) -> str:
    """Replace everything from the transcript heading onward in an existing .md.

    旧标题 ``## 全文转录`` 同样可识别（存量稿无需手工改名），写回时统一为新标题。
    """
    for heading in (TRANSCRIPT_HEADING, LEGACY_TRANSCRIPT_HEADING):
        head, sep, _ = md.partition(heading)
        if sep:
            return head.rstrip() + f"\n\n{TRANSCRIPT_HEADING}\n\n" + transcript_body + "\n"
    raise ValueError(f"目标 .md 中未找到 {TRANSCRIPT_HEADING} 小节，无法替换")


def parse_speaker_map(text: str) -> dict[str, str]:
    """Parse 'SPEAKER_00:主播 Jean,SPEAKER_01:嘉宾姨姨' into a dict."""
    result: dict[str, str] = {}
    if not text:
        return result
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"speaker-map 条目格式应为 标签:名字，收到：{item!r}")
        label, name = item.split(":", 1)
        label = label.strip()
        name = name.strip()
        if not re.fullmatch(r"SPEAKER_\d+", label):
            raise ValueError(f"未知说话人标签：{label!r}（应为 SPEAKER_XX）")
        if not name:
            raise ValueError(f"说话人 {label} 缺少名字")
        result[label] = name
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从会话包保真重建原文转录（防过度删减；输出初稿 .md）"
    )
    parser.add_argument("package", type=Path, help="会话输入包（<节目>_diarized.txt）")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="输出 .md 路径（默认打印转录到 stdout，不写文件）",
    )
    parser.add_argument(
        "--speaker-map", default="",
        help='说话人映射，逗号分隔，如 "SPEAKER_00:主播 Jean,SPEAKER_01:嘉宾姨姨"；'
             "不提供则保留 [SPEAKER_XX] 标签由校订阶段处理",
    )
    parser.add_argument(
        "--replace-transcript", action="store_true",
        help=f"替换既有 .md 的 {TRANSCRIPT_HEADING} 段（保留该文件前面已校订的章节）；"
             "未提供 --output 时无效",
    )
    args = parser.parse_args()

    if not args.package.exists():
        print(f"错误：找不到会话输入包 {args.package}", file=sys.stderr)
        sys.exit(1)
    package_text = args.package.read_text(encoding="utf-8")
    title, source_line, show_notes, _, segs = parse_package(package_text)
    if not segs:
        print(f"错误：会话包中未找到带 [SPEAKER_XX] 的转录段：{args.package}", file=sys.stderr)
        sys.exit(1)

    try:
        speaker_map = parse_speaker_map(args.speaker_map)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

    body = build_transcript(segs, speaker_map)
    logger.info("保真重建完成：%d 段 -> %d 段（合并同标签连续段）", len(segs), body.count("\n\n") + 1)
    logger.info("重建转录 %d 字符（原始包 %d 字符）", len(body), len(package_text))

    if args.output is None:
        print(body)
        return

    out = args.output
    if out.exists() and args.replace_transcript:
        existing = out.read_text(encoding="utf-8")
        new_md = replace_transcript_section(existing, body)
    else:
        new_md = build_markdown(title, source_line, show_notes, body)
    out.write_text(new_md, encoding="utf-8")
    print(f"\n[OK] 保真重建初稿已写入: {out}")


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
