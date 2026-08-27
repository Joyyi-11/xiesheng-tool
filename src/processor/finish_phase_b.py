"""finish_phase_b — Phase B 幂等续跑工具（防「自动续跑断点」）。

背景：批量转录分 Phase A（本地 ASR → *_diarized.txt）与 Phase B（校订 → *.md）。
Phase B 的断点恢复不能依赖「主对话跨轮次唤醒」，而要依赖「状态幂等」：
本脚本每次运行都扫描全部会话包，把「能自动完成的」做完（生成缺失初稿、
补来源行三要素），把「必须人工的」列成清单（写上层章节、说话人映射）。

用法::

    # 只读扫描：报告每期状态（缺 md / 校验问题 / 残留标签 / 待校订章节）
    python -m src.processor.finish_phase_b

    # 生成缺失的 .md 初稿（v6 结构占位 + 保真全文转录，保留 [SPEAKER_XX] 待映射）
    python -m src.processor.finish_phase_b --gen-missing

    # 自动补齐来源行三要素（播客 | 节目标题 | 日期，节目标题取自 # 标题）
    python -m src.processor.finish_phase_b --fix-source

    # 全部可自动项一次做完（gen-missing + fix-source），并打印校验报告
    python -m src.processor.finish_phase_b --all

    # 指定扫描目录（默认 output/），用于隔离测试
    python -m src.processor.finish_phase_b --dir output

    # 可选：说话人映射文件（JSON: {"<diarized 文件名>": "SPEAKER_00:名字,..."}）
    # 生成初稿时若命中则带上映射，否则保留 [SPEAKER_XX] 标签待人工映射
    python -m src.processor.finish_phase_b --gen-missing --speaker-map-file speaker_maps.json

说明：
- 幂等：重复运行结果一致；除 --fix-source 外不修改已有 .md；--fix-source 只改来源行。
- 不自动猜说话人：残留 [SPEAKER_XX] 一律列进报告，由人工按内容归并后用
  rebuild_transcript --replace-transcript --speaker-map 替换全文转录。
- 上层章节（摘要/内容提要/闪光语句/问题与思考/关键词/人物简介）必须人工撰写，
  初稿中以「（待校订）」占位，脚本据此列出待办清单。
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.processor.rebuild_transcript import (
    build_markdown,
    build_transcript,
    parse_package,
    parse_speaker_map,
)
from src.processor.session_edit import _split_sections, validate_session_output

PLACEHOLDER_SECTIONS = {
    "## 摘要": "（待校订：2-3 句总结本期主题与核心结论，不列要点）",
    "## 内容提要": "- **待校订。** 待提炼本期重要观点（每条以加粗主题句开头、句号结尾）",
    "## 闪光语句": "- **待校订**：待摘录原文精彩表述（关键词加粗，忠实原话）",
    "## 问题与思考": "- **① 待校订？** 待提炼 3-5 个节目真正讨论的核心问题与张力",
    "## 关键词": "- **待校订**：待提取 4-8 个贯穿节目的核心概念",
    "## 人物简介": "**待校订**：（填写主播/嘉宾身份与姓名，一行一位，正文格式）",
}
PLACEHOLDER_MARK = "待校订"


def find_diarized(directory: Path) -> list[Path]:
    return sorted(directory.glob("*_diarized.txt"))


def md_path_for(diarized: Path) -> Path:
    return diarized.with_name(diarized.name[: -len("_diarized.txt")] + ".md")


def source_line_fix(md_text: str, title: str) -> str:
    """补齐来源行三要素（播客 | 节目标题 | 日期）；缺节目标题时用 # 标题补。"""
    lines = md_text.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("> 来源："):
            body = s[len("> 来源：") :].strip()
            parts = [p.strip() for p in body.split("|")]
            # 三要素 = 播客名 | 节目标题 | 日期（任意顺序无法判断，按 播客|标题|日期 约定）
            if len(parts) >= 3 and all(parts[:3]):
                return md_text  # 已合规
            # 缺中段：把标题插到第 2 位（若只有 2 段）
            if len(parts) == 2 and parts[0] and parts[1]:
                new_body = f"{parts[0]}  |  {title}  |  {parts[1]}"
                lines[i] = "> 来源：" + new_body
                return "\n".join(lines)
    return md_text


def build_draft(diarized: Path, speaker_map: dict[str, str]) -> str:
    """生成 v6 结构初稿：标题/来源/Show Notes/上层章节占位/全文转录。"""
    package_text = diarized.read_text(encoding="utf-8")
    title, source_line, show_notes, _, segs = parse_package(package_text)
    if not source_line:
        source_line = "> 来源："
    body = build_transcript(segs, speaker_map)
    parts = [
        f"# {title}",
        "",
        source_line,
        "",
        "# Show Notes",
        "",
        show_notes or "（无）",
    ]
    for heading, placeholder in PLACEHOLDER_SECTIONS.items():
        parts += ["", heading, "", placeholder]
    parts += ["", "## 全文转录", "", body]
    return "\n".join(parts) + "\n"


def analyze_one(diarized: Path, md_path: Path, fix_source: bool) -> dict:
    src = diarized.read_text(encoding="utf-8")
    m = re.search(r"# 转录全文\n(.*)", src, re.S)
    src_text = m.group(1) if m else src

    result = {
        "diarized": diarized.name,
        "md": md_path.name,
        "exists": md_path.exists(),
        "probs": [],
        "placeholders": [],
        "ratio": 0.0,
        "labels": 0,
    }
    if not md_path.exists():
        return result

    md = md_path.read_text(encoding="utf-8")
    if fix_source:
        title_m = re.match(r"#\s+(.+)", md)
        title = title_m.group(1).strip() if title_m else ""
        fixed = source_line_fix(md, title)
        if fixed != md:
            md_path.write_text(fixed, encoding="utf-8")
            md = fixed
            result["source_fixed"] = True

    result["probs"] = validate_session_output(md, src_text)
    # 精确取「## 全文转录」段统计残留标签（避免把「## 校订说明」等后续章节的引用算进去）
    trans_section = _split_sections(md).get("## 全文转录", "")
    result["ratio"] = len(trans_section) / max(len(src_text), 1) if src_text else 0
    result["labels"] = len(re.findall(r"\[SPEAKER_\d+\]", trans_section))
    for heading in PLACEHOLDER_SECTIONS:
        if PLACEHOLDER_MARK in (md.split(heading, 1)[1].split("\n## ", 1)[0] if heading in md else ""):
            result["placeholders"].append(heading)
    return result


def print_speaker_preview(directory: Path) -> None:
    """只读打印每个会话包的说话人画像：段数、字符数、首段样本。"""
    from collections import Counter, defaultdict

    for diarized in find_diarized(directory):
        text = diarized.read_text(encoding="utf-8")
        segs = re.findall(r"^\[(SPEAKER_\d+)\]\s*(.*)$", text, re.M)
        cnt: Counter[str] = Counter(l for l, _ in segs)
        chars: dict[str, int] = defaultdict(int)
        first: dict[str, str] = {}
        for l, t in segs:
            t2 = t.strip()
            chars[l] += len(t2)
            if t2 and l not in first:
                first[l] = t2
        print(f"=== {diarized.name}（{len(segs)} 段 / {len(cnt)} 个标签）===")
        for l in sorted(cnt):
            sample = first.get(l, "")
            print(f"  {l}  segs={cnt[l]}  chars={chars[l]}  |  {sample[:90]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B 幂等续跑：扫描、生成缺失初稿、补齐来源行、校验")
    parser.add_argument("--dir", type=Path, default=Path("output"), help="扫描目录（默认 output/）")
    parser.add_argument("--gen-missing", action="store_true", help="为缺 .md 的会话包生成 v6 初稿")
    parser.add_argument("--fix-source", action="store_true", help="自动补齐来源行三要素")
    parser.add_argument("--speaker-map-file", type=Path, default=None, help="JSON: {diarized文件名: 'SPEAKER_00:名字,...'}")
    parser.add_argument("--speaker-preview", action="store_true", help="打印每个会话包的说话人画像（段数/字符数/首段样本），辅助映射判定，只读")
    parser.add_argument("--all", action="store_true", help="等价 --gen-missing --fix-source")
    args = parser.parse_args()

    if args.all:
        args.gen_missing = True
        args.fix_source = True

    if args.speaker_preview:
        print_speaker_preview(args.dir)
        return

    speaker_maps: dict[str, dict[str, str]] = {}
    if args.speaker_map_file and args.speaker_map_file.exists():
        raw = json.loads(args.speaker_map_file.read_text(encoding="utf-8"))
        for k, v in raw.items():
            speaker_maps[k] = parse_speaker_map(v)

    generated = 0
    for diarized in find_diarized(args.dir):
        md_path = md_path_for(diarized)
        if args.gen_missing and not md_path.exists():
            smap = speaker_maps.get(diarized.name, {})
            md_path.write_text(build_draft(diarized, smap), encoding="utf-8")
            generated += 1

    print(f"扫描目录: {args.dir.resolve()} | 会话包 {len(find_diarized(args.dir))} 个 | 新生成初稿 {generated} 个\n")
    print(f"{'状态':<6} {'节目':<40} {'占比':>5} {'残留标签':>6}  说明")
    print("-" * 100)
    done = pending = problem = 0
    for diarized in find_diarized(args.dir):
        md_path = md_path_for(diarized)
        r = analyze_one(diarized, md_path, args.fix_source)
        title = md_path.stem if r["exists"] else diarized.stem
        if not r["exists"]:
            print(f"{'缺.md':<6} {title[:40]:<40} {'-':>5} {'-':>6}  运行 --gen-missing 生成初稿")
            pending += 1
            continue
        notes = []
        if r["placeholders"]:
            notes.append("待校订章节:" + ",".join(h.replace("## ", "") for h in r["placeholders"]))
        if r["labels"]:
            notes.append(f"残留标签 {r['labels']} 个（需人工映射）")
        for p in r["probs"]:
            notes.append(p)
        if r.get("source_fixed"):
            notes.append("来源行已补齐")
        status = "✅" if not notes else ("⏳" if any("待校订" in n or "残留" in n for n in notes) else "⚠️")
        if status == "✅":
            done += 1
        elif status == "⏳":
            pending += 1
        else:
            problem += 1
        print(f"{status:<6} {title[:40]:<40} {r['ratio']:>4.0%} {r['labels']:>6}  {'; '.join(notes) if notes else 'OK'}")
    print("-" * 100)
    print(f"已完成 {done} | 待校订/待映射 {pending} | 校验问题 {problem}")


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
