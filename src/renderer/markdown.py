"""Markdown renderer for the structured OutputDoc.

Rendering is kept a pure function of the document so the content model stays
decoupled from presentation. Future renderers (HTML, JSON, ...) can be added
without changing the pipeline.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.models.schemas import OutputDoc, KeyPoint, Keyword, QuestionItem, TermDef
from src.utils import TRANSCRIPT_HEADING, fmt_time, merge_same_speaker_blocks

_TIMING_LABELS = {
    "scrape": "抓取",
    "download": "下载",
    "transcribe": "转写",
    "diarization": "说话人",
    "process": "校订",
}


_SENT_END = ("。", ".", "！", "!", "？", "?", "；", ";", "…")


def _as_sentence(text: str) -> str:
    """规整为一句以句末标点收尾的话（已有句末标点则不动）。

    核心观点由「观点句 + 展开句 + 引用句」顺次组成，不用冒号连接，
    因此每段都必须是能独立收尾的完整句。
    """
    t = (text or "").strip()
    # 中文标点后多余的半角空格（LLM 常见），只删标点后空白，不动中英之间的空格
    t = re.sub(r"([。！？；，、）」』])\s+", r"\1", t)
    if t.endswith(("：", ":")):
        # 观点句以冒号收尾＝旧的「主题句：展开」框架式写法，降级为句号
        t = t[:-1]
    if t and not t.endswith(_SENT_END):
        t += "。"
    return t


def _as_bold_question(text: str) -> str:
    """把「问题与思考」的问题整句加粗（序号留在加粗外，保持有序列表语义）。

    question 字段应为纯文本；若 LLM 已自带 ** 包裹，先剥掉再统一加，避免双重加粗。
    """
    t = (text or "").strip()
    t = re.sub(r"([。！？；，、）」』])\s+", r"\1", t)
    if t.startswith("**") and t.endswith("**") and len(t) > 4:
        t = t[2:-2].strip()
    if not t:
        return ""
    return f"**{t}**"


def _collect_bold_anchors(doc: OutputDoc) -> list[str]:
    """收集原文转录区的回标锚点（用于加粗关键内容）。

    锚点来源两类：
    - 自动锚：核心观点的引用原话 quote（逐字摘录）、观点内术语 terms、术语表残留
      keywords——这些词/句本身逐字来自原文，直接按文本匹配即可；
    - 显式锚：核心观点句与问题句是提炼改写、无逐字原文，由 L3 提炼方在
      KeyPoint.anchor / QuestionItem.anchor 中给出「转录区逐字存在的支撑句/短语」，
      渲染时按它回标。
    匹配原则：转录区**首次出现**处加粗（克制，避免满篇加粗淹没正文）；
    锚点未在转录区出现（校订后措辞变化）时静默跳过，不报错、不改字。
    长度克制：quote 若过长（>24 字）整句加粗会淹没正文，跳过；术语词条限 ≤12 字；
    显式 anchor 限 ≤40 字（超出视为 L3 给错，跳过）。
    """
    anchors: list[str] = []
    for kp in doc.key_points:
        if kp.quote and kp.quote.strip() and len(kp.quote.strip()) <= 24:
            anchors.append(kp.quote.strip())
        if kp.anchor and kp.anchor.strip() and len(kp.anchor.strip()) <= 40:
            anchors.append(kp.anchor.strip())
        for term in kp.terms:
            if term.term and term.term.strip() and len(term.term.strip()) <= 12:
                anchors.append(term.term.strip())
    for kw in doc.keywords:
        if kw.key and kw.key.strip() and len(kw.key.strip()) <= 12:
            anchors.append(kw.key.strip())
    for q in doc.questions:
        if q.anchor and q.anchor.strip() and len(q.anchor.strip()) <= 40:
            anchors.append(q.anchor.strip())
    # 去重且按长度降序：长锚先加粗，短锚（可能是长锚子串）不再叠加
    seen: set[str] = set()
    out: list[str] = []
    for a in sorted(anchors, key=len, reverse=True):
        a = a.strip()
        if not a or a in seen:
            continue
        if len(a) < 2:  # 单字锚噪音大，忽略
            continue
        seen.add(a)
        out.append(a)
    return out


def _apply_bold_anchors(transcript_md: str, anchors: list[str]) -> str:
    """把锚点在转录文本的首次出现处加粗为 **锚点**。

    只匹配【标签】之后的正文（每行标签如【swyx】不参与），避免把说话人标签误加粗；
    同一锚点只在全文首次出现处加粗一次；已加粗区间内的子串不再二次加粗。
    """
    if not anchors:
        return transcript_md

    # 行级处理：把【标签】与正文分开，正文参与匹配
    lines = transcript_md.split("\n")
    bolded: set[str] = set()  # 已加粗过的锚（全局只加粗首次出现）
    out_lines: list[str] = []
    for line in lines:
        m = re.match(r"^(\s*【[^】]*】)(.*)$", line)
        label = m.group(1) if m else ""
        body = m.group(2) if m else line
        if not body.strip() or not body:
            out_lines.append(line)
            continue
        # 该行内按锚长降序找首次出现位置（跳过已加粗区间）
        spans: list[tuple[int, int, str]] = []
        occupied: list[tuple[int, int]] = []
        for a in anchors:
            if a in bolded:
                continue
            idx = _find_first_free(body, a, occupied)
            if idx is None:
                continue
            spans.append((idx, idx + len(a), a))
            occupied.append((idx, idx + len(a)))
            bolded.add(a)  # 全局首次出现即锁定，后文不再重复加粗
        if not spans:
            out_lines.append(line)
            continue
        # 从右往左插入 **，避免破坏后续偏移
        spans.sort(reverse=True)
        new_body = body
        for start, end, a in spans:
            new_body = new_body[:end] + "**" + new_body[end:]
            new_body = new_body[:start] + "**" + new_body[start:]
        out_lines.append(label + new_body if label else new_body)
    return "\n".join(out_lines)


def _find_first_free(text: str, needle: str, occupied: list[tuple[int, int]]) -> int | None:
    """在 text 中找 needle 首次出现、且不与 occupied 区间重叠的位置（-1 表示无）。"""
    if not needle:
        return None
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return None
        end = idx + len(needle)
        if all(end <= o_start or idx >= o_end for o_start, o_end in occupied):
            return idx
        start = idx + 1


def fmt_ts(sec: float | None) -> str:
    """Format a start timestamp as ``MM:SS`` or ``H:MM:SS``."""
    if sec is None:
        return "??:??"
    total = int(round(sec))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_output_markdown(doc: OutputDoc) -> str:
    """Render a structured document to Markdown."""
    lines = [
        f"# {doc.title}",
        "",
        f"> 来源：{doc.podcast_name} | {doc.title} | {doc.pub_date}",
    ]
    for warning in doc.warnings:
        lines += ["", f"> [WARN] {warning}"]
    lines += ["", "## Show Notes", ""]
    if doc.show_notes and doc.show_notes.strip():
        lines.append(doc.show_notes)
    else:
        lines.append("（无）")

    lines += ["", "## 摘要", "", doc.summary]

    lines += ["", "## 核心观点", ""]
    for kp in doc.key_points:
        # 结构＝观点句、展开论述句、引用句，三段顺次相接，中间用空格、
        # 不用冒号（旧的「**观点句**：展开」是本节冒号泛滥的主因，v17 起废止）。
        line = f"- **{_as_sentence(kp.point)}**"
        if kp.evidence:
            line += f" {_as_sentence(kp.evidence)}"
        if kp.quote:
            # 原文佐证行内化：直接跟在展开句后，用直角引号包裹，不单独成引用块；
            # 展开句已以句号收尾，句号后不再留空格（无展开句时与观点句间留一空格）
            line += f"{'' if kp.evidence else ' '}「{kp.quote}」"
        lines.append(line)
        for term in kp.terms:
            desc = (term.desc or "").strip()
            if desc and not desc.endswith(("。", ".")):
                desc += "。"
            sep = "：" if desc else ""
            lines.append(f"  - **{term.term}**{sep}{desc}")

    lines += ["", "## 问题与思考", ""]
    for i, q in enumerate(doc.questions, 1):
        # 问题整句加粗、序号留在加粗外（`1. **问题？**`）；答案不加粗，换行缩进另写
        question = _as_bold_question(q.question)
        lines.append(f"{i}. {question}" if question else f"{i}.")
        if q.answer:
            lines.append(f"   {q.answer}")
        if i < len(doc.questions):
            lines.append("")

    lines += ["", "## 术语表", ""]
    for kw in doc.keywords:
        desc = (kw.desc or "").strip()
        if desc and not desc.endswith(("。", ".")):
            desc += "。"
        sep = "：" if desc else ""
        lines.append(f"- **{kw.key}**{sep}{desc}")

    lines += ["", "## 人物简介", "", doc.speaker_intro or "（无）"]

    # 同一说话人的相邻段落在渲染时确定性合并（脚本保证，不依赖 LLM）：源转录按停顿
    # 切碎、且同一人常被 diarization 判成多个簇，合并后才是「一人一段」的可读形态。
    # 合并后再做回标加粗：术语/金句/观点支撑句首次出现处加粗，未命中静默跳过。
    transcript_md = merge_same_speaker_blocks(doc.full_text)
    transcript_md = _apply_bold_anchors(transcript_md, _collect_bold_anchors(doc))
    lines += ["", TRANSCRIPT_HEADING, "", transcript_md]

    footer = _build_footer(doc)
    if footer:
        lines += ["", "---", "", footer]

    return "\n".join(lines)


def _build_footer(doc: OutputDoc) -> str:
    """Build a compact generation-info footer from timings and costs, if present."""
    if not doc.timings and not doc.costs:
        return ""
    parts = []
    if doc.timings:
        present = [key for key, _label in _TIMING_LABELS.items() if key in doc.timings]
        if present:
            chunks = [f"{_TIMING_LABELS[k]} {fmt_time(doc.timings[k])}" for k in present]
            parts.append("耗时 " + " / ".join(chunks))
    llm_cost = doc.costs.get("llm")
    if isinstance(llm_cost, (int, float)):
        parts.append(f"LLM 费用约 {llm_cost:.4f} 元")
    return "；".join(parts)


def _doc_from_dict(data: dict) -> OutputDoc:
    """从 OutputDoc JSON 字典构造 OutputDoc（含嵌套子对象）。

    会话链路 handoff JSON 含 ``speaker_mapping``（不属于 OutputDoc 字段），调用方
    应先据其对 ``full_text`` 落地显示标签再传入本函数；此处仅负责结构还原。
    """
    return OutputDoc(
        title=data["title"],
        podcast_name=data["podcast_name"],
        pub_date=data["pub_date"],
        show_notes=data.get("show_notes", ""),
        key_points=[
            KeyPoint(
                point=kp.get("point", ""),
                evidence=kp.get("evidence", ""),
                quote=kp.get("quote", ""),
                anchor=kp.get("anchor", ""),
                terms=[
                    TermDef(term=t.get("term", ""), desc=t.get("desc", ""))
                    for t in kp.get("terms", [])
                ],
            )
            for kp in data.get("key_points", [])
        ],
        full_text=data.get("full_text", ""),
        speaker_intro=data.get("speaker_intro", ""),
        keywords=[Keyword(key=kw.get("key", ""), desc=kw.get("desc", "")) for kw in data.get("keywords", [])],
        summary=data.get("summary", ""),
        questions=[
            QuestionItem(
                question=q.get("question", ""),
                answer=q.get("answer", ""),
                anchor=q.get("anchor", ""),
            )
            for q in data.get("questions", [])
        ],
        costs=data.get("costs", {}) or {},
        timings=data.get("timings", {}) or {},
        warnings=data.get("warnings", []) or [],
    )


def main() -> None:
    """CLI：``python -m src.renderer.markdown <json> -o <md>``。

    把 OutputDoc JSON 渲染为 v12 Markdown。若 ``full_text`` 仍含 ``[SPEAKER_XX]``
    且提供 ``speaker_mapping``，由脚本统一经 ``to_display_label`` 落地【短名】标签，
    与 API 链路显示格式等价（不依赖 LLM）。
    """
    parser = argparse.ArgumentParser(description="渲染 OutputDoc JSON 为 v12 Markdown（脚本固定【短名】标签）")
    parser.add_argument("json_path", type=Path, help="OutputDoc JSON 文件（会话 handoff 骨架）")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出 .md 路径")
    args = parser.parse_args()

    raw = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    # 显示标签统一由脚本落地：speaker_mapping {SPEAKER_XX: canonical_name} → 【短名】
    mapping = raw.pop("speaker_mapping", None)
    if raw.get("full_text") and mapping:
        from src.diarization.speaker_resolver import to_display_label

        ft = raw["full_text"]
        for spk, name in mapping.items():
            if re.fullmatch(r"SPEAKER_\d+", str(spk)) and name:
                ft = re.sub(
                    rf"\[{re.escape(str(spk))}\]\s*",
                    to_display_label(str(name)),
                    ft,
                )
        raw["full_text"] = ft

    doc = _doc_from_dict(raw)
    out = build_output_markdown(doc)
    args.output.write_text(out, encoding="utf-8")
    print(f"已渲染: {args.output}（{len(out)} 字符）", file=sys.stderr)


if __name__ == "__main__":
    main()
