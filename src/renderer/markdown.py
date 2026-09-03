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
from src.utils import fmt_time, merge_same_speaker_blocks

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
    lines += ["", "## 全文转录", "", merge_same_speaker_blocks(doc.full_text)]

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
            QuestionItem(question=q.get("question", ""), answer=q.get("answer", ""))
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
