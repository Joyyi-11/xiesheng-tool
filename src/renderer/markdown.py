"""Markdown renderer for the structured OutputDoc.

Rendering is kept a pure function of the document so the content model stays
decoupled from presentation. Future renderers (HTML, JSON, ...) can be added
without changing the pipeline.
"""

from src.models.schemas import OutputDoc
from src.utils import fmt_time

_TIMING_LABELS = {
    "scrape": "抓取",
    "download": "下载",
    "transcribe": "转写",
    "diarization": "说话人",
    "process": "校订",
}


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
    if doc.show_notes:
        lines += ["", "# Show Notes", "", doc.show_notes]

    if doc.summary:
        lines += ["", "## 摘要", "", doc.summary]

    lines += ["", "## 内容提要", ""]
    for kp in doc.key_points:
        evidence = f"：{kp.evidence}" if kp.evidence else ""
        lines.append(f"- **{kp.point}**{evidence}")

    if doc.highlights:
        lines += ["", "## 闪光语句", ""]
        for hl in doc.highlights:
            ts = f"[{fmt_ts(hl.start_sec)}]" if hl.start_sec is not None else ""
            speaker = f"（{hl.speaker}）" if hl.speaker else ""
            meta = f"{ts}{speaker}".strip()
            prefix = f" {meta}" if meta else ""
            lines.append(f"-{prefix} 「{hl.content}」")
    elif doc.highlight_quotes:
        lines += ["", "## 闪光语句", ""]
        for q in doc.highlight_quotes:
            lines.append(f"- {q}")

    if doc.questions:
        lines += ["", "## 问题与思考", ""]
        for q in doc.questions:
            lines.append(f"- **{q.question}** {q.answer}")

    if doc.keywords:
        lines += ["", "## 关键词", ""]
        for kw in doc.keywords:
            desc = f"：{kw.desc}" if kw.desc else ""
            lines.append(f"- **{kw.key}**{desc}")

    if doc.speaker_intro:
        lines += ["", "## 人物简介", "", doc.speaker_intro]

    lines += ["", "## 全文转录", "", doc.full_text]

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
