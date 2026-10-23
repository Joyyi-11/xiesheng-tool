"""Cost tracking and shared helpers (JSON IO, speaker-label regex, etc.)."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CostTracker:
    transcription_yuan: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_yuan: float = 0.0
    llm_cost_known: bool = False

    def add_transcription(self, cost: float) -> None:
        self.transcription_yuan += cost

    def add_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_yuan: float | None = None,
    ) -> None:
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        if cost_yuan is not None:
            self.llm_cost_yuan += cost_yuan
            self.llm_cost_known = True

    @property
    def total_yuan(self) -> float:
        return self.transcription_yuan + self.llm_cost_yuan

    def summary(self) -> str:
        llm_cost = f"{self.llm_cost_yuan:.4f} 元" if self.llm_cost_known else "以供应商账单为准"
        return (
            f"转录: {self.transcription_yuan:.4f} 元, "
            f"LLM: {llm_cost} "
            f"(输入 {self.llm_input_tokens} / 输出 {self.llm_output_tokens} tokens), "
            f"已知费用: {self.total_yuan:.4f} 元"
        )


def safe_filename(value: str, max_length: int = 80) -> str:
    """Return a Windows-safe filename stem."""
    translation = str.maketrans({char: "_" for char in '<>:"/\\|?*'})
    cleaned = value.translate(translation).strip().rstrip(".")
    return cleaned[:max_length] or "podcast"


def episode_stem(podcast_name: str, title: str) -> str:
    """产物统一命名：<节目来源>-<本期节目名称>（连漪 2026-10-14 定）。

    例：``十字路口Crossing-AI产品经理指南：我是谁，从哪来，到哪去｜对谈字节AI产品负责人Vanessa，…``

    规则：前缀不占标题的字符配额——标题再长也只截标题、不截来源；
    来源缺失时退回纯标题。全角标点（：｜，）不在 Windows 非法字符集内，原样保留。
    """
    source = (podcast_name or "").strip()
    body = (title or "").strip()
    parts = [
        safe_filename(value, max_length=limit)
        for value, limit in ((source, 40), (body, 100))
        if value
    ]
    return safe_filename("-".join(parts), max_length=160)


class Timer:
    """Simple timer context manager."""
    def __init__(self):
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}分{s}秒"


# --- 共享正则：避免各模块重复书写 [SPEAKER_XX] 模式 ---
SPEAKER_LABEL_RE = re.compile(r"\[SPEAKER_\d+\]")
SPEAKER_LINE_RE = re.compile(r"^\[(SPEAKER_\d+)\]\s*(.*)$", re.MULTILINE)

# --- 转录小节标题（单一权威源，各链路一律引用以下常量，禁止字面量硬编码）---
# 2026-09-04 更名：「全文转录」→「原文转录」。读取侧仍兼容旧标题，避免存量成稿
# 在校验、重建、续跑时被判为「缺少小节」。
TRANSCRIPT_HEADING = "## 原文转录"
LEGACY_TRANSCRIPT_HEADING = "## 全文转录"
TRANSCRIPT_HEADINGS = (TRANSCRIPT_HEADING, LEGACY_TRANSCRIPT_HEADING)
# 匹配任一历史写法的二级标题（用于正则定位转录段）
TRANSCRIPT_HEADING_RE = re.compile(r"^##\s*(?:原文|全文)转录\s*$", re.MULTILINE)


def get_transcript_body(sections: dict[str, str]) -> str:
    """按 {标题: 正文} 取转录段正文，兼容旧标题「## 全文转录」。"""
    for heading in TRANSCRIPT_HEADINGS:
        if heading in sections:
            return sections[heading]
    return ""


# --- 保留率（成稿转录区 ÷ 会话包语音正文）的唯一口径 ---
# 分母若含标题 / Show Notes / 说话人标签，保留率会虚低近一半（实测报告 47%、
# 真实 100%）。校验器与 Phase B 报告都必须走这里，不要在各自模块里另算一份。
_SPEAKER_OR_SPACE_RE = re.compile(r"【[^】]*】|\[SPEAKER_\d+\]|\s")


def strip_speaker_labels(text: str) -> str:
    """剥掉【姓名】/ [SPEAKER_XX] 标签与空白，只留语音正文（用于比长度）。"""
    return _SPEAKER_OR_SPACE_RE.sub("", text)


def package_transcript_body(package_text: str) -> str:
    """取会话包里的语音转录部分：首个 [SPEAKER_XX] 起，丢掉标题与 Show Notes。

    部分会话包在首个标签前嵌了 Show Notes / 简介等非语音文本，分母若含这些
    头部，保留率会被算成 30%（真实 103%）。
    """
    m = SPEAKER_LABEL_RE.search(package_text)
    return package_text[m.start():] if m else package_text


def transcript_retention(transcript: str, package_text: str) -> float:
    """成稿转录区相对会话包的保留率（1.0 = 未删减）。"""
    denominator = len(strip_speaker_labels(package_transcript_body(package_text)))
    return len(strip_speaker_labels(transcript)) / max(denominator, 1)


# --- 原文转录：合并同一说话人的相邻段落（确定性操作，脚本化而非交给 LLM）---
# 成因：源转录按语音停顿切碎（同一人连续多段），且 diarization 常把同一人判成
# 多个簇（如 SPEAKER_01/SPEAKER_03 都映射到同一姓名），映射后即为同名相邻段。
# LLM 校订时是否合并不可控（同一份素材有时合并、有时不合并），故由脚本兜底。
SPEAKER_BLOCK_RE = re.compile(r"^【([^】]+)】\s*(.*)$")
# 段尾已有句读时直接拼接；否则补句号，避免合并后产生无断点的连读长串。
_NO_SEP_TAIL = ("。", "！", "？", "…", "；", "，", "、", ".", "!", "?", ";", ",", " ")


def merge_same_speaker_blocks(transcript: str) -> str:
    """把「## 原文转录」正文里同一说话人的相邻段落合并为一段。

    只做拼接与去重标签、不改动文字：连续（中间无其他说话人）的同名段落并成一段，
    段首只保留一次【身份姓名】；无标签的续行并入上一段。输入为空则原样返回。
    """
    if not transcript.strip():
        return transcript

    items: list[tuple[str | None, list[str]]] = []
    for block in re.split(r"\n\s*\n", transcript):
        for line in block.splitlines():
            if not line.strip():
                continue
            m = SPEAKER_BLOCK_RE.match(line.strip())
            # 分隔线「---」等结构性行独立成项：不得并入上一位说话人的段落，
            # 否则页脚/分节会被粘进转录正文并多出一个句号。
            if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", line.strip()):
                items.append((None, [line.strip()]))
            elif m:
                items.append((m.group(1), [m.group(2).strip()]))
            elif items:
                items[-1][1].append(line.strip())
            else:
                items.append((None, [line.strip()]))

    merged: list[tuple[str | None, list[str]]] = []
    for speaker, parts in items:
        if speaker is not None and merged and merged[-1][0] == speaker:
            merged[-1][1].extend(parts)
        else:
            merged.append((speaker, list(parts)))

    out: list[str] = []
    for speaker, parts in merged:
        text = ""
        for part in parts:
            if not text or not part:
                text += part
            elif text.endswith(_NO_SEP_TAIL):
                text += part
            else:
                text += "。" + part
        # 关闭话轮：ASR 常在话轮边界把一句话切开，导致「一个人讲完却用逗号结尾」。
        # 话轮边界应视作句末，逗号收尾须转句号（下一行本就是另一位说话人）。
        if text.endswith(("，", ",")):
            text = text[:-1] + "。"
        out.append(f"【{speaker}】{text}" if speaker else text)
    return "\n\n".join(out)


def read_json(path: Path | str) -> Any | None:
    """读取 JSON 文件；不存在或解析失败返回 None（安全读，供缓存复用）。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path | str, obj: Any) -> None:
    """写入 JSON 文件（ensure_ascii=False, indent=2）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
