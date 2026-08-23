"""Speaker diarization — assign speaker labels to transcribed segments."""

import logging
from pathlib import Path

from diarize import diarize

logger = logging.getLogger(__name__)

SPEAKER_GAP = 0.2  # seconds — 同说话人碎片合并的间隔上限（0.5s→0.2s 收紧，
# 避免把短停顿的相邻段拼成更长的段；跨说话人绝不合并，见 assign_speakers）


def run_diarization(audio_path: Path, num_speakers: int | None = None) -> list[dict]:
    """Run speaker diarization on an audio file.

    Returns list of dicts with start, end, speaker keys.
    """
    logger.info("Running speaker diarization on %s...", audio_path.name)
    result = diarize(str(audio_path), num_speakers=num_speakers)
    segments = [
        {"start": seg.start, "end": seg.end, "speaker": seg.speaker}
        for seg in result.segments
    ]
    logger.info(
        "Diarization done: %d speakers, %d segments",
        result.num_speakers,
        len(segments),
    )
    return segments


def assign_speakers(
    asr_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """Merge diarization speaker labels with ASR text segments.

    Each asr_segment: {"text": ..., "start_time": ..., "end_time": ...}
    Each diarization_segment: {"start": ..., "end": ..., "speaker": ...}

    Returns a list of labeled utterance segments:
    [{"speaker": ..., "text": ..., "start": ..., "end": ...}]

    Complexity is near-linear: diarization segments are sorted once, and a moving
    pointer skips segments that end before the current ASR segment starts.
    """
    dseg = sorted(diarization_segments, key=lambda d: d["start"])
    n = len(dseg)
    ptr = 0

    # Assign best-overlap speaker to each ASR segment
    labeled = []
    for wseg in asr_segments:
        ws, we = wseg["start_time"], wseg["end_time"]
        while ptr < n and dseg[ptr]["end"] <= ws:
            ptr += 1
        best_overlap, best_speaker = 0.0, "UNKNOWN"
        for k in range(ptr, n):
            if dseg[k]["start"] >= we:
                break
            overlap = min(we, dseg[k]["end"]) - max(ws, dseg[k]["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg[k]["speaker"]
        labeled.append(
            {"speaker": best_speaker, "text": wseg["text"], "start": ws, "end": we}
        )

    # Merge consecutive same-speaker segments (with a small gap tolerance) into utterances
    # 注意：仅在 speaker 完全相同时才合并；跨说话人绝不合并，防止两人问答被拼成
    # 一段、LLM 只能按段首贴名（Vol.11 说话人错乱的直接成因）。
    merged = []
    for seg in labeled:
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and seg["start"] - merged[-1]["end"] <= SPEAKER_GAP
        ):
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append({"speaker": seg["speaker"], "text": seg["text"], "start": seg["start"], "end": seg["end"]})

    return merged


def format_labeled_segments(segments: list[dict], max_line_chars: int = 1500) -> str:
    """Format labeled utterance segments into the [SPEAKER_XX] text form.

    每个逻辑 utterance 以 ``[SPEAKER_XX]`` 开头；段内文本按句（。！？；）换行，
    避免单人节目合并后整段压成一行、超出 Read 工具 2000 字符行上限被截断。
    仅在句末换行、不改变说话人归属，会话内校订仍可逐段贴名。
    """
    import re

    out: list[str] = []
    for seg in segments:
        tag = f"[{seg['speaker']}] "
        text = seg["text"]
        # 按句切分（保留句末标点），再按 max_line_chars 折行，每行都带说话人标签
        chunks = re.split(r"(?<=。|！|？|；)", text)
        buf = ""
        for ch in chunks:
            if len(buf) + len(ch) > max_line_chars:
                if buf:
                    out.append(tag + buf.strip())
                    buf = ""
                # 单行本身过长（罕见）直接输出，下一行重新带标签
                out.append(tag + ch.strip())
            else:
                buf += ch
        if buf.strip():
            out.append(tag + buf.strip())
    return "\n".join(out)
