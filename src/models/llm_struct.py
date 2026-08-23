"""Pydantic models for LLM structured output.

LLM JSON responses are used to drive multiple, previously hand-written parsing
paths (key points, quotes, keywords, speaker mapping, outline). Centralizing
their validation in Pydantic models keeps parsing robust, explicit, and
testable, and drops the brittle manual dict-walking code in llm_processor.
"""

from typing import Any

from pydantic import BaseModel, Field

MAX_KEY_POINTS = 20
MAX_QUOTES = 20
MAX_KEYWORDS = 10
MAX_OUTLINE_ITEMS = 15
MAX_OUTLINE_POINTS = 6


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_seconds(value: Any) -> float | None:
    """Best-effort parse of a timeline start into seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            return max(0.0, float(text))
        except ValueError:
            pass
        # optional h:mm:ss / mm:ss
        parts = text.split(":")
        if 2 <= len(parts) <= 3:
            try:
                nums = [int(x) for x in parts]
                if len(nums) == 3:
                    return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
                return float(nums[0] * 60 + nums[1])
            except ValueError:
                return None
        return None
    return None


class KeyPointOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    point: str = ""
    evidence: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.point)


class KeywordOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    key: str = ""
    desc: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.key)


class OutlineItemOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    title: str = ""
    points: list[str] = Field(default_factory=list)
    start: Any = None  # mm:ss / h:mm:ss / 秒，渲染为一级主题的时间前缀

    @property
    def valid(self) -> bool:
        return bool(self.title)


class SpeakerMappingOut(BaseModel):
    model_config = {"extra": "allow", "str_strip_whitespace": True}


class StructOut(BaseModel):
    """Whole structured reading-aids payload returned by the LLM."""

    key_points: list[KeyPointOut] = Field(default_factory=list)
    highlight_quotes: list[str] = Field(default_factory=list)
    speaker_intro: str = ""
    speaker_mapping: dict[str, Any] = Field(default_factory=dict)
    keywords: list[KeywordOut] = Field(default_factory=list)
    outline: list[OutlineItemOut] = Field(default_factory=list)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_struct(raw: dict[str, Any] | None) -> StructOut:
    """Coerce an LLM JSON payload into a validated StructOut (tolerant).

    Invalid fields are dropped individually rather than failing the whole parse,
    so one malformed LLM field never blocks the document.
    """
    raw = raw if isinstance(raw, dict) else {}

    def key_points() -> list[KeyPointOut]:
        out: list[KeyPointOut] = []
        for item in _as_list(raw.get("key_points")):
            if isinstance(item, dict):
                try:
                    out.append(KeyPointOut.model_validate(item))
                except Exception:
                    continue
        return out[:MAX_KEY_POINTS]

    def quotes() -> list[str]:
        return [_clean_str(q) for q in _as_list(raw.get("highlight_quotes")) if _clean_str(q)][:MAX_QUOTES]

    def keywords() -> list[KeywordOut]:
        out: list[KeywordOut] = []
        for item in _as_list(raw.get("keywords")):
            if isinstance(item, dict):
                try:
                    kw = KeywordOut.model_validate(item)
                    if kw.valid:
                        out.append(kw)
                except Exception:
                    continue
        return out[:MAX_KEYWORDS]

    def outline() -> list[OutlineItemOut]:
        out: list[OutlineItemOut] = []
        for item in _as_list(raw.get("outline")):
            if isinstance(item, dict):
                try:
                    cleaned = dict(item)
                    points = cleaned.get("points")
                    if isinstance(points, list):
                        # 容忍子点中的非字符串元素，避免一个坏子点拖垮整个大纲节点
                        cleaned["points"] = [
                            _clean_str(p) for p in points if p is not None
                        ]
                    node = OutlineItemOut.model_validate(cleaned)
                    if node.valid:
                        node.points = [
                            p for p in node.points if p
                        ][:MAX_OUTLINE_POINTS]
                        out.append(node)
                except Exception:
                    continue
        return out[:MAX_OUTLINE_ITEMS]

    mapping = raw.get("speaker_mapping")
    intro = raw.get("speaker_intro")
    return StructOut(
        key_points=key_points(),
        highlight_quotes=quotes(),
        speaker_intro=_clean_str(intro) if isinstance(intro, str) else "",
        speaker_mapping=mapping if isinstance(mapping, dict) else {},
        keywords=keywords(),
        outline=outline(),
    )


def struct_to_doc(struct: StructOut) -> dict[str, Any]:
    """Convert a validated StructOut to the plain-dict form used by build doc."""
    return {
        "key_points": [
            {"point": k.point, "evidence": k.evidence} for k in struct.key_points if k.valid
        ],
        "highlight_quotes": [_clean_str(q) for q in struct.highlight_quotes if _clean_str(q)],
        "speaker_intro": _clean_str(struct.speaker_intro),
        "speaker_mapping": struct.speaker_mapping or {},
        "keywords": [{"key": k.key, "desc": k.desc} for k in struct.keywords if k.valid],
        "outline": [
            {
                "title": node.title,
                "points": [p for p in node.points if p],
                "start": node.start,
            }
            for node in struct.outline
            if node.valid
        ],
    }
