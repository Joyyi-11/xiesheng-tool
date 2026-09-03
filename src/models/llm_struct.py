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
MAX_KEYWORDS_SAFETY_CAP = 10  # 防失控上限，非产品规范（v12 术语表规范 0-4 个）
MAX_QUESTIONS = 5


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


class TermDefOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    term: str = ""
    desc: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.term)


class KeyPointOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    point: str = ""
    evidence: str = ""
    quote: str = ""
    terms: list[TermDefOut] = Field(default_factory=list)

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


class QuestionItemOut(BaseModel):
    model_config = {"str_strip_whitespace": True}
    question: str = ""
    answer: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.question)


class StructOut(BaseModel):
    """Whole structured reading-aids payload returned by the LLM."""

    key_points: list[KeyPointOut] = Field(default_factory=list)
    speaker_intro: str = ""
    speaker_mapping: dict[str, Any] = Field(default_factory=dict)
    keywords: list[KeywordOut] = Field(default_factory=list)
    summary: str = ""
    questions: list[QuestionItemOut] = Field(default_factory=list)


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
        return out[:MAX_KEYWORDS_SAFETY_CAP]

    def questions() -> list[QuestionItemOut]:
        out: list[QuestionItemOut] = []
        for item in _as_list(raw.get("questions")):
            if isinstance(item, dict):
                try:
                    q = QuestionItemOut.model_validate(item)
                    if q.valid:
                        out.append(q)
                except Exception:
                    continue
        return out[:MAX_QUESTIONS]

    mapping = raw.get("speaker_mapping")
    intro = raw.get("speaker_intro")
    return StructOut(
        key_points=key_points(),
        speaker_intro=_clean_str(intro) if isinstance(intro, str) else "",
        speaker_mapping=mapping if isinstance(mapping, dict) else {},
        keywords=keywords(),
        summary=_clean_str(raw.get("summary")),
        questions=questions(),
    )


def struct_to_doc(struct: StructOut) -> dict[str, Any]:
    """Convert a validated StructOut to the plain-dict form used by build doc."""
    return {
        "key_points": [
            {
                "point": k.point,
                "evidence": k.evidence,
                "quote": _clean_str(k.quote),
                "terms": [
                    {"term": t.term, "desc": t.desc} for t in k.terms if t.valid
                ],
            }
            for k in struct.key_points if k.valid
        ],
        "speaker_intro": _clean_str(struct.speaker_intro),
        "speaker_mapping": struct.speaker_mapping or {},
        "keywords": [{"key": k.key, "desc": k.desc} for k in struct.keywords if k.valid],
        "summary": _clean_str(struct.summary),
        "questions": [
            {"question": q.question, "answer": q.answer} for q in struct.questions if q.valid
        ],
    }
