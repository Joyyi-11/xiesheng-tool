from dataclasses import dataclass, field


@dataclass
class EpisodeInfo:
    """Xiaoyuzhou episode metadata scraped from the page."""
    url: str
    title: str
    podcast_name: str
    pub_date: str
    show_notes: str
    audio_url: str
    cover_url: str | None = None


@dataclass
class TranscriptResult:
    """Result from ASR transcription."""
    raw_text: str
    segments: list[dict]  # [{"text": ..., "start_time": ..., "end_time": ...}]
    duration_sec: float
    cost_yuan: float = 0.0


@dataclass
class KeyPoint:
    """A single core viewpoint with supporting evidence and an optional verbatim quote."""
    point: str
    evidence: str
    quote: str = ""  # optional verbatim quote from the transcript (忠实原文)


@dataclass
class Keyword:
    """A single keyword with a short explanation."""
    key: str
    desc: str = ""


@dataclass
class QuestionItem:
    """A curated question + synthesized answer for the 问题与思考 section."""
    question: str
    answer: str = ""


@dataclass
class OutputDoc:
    """Final structured output document."""
    title: str
    podcast_name: str
    pub_date: str
    show_notes: str
    key_points: list[KeyPoint]
    full_text: str = ""  # cleaned and formatted full transcript (Markdown)
    speaker_intro: str = ""  # speaker introduction from Show Notes (Markdown)
    keywords: list[Keyword] = field(default_factory=list)
    summary: str = ""  # 2-3 sentence episode summary
    questions: list[QuestionItem] = field(default_factory=list)  # curated Q&A for reflection
    costs: dict = field(default_factory=dict)  # {"transcription": 0.0, "llm": 0.0}
    timings: dict = field(default_factory=dict)  # {"scrape": 0, "transcribe": 0, "process": 0}
