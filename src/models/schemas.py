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
    """A single key point with supporting evidence."""
    point: str
    evidence: str


@dataclass
class Highlight:
    """A verbatim quote enriched with its location in the episode."""
    content: str
    start_sec: float | None = None
    speaker: str = ""


@dataclass
class Keyword:
    """A single keyword with a short explanation."""
    key: str
    desc: str = ""


@dataclass
class OutlineItem:
    """A hierarchical outline node: a topic (optionally with start time) and sub-points."""
    title: str
    points: list[str] = field(default_factory=list)
    start_sec: float | None = None


@dataclass
class OutputDoc:
    """Final structured output document."""
    title: str
    podcast_name: str
    pub_date: str
    show_notes: str
    key_points: list[KeyPoint]
    highlight_quotes: list[str] = field(default_factory=list)
    full_text: str = ""  # cleaned and formatted full transcript (Markdown)
    speaker_intro: str = ""  # speaker introduction from Show Notes (Markdown)
    highlights: list[Highlight] = field(default_factory=list)  # quotes with timestamp/speaker
    keywords: list[Keyword] = field(default_factory=list)
    outline: list[OutlineItem] = field(default_factory=list)  # hierarchical content outline (mindmap)
    costs: dict = field(default_factory=dict)  # {"transcription": 0.0, "llm": 0.0}
    timings: dict = field(default_factory=dict)  # {"scrape": 0, "transcribe": 0, "process": 0}
