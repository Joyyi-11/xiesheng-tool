"""Abstract base class for transcription providers."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import TranscriptResult


class Transcriber(ABC):
    """Abstract transcriber that converts audio to text."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        duration_sec: float | None = None,
        *,
        work_dir: Path | None = None,
        jobs: int = 1,
    ) -> TranscriptResult:
        """Transcribe audio file and return result with cost tracking.

        Args:
            work_dir: Optional directory for per-chunk resume cache.
            jobs: Number of parallel worker processes for long audio.
        """
        ...
