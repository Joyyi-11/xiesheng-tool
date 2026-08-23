"""Local ASR using FunASR (SenseVoice-Small / Paraformer-Large) on CPU.

Replaces the old faster-whisper backend. FunASR models ship their own VAD and
long-audio handling, so we transcribe the whole WAV in one pass -- no ffmpeg
chunking, no multi-process parallelism. Typical CPU speed: SenseVoice-Small
~9x real-time, Paraformer-Large ~4x on the reference machine (i5-12500H).

SenseVoice output carries emotion/event emoji tags (e.g. <HAPPY>) and no
punctuation by default; we strip the tags and add punctuation via the shared
ct-punc model. Paraformer-Large includes ct-punc at load time.

Speaker diarization: SenseVoice loads with ``spk_model`` (cam++) so FunASR
returns ``sentence_info`` entries carrying a ``spk`` label plus word-level
``timestamp``/``words``; ``_build_segments`` folds the spk label into each
segment (``speaker`` key). This replaces the external pyannote-style diarize
step for the default model: labels come out of the same pass, no separate
audio round-trip needed. Paraformer keeps external diarization (no spk_model
in its FunASR output path).

Segment granularity is controlled by ``spk_max_seg_ms`` (VAD 段上限，默认 4000ms):
cam++ 按 VAD 段聚类，段越长越容易把两人对话（快速问答、无缝接话）并成一段
只给一个标签，这正是 Vol.11 说话人错乱的根源。benchmark/vol11-asr-spk-comparison
实测 8000ms 档位 882 段 6 人核验正确；默认 4000ms 进一步降低段内混人概率，
若聚类质量下降可用 ``--spk-max-seg-ms 8000`` 回退实验档位。
"""

import json
import logging
import os
import re
import time
import wave
from pathlib import Path

from src.models.schemas import TranscriptResult
from src.transcriber.base import Transcriber

logger = logging.getLogger(__name__)

# 中文全角标签：SenseVoice 残留的情绪/事件标记 <...> / <|...|> 与中文全角 〈...〉
_TAG_RE = re.compile(r"<[^>]*>|<\|\^?[^>]*\|>|〈[^〉]*〉")

MODELSCOPE_CACHE = str(Path.home() / ".cache" / "modelscope")

# SenseVoice 说话人区分模型（FunASR spk_model，cam++ 声纹）
SENSEVOICE_SPK_MODEL = "iic/speech_campplus_sv_zh_en_16k-common_advanced"

# 各模型在 modelscope 缓存目录（MODELSCOPE_CACHE/models/<id>）下的目录名
# SenseVoice 需要：SenseVoiceSmall + ct-punc 标点 + fsmn-vad + cam++ 说话人
# Paraformer 需要：seaco-paraformer-large + ct-punc + fsmn-vad
_MODEL_CACHE_DIRS = {
    "sensevoice-small": (
        "iic--SenseVoiceSmall",
        "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "iic--speech_campplus_sv_zh_en_16k-common_advanced",
    ),
    "paraformer-large": (
        "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "iic--punc_ct-transformer_cn-en-common-vocab471067-large",
        "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
    ),
}


def _strip_sensevoice_tags(text: str) -> str:
    """Remove SenseVoice emotion/event emoji tags and normalize whitespace.

    SenseVoice prepends language/emotion/event markers such as ``<|zh|>``,
    ``<HAPPY>`` or ``</emoji>``. ``rich_transcription_postprocess`` removes the
    ASCII ``<|...|>`` block but leaves the Chinese-angle ``<...>`` tags, so we
    strip both here.
    """
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _audio_duration_sec(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _postprocess_sensevoice(raw_text: str) -> str:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    return _strip_sensevoice_tags(rich_transcription_postprocess(raw_text))


class FunASRTranscriber(Transcriber):
    """Transcribe locally using FunASR on CPU (SenseVoice-Small or Paraformer-Large)."""

    SUPPORTED = ("sensevoice-small", "paraformer-large")

    def __init__(
        self,
        model_name: str = "sensevoice-small",
        modelscope_cache: str | None = None,
        spk_max_seg_ms: int = 4000,
    ):
        if model_name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported ASR model: {model_name}. Choose from: {', '.join(self.SUPPORTED)}"
            )
        self.model_name = model_name
        self._modelscope_cache = modelscope_cache or MODELSCOPE_CACHE
        # 说话人区分（spk_model）的 VAD 段上限（毫秒）。cam++ 按 VAD 段聚类，
        # 段越长越易把两人对话并成一段只给一个标签；默认 4000ms 兼顾粒度与聚类质量。
        self.spk_max_seg_ms = spk_max_seg_ms
        self._model = None
        self._punc_model = None
        self._loaded = False

    # --- model loading (lazy; funasr is imported only here) ---
    def _model_cache_status(self) -> tuple[int, list[str]]:
        """Return (hit_count, missing_dirs) for the model's required cache dirs."""
        required = _MODEL_CACHE_DIRS.get(self.model_name, ())
        cache_root = Path(self._modelscope_cache) / "models"
        missing = [d for d in required if not (cache_root / d).is_dir()]
        return len(required) - len(missing), missing

    def _local_model_path(self, model_id: str) -> str | None:
        """Resolve a modelscope model id to its local snapshot dir, if cached.

        FunASR/AutoModel accepts either a hub id (``iic/xxx``, triggers download
        + lock files) or a local directory. We prefer the local snapshot path so
        the hub layer is bypassed entirely -- modelscope's lock-file cleanup is
        blocked in some sandboxed environments (Windows recycle-bin unavailable),
        which previously crashed the process with a segfault right at load time.
        """
        cache_root = Path(self._modelscope_cache) / "models"
        model_dir = cache_root / model_id.replace("/", "--")
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        for snap in sorted(snapshots.iterdir()):
            if snap.is_dir() and any(snap.iterdir()):
                return str(snap)
        return None

    def _resolve_model_arg(self, model_id: str, default: str) -> str:
        """Return the local snapshot path if cached, else the hub id (download)."""
        return self._local_model_path(model_id) or default

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        os.environ.setdefault("MODELSCOPE_CACHE", self._modelscope_cache)
        hit, missing = self._model_cache_status()
        if missing:
            logger.info(
                "模型缓存部分命中（%d/%d 在 %s），将下载缺失模型: %s",
                hit, hit + len(missing), self._modelscope_cache, ", ".join(missing),
            )
        else:
            logger.info("模型缓存已命中（%s），直接用本地模型，跳过下载", self._modelscope_cache)
        from funasr import AutoModel

        logger.info("Loading FunASR model: %s (CPU)...", self.model_name)
        t0 = time.time()
        if self.model_name == "sensevoice-small":
            sv = self._resolve_model_arg("iic/SenseVoiceSmall", "iic/SenseVoiceSmall")
            punc = self._resolve_model_arg(
                "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            )
            spk = self._resolve_model_arg(SENSEVOICE_SPK_MODEL, SENSEVOICE_SPK_MODEL)
            vad = self._resolve_model_arg(
                "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "fsmn-vad"
            )
            self._model = AutoModel(
                model=sv,
                trust_remote_code=True,
                vad_model=vad,
                # VAD 段长上限（spk_max_seg_ms）：说话人区分（spk_model）按 VAD 段
                # 聚类，段越长粒度越粗（30s 段长实测仅 2 人 4 段；8s 实测 882 段
                # 6 人）。默认 4000ms 进一步降低「两人问答被并进一段」的概率，
                # 该段正是 Vol.11 说话人错乱的根源。转写不受影响（FunASR 内部自
                # 处理长音频）。
                vad_kwargs={"max_single_segment_time": self.spk_max_seg_ms},
                spk_model=spk,
                disable_update=True,
                ignore_instances=True,
            )
            # 标点模型与 SenseVoice 解耦，单独加载以便公平补齐标点
            self._punc_model = AutoModel(
                model=punc,
                disable_update=True,
                ignore_instances=True,
            )
        else:  # paraformer-large
            pf = self._resolve_model_arg(
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "paraformer-zh",
            )
            vad = self._resolve_model_arg(
                "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "fsmn-vad"
            )
            punc = self._resolve_model_arg(
                "iic/punc_ct-transformer_cn-en-common-vocab471067-large", "ct-punc"
            )
            self._model = AutoModel(
                model=pf,
                vad_model=vad,
                punc_model=punc,
                disable_update=True,
                ignore_instances=True,
            )
        logger.info("Model loaded in %.1f sec", time.time() - t0)
        self._loaded = True

    # --- cache (whole-audio, keyed by audio fingerprint + model) ---
    def _cache_dir(self, work_dir: Path | None) -> Path | None:
        return (work_dir / "transcribe") if work_dir else None

    def _load_cache(self, work_dir, audio_path, model_name) -> dict | None:
        cache_dir = self._cache_dir(work_dir)
        if cache_dir is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        st = audio_path.stat()
        if not (
            meta.get("audio_size") == st.st_size
            and meta.get("audio_mtime") == int(st.st_mtime)
            and meta.get("model_name") == model_name
            # 段粒度参数也纳入缓存指纹：改了 spk_max_seg_ms 后旧段结果必须作废，
            # 否则重跑会静默复用旧粒度的说话人标签，改动不生效。
            and meta.get("spk_max_seg_ms") == self.spk_max_seg_ms
        ):
            return None
        result_path = cache_dir / "result.json"
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, work_dir, audio_path, model_name, result) -> None:
        cache_dir = self._cache_dir(work_dir)
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        st = audio_path.stat()
        meta = {
            "audio_size": st.st_size,
            "audio_mtime": int(st.st_mtime),
            "model_name": model_name,
            "spk_max_seg_ms": self.spk_max_seg_ms,
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (cache_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )

    # --- core ---
    def transcribe(self, audio_path, duration_sec=None, *, work_dir=None, jobs=1) -> TranscriptResult:
        audio_path = Path(audio_path)
        duration_sec = duration_sec or _audio_duration_sec(audio_path)

        cached = self._load_cache(work_dir, audio_path, self.model_name)
        if cached is not None:
            logger.info("复用整段转写缓存（模型: %s）", self.model_name)
            return TranscriptResult(
                raw_text=cached["raw_text"],
                segments=cached["segments"],
                duration_sec=cached.get("duration_sec", duration_sec or 0.0),
                cost_yuan=0.0,
            )

        self._ensure_loaded()
        t0 = time.time()
        logger.info("Transcribing (FunASR %s, CPU)...", self.model_name)
        if self.model_name == "sensevoice-small":
            res = self._model.generate(
                input=str(audio_path),
                cache={},
                language="zh",
                use_itn=False,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
                spk_model=True,
                output_timestamp=True,
            )
            segments = self._build_segments(res, postprocess=_postprocess_sensevoice, add_punc=True)
        else:  # paraformer-large
            res = self._model.generate(input=str(audio_path), batch_size_s=60)
            segments = self._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)

        raw_text = "\n".join(seg["text"] for seg in segments if seg["text"])
        elapsed = time.time() - t0
        actual_duration = duration_sec or 0.0
        rtf = elapsed / actual_duration if actual_duration else 0.0
        logger.info(
            "Transcription done: %d chars in %.1f sec (RTF=%.2f)", len(raw_text), elapsed, rtf
        )

        result = {
            "raw_text": raw_text,
            "segments": segments,
            "duration_sec": actual_duration,
        }
        self._save_cache(work_dir, audio_path, self.model_name, result)
        return TranscriptResult(
            raw_text=raw_text,
            segments=segments,
            duration_sec=actual_duration,
            cost_yuan=0.0,
        )

    def _build_segments(self, res, postprocess, add_punc: bool) -> list[dict]:
        if not res:
            return []
        r = res[0]
        sentences = r.get("sentence_info") or []
        if not sentences:
            # 退路：无时间戳句子时，把整段文本作为单段落
            full = r.get("text", "")
            text = postprocess(full)
            return [{"text": text, "start_time": 0.0, "end_time": 0.0}] if text else []

        words = r.get("words") or []
        timestamps = r.get("timestamp") or []
        has_word_ts = len(words) == len(timestamps) and len(words) > 0

        # FunASR 时间戳单位：实际为毫秒（实测 sentence_info start=400~30410，
        # words timestamp [3250,3310]），测试夹具用秒。统一换算为秒：>=1000 视为
        # 毫秒（1000ms=1s 是真实边界；1000 秒的单个音频段不存在，故用 >= 不漏 1000ms）。
        def to_sec(v) -> float:
            v = float(v)
            return v / 1000.0 if v >= 1000 else v

        cleaned = []
        for s in sentences:
            t = s.get("text")
            if t:
                cleaned.append(postprocess(t))
            elif has_word_ts:
                # spk_model 模式：sentence_info.text 为 None，文本在词级
                # timestamp/words 中，按时间窗归组。
                start_ms = float(s.get("start") or 0)
                end_ms = float(s.get("end") or 0)
                parts = []
                for w, (ts, te) in zip(words, timestamps):
                    if start_ms <= ts < end_ms:
                        parts.append(w)
                cleaned.append(postprocess("".join(parts)))
            else:
                cleaned.append("")

        if add_punc and self._punc_model is not None:
            # 逐条补标点（spk_model 模式下批量调用偶发失败，逐条更稳）；
            # 单条失败仅该条沿用无标点文本，不影响整体。
            for i, t in enumerate(cleaned):
                if not t:
                    continue
                try:
                    pr = self._punc_model.generate(input=[t])
                    if pr and isinstance(pr[0], dict) and pr[0].get("text"):
                        cleaned[i] = pr[0]["text"]
                except Exception as exc:
                    logger.warning("ct-punc 失败（第 %d 条），沿用无标点文本: %s", i, exc)

        segments = []
        for s, text in zip(sentences, cleaned):
            if not text:
                continue
            start = s.get("start")
            end = s.get("end")
            seg: dict = {
                "text": text,
                "start_time": to_sec(start) if start is not None else 0.0,
                "end_time": to_sec(end) if end is not None else 0.0,
            }
            spk = s.get("spk")
            if spk is not None:
                seg["speaker"] = f"SPEAKER_{int(spk):02d}"
            segments.append(seg)

        return self._merge_same_speaker(segments)

    @staticmethod
    def _merge_same_speaker(segments: list[dict], gap: float = 0.2) -> list[dict]:
        """合并相邻同说话人段（与 diarization.assign_speakers 同口径），
        避免 spk_model 按 VAD 段输出的碎片化标签。

        注意：只合并**同说话人**段，跨说话人绝不合并——混段错乱正是
        「两人被并进一段只给一个标签」造成的，收紧 gap（0.5s→0.2s）
        避免把短停顿的相邻段拼成更长的错误段。
        """
        if not segments or "speaker" not in segments[0]:
            return segments
        merged = []
        for seg in segments:
            if (
                merged
                and merged[-1].get("speaker") == seg.get("speaker")
                and seg["start_time"] - merged[-1]["end_time"] <= gap
            ):
                merged[-1]["text"] += " " + seg["text"]
                merged[-1]["end_time"] = seg["end_time"]
            else:
                merged.append(dict(seg))
        return merged
