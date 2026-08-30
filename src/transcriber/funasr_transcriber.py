"""Local ASR using FunASR (SenseVoice-Small / Paraformer-Large) on CPU.

Replaces the old faster-whisper backend. FunASR models ship their own VAD and
long-audio handling. Typical CPU speed: SenseVoice-Small ~9x real-time,
Paraformer-Large ~4x on the reference machine (i5-12500H).

Long-audio strategy (fixes the whole-audio ``MemoryError``):
  62+ min WAVs transcribed in one pass overflow the CPU path (subprocess
  stdout buffering accumulates). For audio longer than ``CHUNK_SEC`` (600s) we
  slice the WAV into fixed-length chunks (Python ``wave``, no ffmpeg), run
  ``model.generate`` per chunk, then re-align speaker labels across chunks.

Speaker diarization: SenseVoice loads with ``spk_model`` (cam++) so FunASR
returns ``sentence_info`` entries carrying a ``spk`` label plus word-level
``timestamp``/``words``; ``_build_segments`` folds the spk label into each
segment (``speaker`` key). This replaces the external pyannote-style diarize
step for the default model: labels come out of the same pass, no separate
audio round-trip needed. Paraformer keeps external diarization (no spk_model
in its FunASR output path).

Cross-chunk speaker consistency: ``SPEAKER_xx`` labels are local to each
``generate`` call, so chunk boundaries would silently swap identities. We
collect the longest utterance of every local speaker as a representative clip,
score all pairs with the cam++ SV model (``inference_sv``), greedily cluster
them into global speaker ids, and remap every segment. Without this, chunked
transcription of a 2-speaker show could interleave labels wrongly.

Segment granularity is controlled by ``spk_max_seg_ms`` (VAD 段上限，默认 4000ms):
cam++ 按 VAD 段聚类，段越长越容易把两人对话（快速问答、无缝接话）并成一段
只给一个标签，这正是 Vol.11 说话人错乱的根源。benchmark/vol11-asr-spk-comparison
实测 8000ms 档位 882 段 6 人核验正确；默认 4000ms 进一步降低段内混人概率，
若聚类质量下降可用 ``--spk-max-seg-ms 8000`` 回退实验档位。

Progress reporting: the transcriber writes ``work_dir/status.json`` at each
stage (start/loaded/transcribe x/y/align/done), so a long background run can
be inspected across sessions without relying on a live task id.
"""

import ctypes
import gc
import json
import logging
import os
import re
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.models.schemas import TranscriptResult
from src.transcriber.base import Transcriber

logger = logging.getLogger(__name__)

# 中文全角标签：SenseVoice 残留的情绪/事件标记 <...> / <|...|> 与中文全角 〈...〉
_TAG_RE = re.compile(r"<[^>]*>|<\|\^?[^>]*\|>|〈[^〉]*〉")

MODELSCOPE_CACHE = str(Path.home() / ".cache" / "modelscope")

# SenseVoice 说话人区分模型（FunASR spk_model，cam++ 声纹）
SENSEVOICE_SPK_MODEL = "iic/speech_campplus_sv_zh_en_16k-common_advanced"

# 长音频分段阈值（秒）：超过该时长的 WAV 走分段转写路径，规避整段一次性
# generate 在 CPU 下的 MemoryError（实测 62 分钟单集崩）。
CHUNK_SEC = 600  # 10 分钟
# 触发分段的时长下限系数：音频长于 CHUNK_SEC * CHUNK_MARGIN 才分段，
# 短音频仍走整段路径（无跨段对齐开销）。
CHUNK_MARGIN = 1.2
# cam++ 说话人相似度阈值：>= 该值判为同一人（官方常用 0.3）。
SV_THRESHOLD = 0.31
# 单个分块转写的偶发失败重试：指数退避；OOM 类错误不重试。
CHUNK_RETRIES = 3
CHUNK_RETRY_BASE_S = 2.0

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


def _strip_sensevoice_tags(text) -> str:
    """Remove SenseVoice emotion/event emoji tags and normalize whitespace.

    SenseVoice prepends language/emotion/event markers such as ``<|zh|>``,
    ``<HAPPY>`` or ``</emoji>``. ``rich_transcription_postprocess`` removes the
    ASCII ``<|...|>`` block but leaves the Chinese-angle ``<...>`` tags, so we
    strip both here.

    FunASR 偶发把单句文本返回成 list / None（如 spk_model 模式下个别
    sentence_info.text 为 list），这里先统一规整为 str，避免后续 .sub 在
    list 上抛 AttributeError 拖垮整段转写。
    """
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    elif text is None:
        text = ""
    else:
        text = str(text)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _audio_duration_sec(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _postprocess_sensevoice(raw_text) -> str:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    # FunASR 偶发把单句文本返回成 list / None（如 spk_model 模式下个别
    # sentence_info.text 为 list），rich_transcription_postprocess 只接受 str，
    # 直接传会 AttributeError: 'list' object has no attribute 'replace'。
    # 先规整为 str 再后处理，避免单集因个别分块崩掉。
    if isinstance(raw_text, list):
        raw_text = " ".join(str(x) for x in raw_text)
    elif raw_text is None:
        raw_text = ""
    else:
        raw_text = str(raw_text)
    return _strip_sensevoice_tags(rich_transcription_postprocess(raw_text))


_WORKER_TRANSCRIBER = None


def _init_chunk_worker(model_name: str, spk_max_seg_ms: int, batch_size_s: int, modelscope_cache: str) -> None:
    """进程池 worker 初始化：每个 worker 加载一份 FunASR 模型并常驻该进程。"""
    global _WORKER_TRANSCRIBER
    tr = FunASRTranscriber(
        model_name=model_name,
        spk_max_seg_ms=spk_max_seg_ms,
        batch_size_s=batch_size_s,
        modelscope_cache=modelscope_cache,
    )
    tr._ensure_loaded()
    _WORKER_TRANSCRIBER = tr


def _chunk_worker_transcribe(chunk_path: str, chunk_idx: int) -> list[dict]:
    """单个分块转写任务（在 worker 进程执行）；偶发错误指数退避重试，OOM 不重试。"""
    for attempt in range(1, CHUNK_RETRIES + 1):
        try:
            return _WORKER_TRANSCRIBER._transcribe_whole(Path(chunk_path))
        except MemoryError:
            raise
        except Exception as exc:
            if attempt >= CHUNK_RETRIES:
                raise
            wait = CHUNK_RETRY_BASE_S * (2 ** (attempt - 1))
            time.sleep(wait)


class FunASRTranscriber(Transcriber):
    """Transcribe locally using FunASR on CPU (SenseVoice-Small or Paraformer-Large)."""

    SUPPORTED = ("sensevoice-small", "paraformer-large")

    def __init__(
        self,
        model_name: str = "sensevoice-small",
        modelscope_cache: str | None = None,
        spk_max_seg_ms: int = 4000,
        batch_size_s: int = 60,
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
        # FunASR VAD 批切段时长（秒）：越大单次送入音频越长、调用次数越少，
        # 但峰值内存越高。本机内存紧张，默认保守取 60。
        self.batch_size_s = batch_size_s
        self._model = None
        self._punc_model = None
        self._punc_enabled = False
        self._punc_model_arg = None
        self._sv_model = None  # cam++ 说话人验证模型（仅跨段对齐懒加载）
        self._chunk_pool = None  # 并行分块转写的进程池（懒创建、常驻复用）
        self._chunk_pool_workers = 0
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
                # spk_model 模式内部依赖 punc_model 做句子切分（缺失时报
                # "Missing punc_model, which is required by spk_model" 且行为不稳定，
                # vol.231 实测逐 chunk 报错、阶段耗时翻倍）；与 paraformer 路径一致，
                # 显式传入，消除不确定的 fallback 行为。
                punc_model=punc,
                disable_update=True,
                ignore_instances=True,
            )
            # 标点模型与 SenseVoice 解耦，单独懒加载以便公平补齐标点
            # （见 _get_punc_model），避免启动时与组合内 punc 同时驻留两份。
            self._punc_enabled = True
            self._punc_model_arg = punc
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

    def _get_punc_model(self):
        """返回用于补齐标点的独立 ct-punc 模型；按需懒加载。

        主 SenseVoice 组合内部已加载一份 punc（spk_model 切句必需），这里
        再驻留一份仅用于「公平补齐标点」。为避免同模型双份常驻内存（本机
        内存紧张），独立副本改为首次需要时才加载。
        """
        if self._punc_model is not None:
            return self._punc_model
        if not self._punc_enabled or not self._punc_model_arg:
            return None
        from funasr import AutoModel

        logger.info("懒加载独立 ct-punc 标点模型...")
        self._punc_model = AutoModel(
            model=self._punc_model_arg,
            disable_update=True,
            ignore_instances=True,
        )
        return self._punc_model

    def _ensure_sv_loaded(self):
        """懒加载 cam++ 说话人验证模型（仅长音频跨段对齐需要时调用）。"""
        if self._sv_model is not None:
            return self._sv_model
        os.environ.setdefault("MODELSCOPE_CACHE", self._modelscope_cache)
        from funasr import AutoModel

        spk = self._resolve_model_arg(SENSEVOICE_SPK_MODEL, SENSEVOICE_SPK_MODEL)
        logger.info("Loading cam++ SV model for cross-chunk speaker alignment...")
        t0 = time.time()
        self._sv_model = AutoModel(
            model=spk,
            disable_update=True,
            ignore_instances=True,
        )
        logger.info("cam++ SV loaded in %.1f sec", time.time() - t0)
        return self._sv_model

    # --- progress status (written to work_dir/status.json, inspectable cross-session) ---
    def _write_status(self, work_dir, stage: str, progress: float, msg: str = "") -> None:
        if work_dir is None:
            return
        try:
            d = Path(work_dir)
            d.mkdir(parents=True, exist_ok=True)
            payload = {
                "stage": stage,
                "progress": round(float(progress), 3),
                "msg": msg,
                "ts": time.time(),
            }
            (d / "status.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("status write failed: %s", exc)

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
            # 分段阈值同样纳入指纹：调整 CHUNK_SEC 后旧结果作废。
            and meta.get("chunk_sec") == CHUNK_SEC
            # 批切段时长纳入指纹：改 batch_size_s 后旧结果作废。
            and meta.get("batch_size_s") == self.batch_size_s
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
            "chunk_sec": CHUNK_SEC,
            "batch_size_s": self.batch_size_s,
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        (cache_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )

    # --- 并行分块转写（B3）：多 worker 进程池，内存护栏 ---
    @staticmethod
    def _available_memory_gb() -> float | None:
        try:
            class _MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullAvailPhys / (1024 ** 3)
        except Exception:
            return None

    def _clamp_workers(self, workers: int) -> int:
        """按可用内存把并行 worker 数压到安全范围，避免 OOM/交换（宁慢不崩）。

        每 worker 估算 ~1GB 常驻（SenseVoice+punc+vad+cam++），要求可用内存的
        60% 至少能容纳；不足则降档并明确告警。
        """
        if workers <= 1:
            return workers
        avail = self._available_memory_gb()
        if avail is None:
            return workers
        est_per_worker = 1.0
        max_by_ram = max(1, int((avail * 0.6) / est_per_worker))
        if max_by_ram < workers:
            logger.warning(
                "可用内存 %.1fGB 不足支撑 %d 个转写 worker（估算每 worker 需 ~%.0fGB），"
                "自动降至 %d 个以避免 OOM/交换；关闭其他程序释放内存，或显式调低 --jobs",
                avail, workers, est_per_worker, max_by_ram,
            )
            return max_by_ram
        return workers

    def _get_chunk_pool(self, workers: int) -> ProcessPoolExecutor:
        """懒创建并行分块转写进程池；同档位 worker 复用（批量/常驻服务下模型只加载一次）。"""
        if self._chunk_pool is not None and self._chunk_pool_workers == workers:
            return self._chunk_pool
        if self._chunk_pool is not None:
            self._chunk_pool.shutdown(wait=True)
        self._chunk_pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_chunk_worker,
            initargs=(self.model_name, self.spk_max_seg_ms, self.batch_size_s, self._modelscope_cache),
        )
        self._chunk_pool_workers = workers
        return self._chunk_pool

    def shutdown(self) -> None:
        """释放并行转写进程池（worker 内的常驻模型随之退出）。"""
        if self._chunk_pool is not None:
            self._chunk_pool.shutdown(wait=True)
            self._chunk_pool = None
            self._chunk_pool_workers = 0

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _absorb_chunk(self, segs, cpath, chunk_idx, offset, work_dir, all_segments, rep_clips) -> None:
        """把一段转写结果并入全局：收集代表片段、叠加全局 offset、回收内存。

        代表片段必须在叠加 offset 前收集（此时时间戳为片内局部值），否则拿全局
        时间戳去切 10 分钟切片音频会越界（实测 wave.Error: position not in range）。
        """
        if self.model_name == "sensevoice-small":
            try:
                rep_clips.extend(self._collect_rep_clips(segs, cpath, chunk_idx, work_dir))
            except Exception as exc:
                logger.warning("第 %d 段代表片段收集失败，跳过该段跨段对齐: %s", chunk_idx + 1, exc)
        for s in segs:
            s["start_time"] += offset
            s["end_time"] += offset
            s["_chunk_idx"] = chunk_idx
        all_segments.extend(segs)
        # 及时回收本段中间张量与 Python 对象，压低长音频全程峰值内存
        gc.collect()

    # --- core ---
    def transcribe(self, audio_path, duration_sec=None, *, work_dir=None, jobs=1, num_speakers=None) -> TranscriptResult:
        audio_path = Path(audio_path)
        duration_sec = duration_sec or _audio_duration_sec(audio_path)
        self._write_status(work_dir, "start", 0.0, "开始转录")

        cached = self._load_cache(work_dir, audio_path, self.model_name)
        if cached is not None:
            logger.info("复用整段转写缓存（模型: %s）", self.model_name)
            self._write_status(work_dir, "done", 1.0, "命中转写缓存")
            return TranscriptResult(
                raw_text=cached["raw_text"],
                segments=cached["segments"],
                duration_sec=cached.get("duration_sec", duration_sec or 0.0),
                cost_yuan=0.0,
            )

        # 模型加载仅对「整段转写」或「串行分块」必需；并行分块由各 worker 进程
        # 各自加载模型，主进程不加载（省一份常驻内存）。
        t0 = time.time()
        logger.info("Transcribing (FunASR %s, CPU)...", self.model_name)
        # 长音频走分段路径（规避整段一次性 generate 的 CPU MemoryError）；
        # 短音频仍整段一次，省去切片与跨段对齐开销。
        if duration_sec and duration_sec > CHUNK_SEC * CHUNK_MARGIN:
            segments = self._transcribe_chunked(audio_path, duration_sec, work_dir, jobs, num_speakers=num_speakers)
        else:
            self._ensure_loaded()
            self._write_status(work_dir, "loaded", 0.05, "模型已加载")
            segments = self._transcribe_whole(audio_path)

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
        self._write_status(work_dir, "done", 1.0, "转录完成")
        return TranscriptResult(
            raw_text=raw_text,
            segments=segments,
            duration_sec=actual_duration,
            cost_yuan=0.0,
        )

    def _transcribe_whole(self, audio_path: Path) -> list[dict]:
        """整段一次性转写（短音频，原始逻辑）。返回 segments 列表。"""
        if self.model_name == "sensevoice-small":
            res = self._model.generate(
                input=str(audio_path),
                cache={},
                language="zh",
                use_itn=False,
                batch_size_s=self.batch_size_s,
                merge_vad=True,
                merge_length_s=15,
                spk_model=True,
                output_timestamp=True,
            )
            return self._build_segments(res, postprocess=_postprocess_sensevoice, add_punc=True)
        # paraformer-large
        res = self._model.generate(input=str(audio_path), batch_size_s=self.batch_size_s)
        return self._build_segments(res, postprocess=_strip_sensevoice_tags, add_punc=False)

    def _transcribe_chunked(self, audio_path: Path, duration_sec: float, work_dir, jobs: int = 1, num_speakers: int | None = None) -> list[dict]:
        """分段转写 + 跨段说话人全局对齐（长音频，修复整段一次性转写 OOM）。

        流程：
        1. 按 CHUNK_SEC 用 wave 按帧把 WAV 切成若干临时片段（不依赖 ffmpeg）；
        2. 逐片 model.generate(spk_model=True) —— 单片内存峰值可控；jobs>1 时
           用进程池并行（各 worker 自带模型），否则串行；
        3. 每片的 SPEAKER_xx 是片内局部编号，跨片会错位：收集每片每说话人的
           「最长 utterance」音频片段，用 cam++ SV 两两打分、贪心聚类，把局部
           标签重映射为全局标签；
        4. 恢复全局时间戳、按时间排序，最后 _merge_same_speaker 合并同说话人。
        """
        chunk_paths = self._slice_wav(audio_path, CHUNK_SEC, work_dir)
        total = len(chunk_paths)
        logger.info("长音频分段转写：共 %d 段（每段 %d s），worker=%d", total, CHUNK_SEC, jobs)
        self._write_status(work_dir, "transcribe", 0.1, f"音频已切分为 {total} 段")

        all_segments: list[dict] = []
        rep_clips: list[dict] = []  # 跨段说话人对齐用的代表片段

        if jobs > 1:
            self._transcribe_chunks_parallel(
                chunk_paths, audio_path, work_dir, all_segments, rep_clips, jobs
            )
        else:
            self._ensure_loaded()
            self._write_status(work_dir, "loaded", 0.05, "模型已加载")
            for i, (cpath, offset) in enumerate(chunk_paths, 1):
                cached_chunk = self._load_chunk_cache(work_dir, audio_path, i - 1, cpath)
                if cached_chunk is not None:
                    segs = cached_chunk
                    logger.info("复用第 %d/%d 段转写缓存", i, total)
                else:
                    segs = self._transcribe_chunk_with_retry(cpath, i - 1)
                    self._save_chunk_cache(work_dir, audio_path, i - 1, cpath, segs)
                self._absorb_chunk(segs, cpath, i - 1, offset, work_dir, all_segments, rep_clips)
                self._write_status(
                    work_dir, "transcribe", 0.1 + 0.8 * i / total, f"转写 {i}/{total} 段"
                )

        if rep_clips:
            self._write_status(work_dir, "align", 0.92, "跨段说话人对齐中")
            mapping = self._align_speakers(rep_clips, num_speakers)
            for s in all_segments:
                spk = s.get("speaker")
                if spk and "_chunk_idx" in s:
                    local = int(spk.split("_")[1])
                    gid = mapping.get((s["_chunk_idx"], local))
                    if gid is not None:
                        s["speaker"] = f"SPEAKER_{gid:02d}"
                s.pop("_chunk_idx", None)

        all_segments.sort(key=lambda s: s["start_time"])
        merged = self._merge_same_speaker(all_segments)
        # 保留 chunk 和中间结果，便于失败后断点续跑与诊断。
        return merged

    def _transcribe_chunks_parallel(
        self, chunk_paths, audio_path, work_dir, all_segments, rep_clips, jobs
    ) -> None:
        """并行转写所有未命中缓存的分块（进程池），完成即并入全局结果。"""
        total = len(chunk_paths)
        pool = self._get_chunk_pool(self._clamp_workers(jobs))
        pending: dict = {}
        # 先消化命中缓存的分块（不占 worker）
        for i, (cpath, offset) in enumerate(chunk_paths, 1):
            cached_chunk = self._load_chunk_cache(work_dir, audio_path, i - 1, cpath)
            if cached_chunk is not None:
                segs = cached_chunk
                logger.info("复用第 %d/%d 段转写缓存", i, total)
                self._absorb_chunk(segs, cpath, i - 1, offset, work_dir, all_segments, rep_clips)
                self._write_status(
                    work_dir, "transcribe", 0.1 + 0.8 * i / total, f"转写 {i}/{total} 段"
                )
            else:
                fut = pool.submit(_chunk_worker_transcribe, str(cpath), i - 1)
                pending[fut] = (i - 1, cpath, offset)
        done = len(chunk_paths) - len(pending)
        for fut in as_completed(pending):
            ci, cpath, offset = pending[fut]
            segs = fut.result()  # worker 内已做指数退避重试
            self._save_chunk_cache(work_dir, audio_path, ci, cpath, segs)
            self._absorb_chunk(segs, cpath, ci, offset, work_dir, all_segments, rep_clips)
            done += 1
            self._write_status(
                work_dir, "transcribe", 0.1 + 0.8 * done / total, f"转写 {done}/{total} 段"
            )

    def _transcribe_chunk_with_retry(self, chunk_path: Path, chunk_idx: int) -> list[dict]:
        """转写单个分块；对偶发异常指数退避重试，OOM 类不重试。

        一个 10 分钟分块的 model.generate 偶发失败（临时资源争抢、Defender
        干扰等）不应让整期转录失败。内存类错误重试无意义且会加剧内存压力，
        直接抛出；其余错误退避重试有限次。
        """
        for attempt in range(1, CHUNK_RETRIES + 1):
            try:
                return self._transcribe_whole(chunk_path)
            except MemoryError:
                raise
            except Exception as exc:
                if attempt >= CHUNK_RETRIES:
                    raise
                wait = CHUNK_RETRY_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "分块 %d 转写失败（第 %d/%d 次尝试），%.0fs 后重试: %s",
                    chunk_idx, attempt, CHUNK_RETRIES, wait, exc,
                )
                time.sleep(wait)

    def _chunk_cache_path(self, work_dir, chunk_idx: int) -> Path:
        return Path(work_dir) / "transcribe" / f"chunk_{chunk_idx:04d}.json"

    def _load_chunk_cache(self, work_dir, audio_path: Path, chunk_idx: int, chunk_path: Path):
        p = self._chunk_cache_path(work_dir, chunk_idx)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            src = audio_path.stat()
            # 仅用音频「体积」校验，不用修改时间：重下载/重转码后 mtime 会变，
            # 但同来源音频体积稳定、内容一致；否则每次被宿主回收后续跑都因 mtime
            # 不匹配而从头重转，长单集永远卡在后台进程寿命内、无法收敛。
            if (
                data.get("audio_size") == src.st_size
                and data.get("chunk_idx") == chunk_idx
                and chunk_path.exists()
                and data.get("model_name") == self.model_name
                and data.get("spk_max_seg_ms") == self.spk_max_seg_ms
                and data.get("batch_size_s") == self.batch_size_s
                and isinstance(data.get("segments"), list)
            ):
                return data["segments"]
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return None

    def _save_chunk_cache(self, work_dir, audio_path: Path, chunk_idx: int, chunk_path: Path, segments):
        p = self._chunk_cache_path(work_dir, chunk_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        src = audio_path.stat()
        payload = {
            "audio_size": src.st_size,
            "audio_mtime": int(src.st_mtime),
            "chunk_idx": chunk_idx,
            "model_name": self.model_name,
            "spk_max_seg_ms": self.spk_max_seg_ms,
            "batch_size_s": self.batch_size_s,
            "segments": segments,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _slice_wav(self, src: Path, chunk_sec: int, work_dir) -> list[tuple[Path, float]]:
        """用 wave 按帧把 WAV 切成 chunk_sec 秒的片段。

        Returns list of (chunk_path, offset_sec) relative to the source start.
        """
        chunk_dir = Path(work_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out: list[tuple[Path, float]] = []
        with wave.open(str(src), "rb") as wf:
            fr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames_per_chunk = int(chunk_sec * fr)
            idx = 0
            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break
                cnt = len(frames) // (ch * sw)
                p = chunk_dir / f"chunk_{idx:03d}.wav"
                with wave.open(str(p), "wb") as ow:
                    ow.setnchannels(ch)
                    ow.setsampwidth(sw)
                    ow.setframerate(fr)
                    ow.writeframes(frames)
                out.append((p, idx * chunk_sec))
                idx += 1
                if cnt < frames_per_chunk:
                    break
        return out

    def _collect_rep_clips(self, segs: list[dict], chunk_path: Path, chunk_idx: int, work_dir) -> list[dict]:
        """对每片每个局部说话人，取最长 utterance 的音频片段作为代表（供跨段对齐）。"""
        best: dict[int, dict] = {}
        for s in segs:
            spk = s.get("speaker")
            if not spk:
                continue
            local = int(spk.split("_")[1])
            dur = s["end_time"] - s["start_time"]
            if local not in best or dur > best[local]["_dur"]:
                best[local] = {**s, "_dur": dur}
        if not best:
            return []
        rep_dir = Path(work_dir) / "rep_clips"
        rep_dir.mkdir(parents=True, exist_ok=True)
        reps: list[dict] = []
        with wave.open(str(chunk_path), "rb") as wf:
            fr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            for local, s in best.items():
                start_f = max(0, int(s["start_time"] * fr))
                end_f = int(s["end_time"] * fr)
                # 边界保护：时间戳异常时夹到文件帧范围内，避免 setpos 越界
                n_frames = wf.getnframes()
                start_f = min(start_f, n_frames)
                end_f = min(max(end_f, start_f), n_frames)
                if end_f <= start_f:
                    continue
                wf.setpos(start_f)
                frames = wf.readframes(end_f - start_f)
                p = rep_dir / f"rep_{chunk_idx:03d}_{local:02d}.wav"
                with wave.open(str(p), "wb") as ow:
                    ow.setnchannels(ch)
                    ow.setsampwidth(sw)
                    ow.setframerate(fr)
                    ow.writeframes(frames)
                reps.append({"audio": str(p), "key": (chunk_idx, local)})
        return reps

    def _align_speakers(self, reps: list[dict], num_speakers: int | None = None) -> dict[tuple[int, int], int]:
        """用 cam++ SV 对代表片段两两打分，聚类为全局说话人。

        - ``num_speakers`` 为 None：保持原贪心 + ``SV_THRESHOLD`` 行为（自动检测，
          碎几类算几类）。
        - ``num_speakers`` 已知（如来自 Show Notes）：在相似度矩阵上做 K-medoids
          **强制聚成 K 类**，根治「两人对话被切成 14 个伪说话人」的过度切分
          （vol.231 根因）。cam++ 把同一人两段打成 0.28（< 门槛）也能被强行归并。

        Returns {(chunk_idx, local_spk): global_id}.
        """
        if not reps:
            return {}
        sv = self._ensure_sv_loaded()
        n = len(reps)
        # 两两相似度矩阵（相似度越高越可能是同一人）
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim[i][i] = 1.0
            for j in range(i + 1, n):
                try:
                    res = sv.inference_sv(reps[i]["audio"], reps[j]["audio"])
                    s = self._sv_score(res)
                except Exception as exc:
                    logger.debug("inference_sv failed: %s", exc)
                    s = -1.0
                if s < 0:
                    s = 0.0
                sim[i][j] = sim[j][i] = s

        if num_speakers is None:
            return self._greedy_cluster(reps, sim)
        k = max(1, min(int(num_speakers), n))
        if k >= n:
            # 已知人数 >= 代表片段数，直接 1:1 映射
            return {rep["key"]: gid for gid, rep in enumerate(reps)}
        assign = self._kmedoids(sim, k)
        mapping: dict[tuple[int, int], int] = {}
        for idx, rep in enumerate(reps):
            mapping[rep["key"]] = assign[idx]
        logger.info(
            "跨段说话人对齐（强制 %d 类）：%d 个全局说话人（来自 %d 个代表片段）",
            k, len(set(assign)), n,
        )
        return mapping

    @staticmethod
    def _greedy_cluster(reps: list[dict], sim: list[list[float]]) -> dict[tuple[int, int], int]:
        """原行为：贪心 + SV_THRESHOLD，同人归并、否则开新簇。"""
        clusters: list[dict] = []
        for idx, rep in enumerate(reps):
            best_score = -1.0
            best_cluster = None
            for c in clusters:
                s = sim[c["idx"]][idx]
                if s > best_score:
                    best_score = s
                    best_cluster = c
            if best_cluster is not None and best_score >= SV_THRESHOLD:
                best_cluster["members"].append(rep["key"])
            else:
                clusters.append({"idx": idx, "members": [rep["key"]]})
        mapping: dict[tuple[int, int], int] = {}
        for gid, c in enumerate(clusters):
            for key in c["members"]:
                mapping[key] = gid
        return mapping

    @staticmethod
    def _kmedoids(sim: list[list[float]], k: int, max_iter: int = 25) -> list[int]:
        """在相似度矩阵上做 K-medoids（相似度越高越「近」）。返回每点的簇号。"""
        n = len(sim)
        # 初始化：max-min 选 k 个彼此最不相似的 medoids（相似度最小）
        medoids = [0]
        while len(medoids) < k:
            cand = min(
                (i for i in range(n) if i not in medoids),
                key=lambda x: max(sim[x][m] for m in medoids),
                default=None,
            )
            if cand is None:
                break
            medoids.append(cand)
        for _ in range(max_iter):
            assign = [max(range(k), key=lambda c: sim[i][medoids[c]]) for i in range(n)]
            new_medoids = []
            for c in range(k):
                members = [i for i in range(n) if assign[i] == c]
                if not members:
                    new_medoids.append(medoids[c])
                    continue
                new_medoids.append(max(members, key=lambda i: sum(sim[i][j] for j in members)))
            if new_medoids == medoids:
                break
            medoids = new_medoids
        return [max(range(k), key=lambda c: sim[i][medoids[c]]) for i in range(n)]

    @staticmethod
    def _sv_score(res) -> float:
        """从 FunASR inference_sv 的返回里稳健地取出相似度分数。"""
        if not res:
            return -1.0
        if isinstance(res, list):
            res = res[0] if res else {}
        scores = res.get("scores") or res.get("score")
        if scores is None:
            return -1.0
        if isinstance(scores, (list, tuple)):
            return float(scores[0]) if scores else -1.0
        return float(scores)

    def _cleanup_chunks(self, work_dir) -> None:
        import shutil

        for name in ("chunks", "rep_clips"):
            p = Path(work_dir) / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

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

        punc_model = self._get_punc_model()
        if add_punc and punc_model is not None:
            # 优先批量补标点，失败时仅退回逐条处理，避免单条失败拖垮整段。
            nonempty = [(i, t) for i, t in enumerate(cleaned) if t]
            if nonempty:
                try:
                    results = punc_model.generate(input=[t for _, t in nonempty])
                    if len(results) != len(nonempty):
                        raise RuntimeError("ct-punc 返回数量与输入不一致")
                    for (i, _), item in zip(nonempty, results):
                        if isinstance(item, dict) and item.get("text"):
                            cleaned[i] = item["text"]
                except Exception as exc:
                    logger.warning("ct-punc 批量调用失败，退回逐条处理: %s", exc)
                    for i, t in nonempty:
                        try:
                            pr = punc_model.generate(input=[t])
                            if pr and isinstance(pr[0], dict) and pr[0].get("text"):
                                cleaned[i] = pr[0]["text"]
                        except Exception as item_exc:
                            logger.warning("ct-punc 失败（第 %d 条），沿用无标点文本: %s", i, item_exc)

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
