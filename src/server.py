"""撷声常驻转写服务。

把 FunASR 模型常驻内存、通过本地 HTTP 提供转写，避免每次 CLI 冷启动重载
4 个模型（本机模型加载受 Defender 实时扫描拖累，可达 ~20min）。

用法：
    启动常驻服务（模型只加载一次，之后常驻）：
        xiesheng --server [--server-port 8765]

    CLI 走 HTTP 转写（不再本地加载模型，直接发请求）：
        xiesheng <url> --use-server http://127.0.0.1:8765

HTTP 接口：
    GET  /health       -> {"status":"ok","model":...}
    POST /transcribe   -> body {"audio": "/path/to.wav",
                                "duration_sec": <float|None>,
                                "work_dir": "/path/to/.work/<ep>"}
                          返回 {"raw_text", "segments", "duration_sec", "cost_yuan"}

转写期间的进度照常写入 work_dir/status.json（与本地转写同路径），可跨会话监控。
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.models.schemas import TranscriptResult
from src.transcriber.base import Transcriber

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# 常驻转写器非线程安全：并发请求共享同一 FunASR 实例，串行化避免竞态
# （含 _align_speakers 内 SV 模型懒加载）；当前 CLI 单客户端，此为防御性保护。
_TRANSCRIBE_LOCK = threading.Lock()


class HttpTranscriber(Transcriber):
    """本地转写器的 HTTP 客户端代理：签名与 FunASRTranscriber 一致，调用常驻服务。"""

    def __init__(self, base_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}", model_name: str = "sensevoice-small", timeout: int = 7200):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def transcribe(self, audio_path, duration_sec=None, *, work_dir=None, jobs=1) -> TranscriptResult:
        import requests

        payload = {
            "audio": str(audio_path),
            "duration_sec": duration_sec,
            "work_dir": str(work_dir) if work_dir else None,
        }
        resp = requests.post(f"{self.base_url}/transcribe", json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"转写服务错误 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return TranscriptResult(
            raw_text=data["raw_text"],
            segments=data["segments"],
            duration_sec=data.get("duration_sec", duration_sec or 0.0),
            cost_yuan=0.0,
        )

    def health(self) -> dict:
        import requests

        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()


class _Handler(BaseHTTPRequestHandler):
    """转写服务 HTTP handler。transcriber 由 start_server 注入类属性。"""

    transcriber = None  # FunASRTranscriber 实例（常驻）

    # --- routes ---
    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/health/"):
            model = self.transcriber.model_name if self.transcriber is not None else "not-loaded"
            self._json(200, {"status": "ok", "model": model})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/transcribe":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        audio = payload.get("audio")
        if not audio or not os.path.exists(audio):
            self._json(400, {"error": "audio path missing or not found"})
            return
        try:
            with _TRANSCRIBE_LOCK:
                result = self.transcriber.transcribe(
                    Path(audio),
                    duration_sec=payload.get("duration_sec"),
                    work_dir=Path(payload["work_dir"]) if payload.get("work_dir") else None,
                )
            self._json(200, {
                "raw_text": result.raw_text,
                "segments": result.segments,
                "duration_sec": result.duration_sec,
                "cost_yuan": result.cost_yuan,
            })
        except Exception as exc:
            logger.exception("transcribe failed")
            self._json(500, {"error": str(exc)})

    # --- helpers ---
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("[server] %s", fmt % args)


def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: str = "sensevoice-small",
    spk_max_seg_ms: int = 4000,
    batch_size_s: int = 60,
) -> None:
    """加载模型并启动常驻转写服务（阻塞运行，Ctrl+C 停止）。"""
    from src.transcriber.funasr_transcriber import FunASRTranscriber

    logger.info("常驻服务加载模型 %s（CPU）...", model_name)
    transcriber = FunASRTranscriber(
        model_name=model_name,
        spk_max_seg_ms=spk_max_seg_ms,
        batch_size_s=batch_size_s,
    )
    transcriber._ensure_loaded()
    _Handler.transcriber = transcriber

    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"\n常驻转写服务已启动: http://{host}:{port}/health")
    print(f"  模型: {model_name}（已加载，常驻内存）")
    print(f"  CLI 对接: xiesheng <url> --use-server http://{host}:{port}")
    print("  Ctrl+C 停止\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务停止")
    finally:
        srv.server_close()
