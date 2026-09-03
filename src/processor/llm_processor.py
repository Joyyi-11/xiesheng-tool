"""LLM-assisted transcript cleaning and structuring."""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

from src.config import LLMConfig
from src.diarization.speaker_resolver import to_display_label
from src.models.llm_struct import parse_struct
from src.models.schemas import KeyPoint, Keyword, OutputDoc, QuestionItem, TermDef
from src.processor.prompt import (
    CLEAN_SYSTEM_PROMPT,
    CLEAN_USER_PROMPT,
    STRUCT_SYSTEM_PROMPT,
    STRUCT_USER_PROMPT,
)

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 8_000
CONTEXT_CHARS = 500
CHUNK_CONCURRENCY = 4  # 分块校订并行度；并发过多易触发限流
MIN_CLEAN_RATIO = 0.55
MAX_CLEAN_RATIO = 1.35
# 修改 CLEAN 提示词或分块/对齐逻辑时递增，使旧校订分块缓存作废。
PROMPT_VERSION = 4
# 结构化（summary+keywords+outline 合并）提示词版本：改动提示词或解析时递增。
# v9：questions.question 明确为纯文本（禁止自带 ** 加粗），渲染侧 markdown.py 统一
#     输出 `1. **问题？**`（序号在加粗外），与 session 路径 v18 规则对齐。
# v8：核心观点改为「观点句（句号结尾）+ 展开论述句 + 引用句」顺次相接，禁止 point 以
#     冒号收尾引出 evidence；渲染侧由 markdown.py 确定性去冒号并补句末标点（v17 结构）。
# v7：CLEAN 提示词 rule 2 扩展「断句」覆盖句间过度切分（相邻两句本应逗号连读却被句号拆开），与 session 路径校订规则第 1 条 (b) 对齐。
# v6：问题与思考要求强化——answer 须直接解答问题（结尾不得反抛新问号），引用人物须首次点明「姓名（身份）」、禁止裸代词；问题须提炼全文核心议题。
# v5：核心观点原话改行内直角引号（不再单独引用块）；新增 key_points.terms 就近解释观点内特有术语；
# keywords 改为只放残留术语（0-4 个）、排除播客名/节目名/嘉宾名/平台名等专名。
# v4：speaker_mapping 增加「多人混段不映射、保留 [SPEAKER_XX]」约束（Vol.11 错乱修复）。
STRUCT_PROMPT_VERSION = 9

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)$")


class ModelNotFoundError(RuntimeError):
    """The requested model is not served by the configured gateway (permanent)."""


class LLMUnavailableError(RuntimeError):
    """No usable model could be resolved for the configured gateway."""


# 最小探测请求：验证网关是否真的提供某模型，不依赖 /models 列表端点。
_PROBE_SYSTEM = "You are a helpful assistant. Reply with a single short word."
_PROBE_USER = "ping"


def probe_available_models(client: OpenAI) -> list[str]:
    """Return the model IDs a gateway advertises via ``GET /models``.

    Returns an empty list when the gateway does not expose the endpoint
    (many OpenAI-compatible gateways do); callers should then fall back to
    ``_probe_model``-based live probing.
    """
    try:
        page = client.models.list()
        return sorted({m.id for m in page.data})
    except Exception as exc:
        logger.info("Gateway does not expose /models; will live-probe instead: %s", exc)
        return []


def _probe_model(client: OpenAI, model: str) -> tuple[bool, Exception | None]:
    """Single minimal request to check whether ``model`` is actually served.

    Returns ``(usable, error)``. Does not retry — the caller decides how to
    treat a transient failure.
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PROBE_SYSTEM},
                {"role": "user", "content": _PROBE_USER},
            ],
            max_tokens=8,
            temperature=0,
        )
        return True, None
    except Exception as exc:
        return False, exc


def _resolve_effective_model(
    client: OpenAI,
    model: str,
    fallback_models: tuple[str, ...] = (),
) -> str:
    """Pick the first configured candidate the gateway actually serves.

    Strategy: check the advertised ``/models`` list first; if that is
    unavailable, live-probe each candidate with a minimal request. Candidates
    are the primary model followed by ``fallback_models`` in order, deduplicated.
    Raises ``LLMUnavailableError`` when none are usable.
    """
    candidates = list(dict.fromkeys([model, *fallback_models]))
    advertised = probe_available_models(client)
    if advertised:
        for cand in candidates:
            if cand in advertised:
                return cand
        raise LLMUnavailableError(
            f"配置模型 {candidates} 均不在网关可用列表（{advertised}）中"
        )
    for cand in candidates:
        usable, exc = _probe_model(client, cand)
        if usable:
            return cand
        if _is_model_not_found(exc):
            logger.info("Model %s not served by gateway; trying next candidate", cand)
            continue
        # 连接/过载等瞬时错误：无法确认模型不可用，返回主候选，交由请求层退避重试。
        logger.info("Probe for %s failed (transient?): %s", cand, exc)
        return cand
    raise LLMUnavailableError(f"没有可用模型：{candidates}")


def resolve_llm_config(llm_config: LLMConfig, client: OpenAI | None = None) -> LLMConfig:
    """Return a copy of ``llm_config`` whose ``model`` is actually usable.

    A one-time, low-cost call (either ``GET /models`` or a minimal probe
    request) that pins the gateway's real model before the expensive chunk
    cleaning starts, so a 404 ``model_not_found`` never burns token/quota.
    """
    if client is None:
        client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    effective = _resolve_effective_model(client, llm_config.model, llm_config.fallback_models)
    if effective != llm_config.model:
        logger.info("LLM model %s -> %s (gateway fallback)", llm_config.model, effective)
        return replace(llm_config, model=effective)
    return llm_config


def process(
    title: str,
    podcast_name: str,
    pub_date: str,
    show_notes: str,
    transcript_text: str,
    *,
    llm_config: "LLMConfig",
    work_dir: Path | None = None,
    segments: list[dict] | None = None,
) -> tuple[OutputDoc, int, int]:
    """Clean a transcript in chunks, then generate compact reading aids.

    Args:
        llm_config: LLM connection/model settings (api_key, base_url, provider, model).
        segments: Optional speaker-labeled utterance segments
            ([{"start", "end", "speaker", "text"}]) used to attach timestamps and
            speakers to highlight quotes. Plain ASR segments (with
            "start_time"/"end_time") are also accepted.

    Returns:
        (OutputDoc, input_tokens, output_tokens)
    """
    # 先固定网关实际可用的模型，避免 404 白白消耗配额（会先读 /models 或做最小探测请求）
    llm_config = resolve_llm_config(llm_config)
    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    provider = llm_config.provider
    model = llm_config.model
    chunks = split_transcript(transcript_text)
    input_tokens = 0
    output_tokens = 0

    logger.info("Cleaning transcript with %s/%s in %d chunks", provider, model, len(chunks))
    notes_key = hashlib.sha256((show_notes or "").encode("utf-8")).hexdigest()[:16]

    cleaned_by_id: dict[int, str] = {}
    failed: list[tuple[int, Exception]] = []
    workers = min(CHUNK_CONCURRENCY, max(1, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _clean_chunk_worker,
                client=client,
                chunks=chunks,
                index=index,
                show_notes=show_notes or "",
                notes_key=notes_key,
                provider=provider,
                model=model,
                work_dir=work_dir,
            ): index
            for index in range(len(chunks))
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result, inp, out = future.result()
                input_tokens += inp
                output_tokens += out
                cleaned_by_id[index] = result
            except Exception as exc:  # 不因单个分块失败而中断整条链
                failed.append((index, exc))
                logger.warning("Chunk %d/%d failed cleaning; continuing without it: %s",
                               index + 1, len(chunks), exc)

    for index, exc in failed:
        if index not in cleaned_by_id:
            # 该块既无缓存时也退回原稿，保证全文信息不丢失
            cleaned_by_id[index] = chunks[index]

    # 按原始顺序重新拼接，保证输出稳定（与并行完成顺序无关）
    cleaned_chunks = [cleaned_by_id[i] for i in range(len(chunks))]
    cleaned_transcript = "\n\n".join(cleaned_chunks)
    struct, inp, out = _generate_struct(
        client,
        model,
        title,
        podcast_name,
        show_notes or "（无）",
        cleaned_transcript,
        work_dir=work_dir,
        provider=provider,
    )
    input_tokens += inp
    output_tokens += out
    doc = _build_output_doc(struct, title, podcast_name, pub_date, show_notes, cleaned_transcript)
    if failed:
        chunk_ids = ", ".join(str(index + 1) for index, _ in sorted(failed))
        doc.warnings.append(f"分块 {chunk_ids} 校订失败，已回退原稿，需人工复核")
    return doc, input_tokens, output_tokens


def split_transcript(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split text at existing line boundaries without losing content."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]

    chunks: list[str] = []
    current = ""
    for line in lines:
        while len(line) > max_chars:
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.append(line[:max_chars].rstrip())
            line = line[max_chars:]
        if current and len(current) + len(line) > max_chars:
            chunks.append(current.rstrip())
            current = ""
        current += line
    if current or not chunks:
        chunks.append(current.rstrip())
    return chunks


def _clean_chunk_worker(
    *,
    client: OpenAI,
    chunks: list[str],
    index: int,
    show_notes: str,
    notes_key: str,
    provider: str,
    model: str,
    work_dir: Path | None,
) -> tuple[str, int, int]:
    """Clean a single chunk (with cache lookup + validation retry).

    Returns ``(cleaned_text, input_tokens, output_tokens)``.

    Raises on persistent failure so the caller can soft-skip this chunk and
    continue the rest of the pipeline (cache is never corrupted by a failure).
    """
    chunk_id = index + 1
    chunk = chunks[index]
    cached = _load_cached_chunk(work_dir, chunk_id, chunk, provider, model, show_notes_key=notes_key)
    if cached is not None:
        logger.info("Using cached cleaned chunk %d/%d", chunk_id, len(chunks))
        return cached, 0, 0

    previous_context = chunks[index - 1][-CONTEXT_CHARS:] if index else "（无）"
    next_context = chunks[index + 1][:CONTEXT_CHARS] if index + 1 < len(chunks) else "（无）"
    # Show Notes 仅用于核对专名，重复发送全部 chunk 会浪费大量 token；
    # 只在靠前的 chunk 携带完整 Notes 即可建立专名基线。
    prompt_show_notes = show_notes[:6_000] if index < 2 else ""
    prompt = CLEAN_USER_PROMPT.format(
        show_notes=prompt_show_notes or "（无）",
        previous_context=previous_context,
        chunk_id=chunk_id,
        chunk_text=chunk,
        next_context=next_context,
    )

    cleaned = ""
    total_in = total_out = 0
    last_error: Exception | None = None
    for validation_attempt in range(2):
        try:
            data, inp, out = _request_json(client, model, CLEAN_SYSTEM_PROMPT, prompt, max_tokens=16_384)
            total_in += inp
            total_out += out
            cleaned = _validate_cleaned_chunk(data, chunk_id, chunk)
            break
        except Exception as exc:
            last_error = exc
            if validation_attempt == 0:
                logger.warning("Cleaned chunk %d failed; retrying once: %s", chunk_id, exc)
    if not cleaned:
        raise RuntimeError(f"chunk {chunk_id} cleaning failed: {last_error}")
    _save_cached_chunk(work_dir, chunk_id, chunk, cleaned, provider, model, show_notes_key=notes_key)
    return cleaned, total_in, total_out


MAX_LLM_ATTEMPTS = 4
_LLM_RETRY_BASE_SEC = 1.0
_LLM_RETRY_MAX_SEC = 12.0


def _request_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
) -> tuple[dict, int, int]:
    last_error: Exception | None = None
    use_json_mode = True
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"LLM response was incomplete: finish_reason={choice.finish_reason}")
            content = choice.message.content or ""
            data = json.loads(content)
            if not isinstance(data, dict):
                raise RuntimeError("LLM JSON response must be an object")
            usage = response.usage
            return (
                data,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            last_error = exc
            # 模型不存在是永久性错误：不重试、不降级 JSON 模式，立即抛出让上层换模型。
            if _is_model_not_found(exc):
                raise ModelNotFoundError(str(exc)) from exc
            # 部分 OpenAI 兼容网关不接受 response_format，降级为纯提示词约束 JSON。
            if use_json_mode and isinstance(exc, openai.BadRequestError):
                use_json_mode = False
                logger.warning("provider 不支持 json_object，降级为提示词约束 JSON: %s", exc)
                continue
            # 限流/连接类错误值得退避重试（如 429、APIConnectionError）
            if _is_rate_limit_error(exc):
                delay = min(_LLM_RETRY_MAX_SEC, _LLM_RETRY_BASE_SEC * (2**attempt))
                logger.warning("LLM rate-limited/overloaded; retrying in %.1fs: %s", delay, exc)
                time.sleep(delay)
                continue
            if attempt < MAX_LLM_ATTEMPTS - 1:
                logger.warning("LLM request failed; retrying: %s", exc)
    assert last_error is not None
    raise last_error


def _request_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Request plain text with the same completion/retry contract as JSON mode."""
    last_error: Exception | None = None
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"LLM response was incomplete: finish_reason={choice.finish_reason}")
            content = choice.message.content or ""
            if not content.strip():
                raise RuntimeError("LLM returned empty text")
            usage = response.usage
            return (
                content,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            last_error = exc
            if _is_model_not_found(exc):
                raise ModelNotFoundError(str(exc)) from exc
            if _is_rate_limit_error(exc):
                delay = min(_LLM_RETRY_MAX_SEC, _LLM_RETRY_BASE_SEC * (2**attempt))
                logger.warning("LLM rate-limited/overloaded; retrying in %.1fs: %s", delay, exc)
                time.sleep(delay)
                continue
            if attempt < MAX_LLM_ATTEMPTS - 1:
                logger.warning("LLM request failed; retrying: %s", exc)
    assert last_error is not None
    raise last_error


def _is_model_not_found(exc: Exception) -> bool:
    """True when the error means the model name itself is not served (permanent)."""
    if isinstance(exc, openai.NotFoundError):
        return True
    text = str(exc).lower()
    for marker in (
        "model_not_found",
        "model not found",
        "no such model",
        "invalid model",
        "unknown model",
        "the model does not exist",
    ):
        if marker in text:
            return True
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """True when the exception is a 429 / server-overload and worth backing off."""
    if isinstance(exc, openai.RateLimitError):
        return True
    text = str(exc).lower()
    for marker in (
        "429",
        "rate limit",
        "too many requests",
        "overloaded",
        "temporarily unavailable",
        "500",
        "502",
        "503",
    ):
        if marker in text:
            return True
    return False


def _validate_cleaned_chunk(data: dict, expected_id: int, source: str) -> str:
    if data.get("chunk_id") != expected_id:
        raise RuntimeError(f"LLM returned the wrong chunk_id: expected {expected_id}")
    cleaned = data.get("cleaned_text")
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise RuntimeError(f"LLM returned an empty cleaned_text for chunk {expected_id}")
    ratio = len(cleaned) / max(len(source), 1)
    if not MIN_CLEAN_RATIO <= ratio <= MAX_CLEAN_RATIO:
        raise RuntimeError(
            f"Cleaned chunk {expected_id} has suspicious length ratio {ratio:.2f} "
            f"(expected {MIN_CLEAN_RATIO:.2f}-{MAX_CLEAN_RATIO:.2f})"
        )
    return cleaned.strip()


def _chunk_cache_path(work_dir: Path | None, chunk_id: int) -> Path | None:
    return work_dir / f"chunk_{chunk_id:03d}.json" if work_dir else None


def _load_cached_chunk(
    work_dir: Path | None,
    chunk_id: int,
    source: str,
    provider: str,
    model: str,
    show_notes_key: str = "",
) -> str | None:
    path = _chunk_cache_path(work_dir, chunk_id)
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        data.get("source_sha256") == expected_hash
        and data.get("provider") == provider
        and data.get("model") == model
        and data.get("prompt_version") == PROMPT_VERSION
        and data.get("notes_key", "") == show_notes_key
        and isinstance(data.get("cleaned_text"), str)
    ):
        return data["cleaned_text"]
    return None


def _save_cached_chunk(
    work_dir: Path | None,
    chunk_id: int,
    source: str,
    cleaned: str,
    provider: str,
    model: str,
    show_notes_key: str = "",
) -> None:
    path = _chunk_cache_path(work_dir, chunk_id)
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "notes_key": show_notes_key,
        "cleaned_text": cleaned,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_start_sec(value: Any) -> float | None:
    """Parse a timeline start timestamp into seconds, or None if unparseable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        text = value.strip()
        m = _TIME_RE.match(text)
        if m:
            h, mnt, s = m.groups()
            return float((int(h) if h else 0) * 3600 + int(mnt) * 60 + int(s))
        try:
            return max(0.0, float(text))
        except ValueError:
            return None
    return None


def _struct_cache_path(work_dir: Path | None) -> Path | None:
    return work_dir / "struct.json" if work_dir else None


def _load_cached_struct(
    work_dir: Path | None,
    title: str,
    podcast_name: str,
    show_notes: str,
    cleaned_transcript: str,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    """Return the cached combined struct JSON, or None if stale/absent."""
    path = _struct_cache_path(work_dir)
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("key")
        != _struct_cache_key(title, podcast_name, show_notes, cleaned_transcript, provider, model)
        or data.get("prompt_version") != STRUCT_PROMPT_VERSION
    ):
        return None
    payload = data.get("data")
    return payload if isinstance(payload, dict) else None


def _struct_cache_key(
    title: str,
    podcast_name: str,
    show_notes: str,
    cleaned_transcript: str,
    provider: str,
    model: str,
) -> str:
    payload = json.dumps(
        {
            "title": title,
            "podcast_name": podcast_name,
            "show_notes": show_notes,
            "cleaned_transcript": cleaned_transcript,
            "provider": provider,
            "model": model,
        },
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_cached_struct(
    work_dir: Path | None,
    title: str,
    podcast_name: str,
    show_notes: str,
    cleaned_transcript: str,
    data: dict[str, Any],
    provider: str,
    model: str,
) -> None:
    path = _struct_cache_path(work_dir)
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": _struct_cache_key(title, podcast_name, show_notes, cleaned_transcript, provider, model),
        "prompt_version": STRUCT_PROMPT_VERSION,
        "data": data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_struct(
    client: OpenAI,
    model: str,
    title: str,
    podcast_name: str,
    show_notes: str,
    cleaned_transcript: str,
    *,
    work_dir: Path | None,
    provider: str,
) -> tuple[dict[str, Any], int, int]:
    """Generate all reading aids (key points/quotes/keywords/outline) in one call (cacheable)."""
    if not cleaned_transcript.strip():
        return {}, 0, 0
    cached = _load_cached_struct(
        work_dir, title, podcast_name, show_notes, cleaned_transcript, provider, model
    )
    if cached is not None:
        logger.info("Using cached struct aids")
        return cached, 0, 0
    prompt = STRUCT_USER_PROMPT.format(
        title=title,
        podcast_name=podcast_name,
        show_notes=show_notes,
        transcript_text=cleaned_transcript,
    )
    data, inp, out = _request_json(client, model, STRUCT_SYSTEM_PROMPT, prompt, max_tokens=8_192)
    _save_cached_struct(
        work_dir, title, podcast_name, show_notes, cleaned_transcript, data, provider, model
    )
    return data, inp, out


def _parse_keywords(data: dict[str, Any]) -> list[Keyword]:
    """Extract Keyword objects from an LLM JSON response."""
    return [Keyword(key=kw.key, desc=kw.desc) for kw in parse_struct(data).keywords if kw.valid]


def _build_output_doc(
    data: dict[str, Any],
    title: str,
    podcast_name: str,
    pub_date: str,
    show_notes: str,
    full_text: str,
) -> OutputDoc:
    struct = parse_struct(data)

    key_points = [
        KeyPoint(
            point=kp.point,
            evidence=kp.evidence,
            quote=kp.quote,
            terms=[TermDef(term=t.term, desc=t.desc) for t in kp.terms if t.valid],
        )
        for kp in struct.key_points
        if kp.valid
    ]
    speaker_intro = struct.speaker_intro
    speaker_mapping = struct.speaker_mapping or {}
    for speaker, name in speaker_mapping.items():
        if re.fullmatch(r"SPEAKER_\d+", str(speaker)) and isinstance(name, str) and name.strip():
            # 显示标签由脚本统一落地（剥角色+外文取 given name+【】），
            # 不再直接用 LLM 给的 speaker_mapping 原始字符串，保证两链路格式等价。
            full_text = re.sub(
                rf"\[{re.escape(str(speaker))}\]\s*",
                to_display_label(name.strip()),
                full_text,
            )
    questions = [
        QuestionItem(question=q.question, answer=q.answer)
        for q in struct.questions
        if q.valid
    ]
    return OutputDoc(
        title=title,
        podcast_name=podcast_name,
        pub_date=pub_date,
        show_notes=show_notes,
        key_points=key_points,
        full_text=full_text,
        speaker_intro=speaker_intro,
        keywords=[Keyword(key=kw.key, desc=kw.desc) for kw in struct.keywords if kw.valid],
        summary=struct.summary,
        questions=questions,
    )
