"""Tests for the LLM processor module."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.processor.llm_processor import (
    _build_output_doc,
    _clean_chunk_worker,
    _is_model_not_found,
    _load_cached_chunk,
    _parse_keywords,
    _parse_start_sec,
    _probe_model,
    _request_json,
    _request_text,
    _resolve_effective_model,
    _save_cached_chunk,
    _validate_cleaned_chunk,
    probe_available_models,
    resolve_llm_config,
    split_transcript,
)


class TestBuildOutputDoc:
    def test_builds_key_points_and_quotes(self):
        data = {
            "key_points": [
                {"point": "核心观点一", "evidence": "这是支撑证据第一句。", "quote": "这是第一句原话。"},
                {"point": "重要数据", "evidence": "数据显示增长50%。", "quote": "这是第二句原话。"},
            ],
            "speaker_intro": "> **主持人**：Nina，主播",
            "speaker_mapping": {"SPEAKER_00": "主持人Nina"},
        }
        doc = _build_output_doc(
            data, "Test Title", "Test Podcast", "2026-01-01", "notes", "[SPEAKER_00] 你好。"
        )
        assert doc.title == "Test Title"
        assert doc.podcast_name == "Test Podcast"
        assert len(doc.key_points) == 2
        assert doc.key_points[0].point == "核心观点一"
        assert doc.key_points[1].evidence == "数据显示增长50%。"
        assert doc.key_points[0].quote == "这是第一句原话。"
        assert doc.key_points[1].quote == "这是第二句原话。"
        assert "Nina" in doc.speaker_intro
        # 显示标签只留短名、剥掉角色前缀（speaker_resolver.to_display_label 为唯一真相源）
        assert "【Nina】你好。" in doc.full_text

    def test_empty_data_yields_empty_doc(self):
        doc = _build_output_doc({}, "标题", "播客", "2026-01-01", "", "正文")
        assert len(doc.key_points) == 0
        assert doc.speaker_intro == ""
        assert doc.full_text == "正文"

    def test_ignores_malformed_key_points(self):
        data = {
            "key_points": [
                {"point": "有效", "evidence": "证据"},
                {"evidence": "没有 point 字段"},
                "不是字典",
            ],
        }
        doc = _build_output_doc(data, "T", "P", "2026-01-01", "", "")
        assert len(doc.key_points) == 1
        assert doc.key_points[0].point == "有效"

    def test_speaker_intro_not_string_is_ignored(self):
        doc = _build_output_doc({"speaker_intro": 123}, "T", "P", "2026-01-01", "", "")
        assert doc.speaker_intro == ""


class TestChunkProcessing:
    def test_split_preserves_content(self):
        source = "第一行。\n第二行更长。\n第三行。"
        chunks = split_transcript(source, max_chars=10)
        assert "".join(chunks).replace("\n", "") == source.replace("\n", "")
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_rejects_suspiciously_short_cleaned_chunk(self):
        with pytest.raises(RuntimeError, match="suspicious length ratio"):
            _validate_cleaned_chunk({"chunk_id": 1, "cleaned_text": "太短"}, 1, "原文" * 100)

    def test_chunk_cache_is_bound_to_model_and_source(self, tmp_path):
        source = "这是一段原始转录。"
        _save_cached_chunk(tmp_path, 1, source, "这是一段原始转录。", "qwen", "qwen3.7-plus")
        assert _load_cached_chunk(tmp_path, 1, source, "qwen", "qwen3.7-plus") is not None
        assert _load_cached_chunk(tmp_path, 1, source + "变化", "qwen", "qwen3.7-plus") is None
        assert _load_cached_chunk(tmp_path, 1, source, "deepseek", "deepseek-v4-flash") is None

        cache = json.loads((tmp_path / "chunk_001.json").read_text(encoding="utf-8"))
        assert "source_sha256" in cache
        assert source not in cache["source_sha256"]

    def test_chunk_cache_respects_prompt_version(self, tmp_path):
        source = "这是一段原始转录。"
        _save_cached_chunk(tmp_path, 1, source, "校订结果", "qwen", "qwen3.7-plus")
        assert _load_cached_chunk(tmp_path, 1, source, "qwen", "qwen3.7-plus") is not None
        # 手工把缓存标记为旧版提示词，应视为无效
        cache = json.loads((tmp_path / "chunk_001.json").read_text(encoding="utf-8"))
        cache["prompt_version"] = 0
        (tmp_path / "chunk_001.json").write_text(json.dumps(cache), encoding="utf-8")
        assert _load_cached_chunk(tmp_path, 1, source, "qwen", "qwen3.7-plus") is None

    def test_chunk_cache_is_bound_to_show_notes(self, tmp_path):
        source = "这是一段原始转录。"
        _save_cached_chunk(tmp_path, 1, source, "校订结果", "qwen", "qwen3.7-plus", show_notes_key="aaa")
        assert _load_cached_chunk(tmp_path, 1, source, "qwen", "qwen3.7-plus", show_notes_key="aaa") is not None
        # Show Notes 变化应使依赖专名校订的旧分块作废
        assert _load_cached_chunk(tmp_path, 1, source, "qwen", "qwen3.7-plus", show_notes_key="bbb") is None

    def test_speaker_mapping_replaces_only_valid_labels(self):
        data = {
            "speaker_mapping": {
                "SPEAKER_00": "主播连漪",
                "invalid": "不能替换",
            }
        }
        doc = _build_output_doc(
            data,
            "标题",
            "播客",
            "2026-01-01",
            "",
            "[SPEAKER_00] 你好。\n[SPEAKER_01] 你好。",
        )
        # 同上：显示标签剥角色前缀，主播连漪 → 【连漪】
        assert "【连漪】你好。" in doc.full_text
        assert "[SPEAKER_01]" in doc.full_text


class TestStartTimeParsing:
    def test_mm_ss(self):
        assert _parse_start_sec("05:49") == 5 * 60 + 49

    def test_h_mm_ss(self):
        assert _parse_start_sec("1:05:49") == 3600 + 5 * 60 + 49

    def test_numeric_seconds(self):
        assert _parse_start_sec(349) == 349
        assert _parse_start_sec("45.0") == 45.0

    def test_malformed_returns_none(self):
        assert _parse_start_sec("") is None
        assert _parse_start_sec("abc") is None
        assert _parse_start_sec(True) is None
        assert _parse_start_sec(None) is None


class TestParsingSummaryAndQuestions:
    def test_build_output_doc_includes_summary_and_questions(self):
        data = {
            "summary": " 本期讲求职策略 ",
            "questions": [
                {"question": "找不到工作是谁的问题？", "answer": "策略错位。"},
                {"question": "副业该不该搞？"},
            ],
        }
        doc = _build_output_doc(data, "T", "P", "2026-01-01", "", "")
        assert doc.summary == "本期讲求职策略"
        assert len(doc.questions) == 2
        assert doc.questions[0].question == "找不到工作是谁的问题？"
        assert doc.questions[0].answer == "策略错位。"
        assert doc.questions[1].answer == ""


class TestParsingKeywords:
    def test_parses_and_trims(self):
        data = {"keywords": [{"key": " AI 产品 ", "desc": " 说明 "}, {"key": "术语"}]}
        keywords = _parse_keywords(data)
        assert [k.key for k in keywords] == ["AI 产品", "术语"]
        assert keywords[0].desc == "说明"

    def test_skips_invalid_items_and_limits(self):
        data = {
            "keywords": [
                {"key": "一", "desc": "d"},
                {"key": ""},
                "坏条目",
                {"desc": "没有 key"},
            ]
            + [{"key": f"词{i}", "desc": "d"} for i in range(20)]
        }
        keywords = _parse_keywords(data)
        assert all(k.key for k in keywords)
        assert len(keywords) == 10  # MAX_KEYWORDS

    def test_missing_or_non_list_returns_empty(self):
        assert _parse_keywords({}) == []
        assert _parse_keywords({"keywords": "不是列表"}) == []


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt=10, completion=20):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class FakeCompletions:
    """Scriptable stub standing in for `client.chat.completions`."""

    def __init__(self, script):
        # script: list of values (response) or exceptions to raise in order
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("create called more times than scripted")
        value = self.script.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeModels:
    def __init__(self, ids=None, error=None):
        self._ids = ids or []
        self._error = error
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        if self._error:
            raise self._error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self._ids])


class FakeClient:
    def __init__(self, responses, model_ids=None, models_error=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))
        self.models = FakeModels(ids=model_ids, error=models_error)


def _response(content="{}", finish_reason="stop", prompt=10, completion=20):
    return SimpleNamespace(
        choices=[FakeChoice(content, finish_reason)],
        usage=FakeUsage(prompt, completion),
    )


def _json_response(data, **kw):
    return _response(json.dumps(data, ensure_ascii=False), **kw)


class TestRequestJson:
    def test_json_mode_fallback_on_bad_request(self):
        httpx_response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        client = FakeClient(
            [
                openai.BadRequestError(
                    "response_format unsupported",
                    response=httpx_response,
                    body=None,
                ),
                _json_response({"ok": True}),
            ]
        )
        data, inp, out = _request_json(client, "m", "sys", "user", max_tokens=100)
        assert data == {"ok": True}
        first, second = client.chat.completions.calls
        assert "response_format" in first
        assert "response_format" not in second

    def test_rate_limit_retries_with_backoff_then_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("src.processor.llm_processor.time.sleep", sleeps.append)
        client = FakeClient([RuntimeError("429 Token rate limit"), _json_response({"ok": 1})])
        data, _, _ = _request_json(client, "m", "sys", "u", max_tokens=10)
        assert data == {"ok": 1}


class TestRequestText:
    def test_incomplete_finish_reason_is_retried(self):
        client = FakeClient([_response("正文", finish_reason="length"), _response("正文。")])
        text, _, _ = _request_text(client, "m", "s", "u", max_tokens=100)
        assert text == "正文。"

    def test_empty_response_is_retried(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("src.processor.llm_processor.time.sleep", sleeps.append)
        client = FakeClient([_response(""), _response("正文。")])
        text, _, _ = _request_text(client, "m", "s", "u", max_tokens=100)
        assert text == "正文。"
        assert len(sleeps) == 0


class TestRequestJsonRetries:

    def test_persistent_failure_raises_last_error(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("src.processor.llm_processor.time.sleep", sleeps.append)
        client = FakeClient([RuntimeError("boom")] * 10)
        with pytest.raises(RuntimeError, match="boom"):
            _request_json(client, "m", "s", "u", max_tokens=10)
        assert len(sleeps) == 0  # 非限流类错误不退避

    def test_incomplete_finish_reason_is_retried(self):
        client = FakeClient([_response("{}", finish_reason="length"), _json_response({"ok": 1})])
        data, _, _ = _request_json(client, "m", "s", "u", max_tokens=10)
        assert data == {"ok": 1}


class TestCleanChunkWorker:
    def test_uses_cache_and_skips_api(self, tmp_path):
        source = "原始转录内容" * 20
        _save_cached_chunk(tmp_path, 1, source, "已校订结果", "qwen", "qwen3.7-plus")
        client = FakeClient([])
        cleaned, inp, out = _clean_chunk_worker(
            client=client,
            chunks=[source],
            index=0,
            show_notes="",
            notes_key="",
            provider="qwen",
            model="qwen3.7-plus",
            work_dir=tmp_path,
        )
        assert cleaned == "已校订结果"
        assert inp == out == 0
        assert client.chat.completions.calls == []

    def test_cleans_and_saves_cache_on_miss(self, tmp_path):
        source = "这是第一段转录" * 20
        content = {"chunk_id": 1, "cleaned_text": "这是校订后的结果文本。" + "补" * len(source)}
        client = FakeClient([_json_response(content)])
        cleaned, inp, out = _clean_chunk_worker(
            client=client, chunks=[source], index=0, show_notes="note", notes_key="abc",
            provider="qwen", model="m", work_dir=tmp_path,
        )
        assert cleaned == content["cleaned_text"]
        assert (tmp_path / "chunk_001.json").exists()
        cached = _load_cached_chunk(tmp_path, 1, source, "qwen", "m", show_notes_key="abc")
        assert cached == content["cleaned_text"]

    def test_persistent_validation_failure_raises_and_writes_no_cache(self, tmp_path):
        source = "原文内容" * 50
        wrong = {"chunk_id": 999, "cleaned_text": "wrong"}
        client = FakeClient([_json_response(wrong), _json_response(wrong)])
        with pytest.raises(RuntimeError):
            _clean_chunk_worker(
                client=client, chunks=[source], index=0, show_notes="", notes_key="",
                provider="p", model="m", work_dir=tmp_path,
            )
        assert not (tmp_path / "chunk_001.json").exists()


class TestModelResolution:
    def test_advertised_models_used_for_match(self):
        client = FakeClient([], model_ids=["gpt-5.2", "gpt-5.1"])
        assert _resolve_effective_model(client, "qwen3.7-plus", ("gpt-5.2",)) == "gpt-5.2"

    def test_primary_model_preferred_when_advertised(self):
        client = FakeClient([], model_ids=["gpt-5.2", "qwen3.7-plus"])
        assert _resolve_effective_model(client, "qwen3.7-plus", ("gpt-5.2",)) == "qwen3.7-plus"

    def test_no_matching_advertised_model_raises(self):
        client = FakeClient([], model_ids=["gpt-5.2", "gpt-5.1"])
        with pytest.raises(Exception, match="不在网关可用列表"):
            _resolve_effective_model(client, "qwen3.7-plus")

    def test_live_probe_falls_back_through_candidates(self):
        # /models 不可用；qwen 404，回退到 gpt-5.2 成功
        httpx_response = httpx.Response(404, request=httpx.Request("POST", "http://x"))
        client = FakeClient(
            [openai.NotFoundError("model_not_found", response=httpx_response, body=None), _json_response({"ok": 1})],
            models_error=RuntimeError("no /models"),
        )
        assert _resolve_effective_model(client, "qwen3.7-plus", ("gpt-5.2",)) == "gpt-5.2"

    def test_live_probe_transient_error_keeps_primary(self):
        # 连接类错误不是 model_not_found：按可用处理，返回主候选
        client = FakeClient([RuntimeError("502 upstream")], models_error=RuntimeError("no /models"))
        assert _resolve_effective_model(client, "qwen3.7-plus") == "qwen3.7-plus"

    def test_probe_model_reports_usable(self):
        client = FakeClient([_json_response({"ok": 1})])
        usable, exc = _probe_model(client, "gpt-5.2")
        assert usable is True
        assert exc is None

    def test_probe_model_reports_not_found(self):
        httpx_response = httpx.Response(404, request=httpx.Request("POST", "http://x"))
        client = FakeClient([openai.NotFoundError("model_not_found", response=httpx_response, body=None)])
        usable, exc = _probe_model(client, "nope")
        assert usable is False
        assert isinstance(exc, openai.NotFoundError)

    def test_probe_available_models_returns_sorted_ids(self):
        client = FakeClient([], model_ids=["gpt-5.2", "gpt-5.1"])
        assert probe_available_models(client) == ["gpt-5.1", "gpt-5.2"]

    def test_probe_available_models_empty_on_error(self):
        client = FakeClient([], models_error=RuntimeError("no /models"))
        assert probe_available_models(client) == []

    def test_resolve_llm_config_pins_advertised_model(self):
        from src.config import LLMConfig

        client = FakeClient([], model_ids=["gpt-5.2"])
        original = LLMConfig(
            api_key="k", base_url="https://x/v1", provider="qwen",
            model="qwen3.7-plus", fallback_models=("gpt-5.2",),
        )
        resolved = resolve_llm_config(original, client=client)
        assert resolved.model == "gpt-5.2"
        assert resolved.provider == "qwen"

    def test_resolve_llm_config_noop_when_model_served(self):
        from src.config import LLMConfig

        client = FakeClient([], model_ids=["qwen3.7-plus", "gpt-5.2"])
        original = LLMConfig(api_key="k", base_url="https://x/v1", provider="qwen", model="qwen3.7-plus")
        resolved = resolve_llm_config(original, client=client)
        assert resolved is original


class TestModelNotFoundClassification:
    def test_openai_not_found_is_model_missing(self):
        httpx_response = httpx.Response(404, request=httpx.Request("POST", "http://x"))
        exc = openai.NotFoundError("model_not_found", response=httpx_response, body=None)
        assert _is_model_not_found(exc)

    def test_message_marker_detected(self):
        assert _is_model_not_found(RuntimeError("The model does not exist"))
        assert _is_model_not_found(RuntimeError("Model not found: qwen3.7-plus"))
        assert not _is_model_not_found(RuntimeError("502 Bad Gateway"))
        assert not _is_model_not_found(RuntimeError("429 rate limit"))
