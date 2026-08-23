"""Tests for provider-neutral LLM configuration."""

import pytest

from src.config import get_llm_config, llm_configured, parse_model_candidates


def test_llm_configured_requires_both(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert not llm_configured()
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert not llm_configured()
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    assert llm_configured()


def test_qwen_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    config = get_llm_config("qwen")
    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "qwen3.7-plus"


def test_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert get_llm_config("deepseek").model == "deepseek-v4-flash"


def test_parse_model_candidates():
    assert parse_model_candidates(None) == ()
    assert parse_model_candidates("") == ()
    assert parse_model_candidates("   ") == ()
    assert parse_model_candidates("gpt-5.2") == ("gpt-5.2",)
    assert parse_model_candidates("qwen3.7-plus,gpt-5.2") == ("qwen3.7-plus", "gpt-5.2")
    assert parse_model_candidates(" a , b , a ") == ("a", "b")  # 去重且保序


def test_comma_separated_model_sets_primary_and_fallback(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    config = get_llm_config("qwen", "qwen3.7-plus,gpt-5.2")
    assert config.model == "qwen3.7-plus"
    assert config.fallback_models == ("gpt-5.2",)


def test_single_model_has_no_fallback(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    config = get_llm_config("qwen", "gpt-5.2")
    assert config.model == "gpt-5.2"
    assert config.fallback_models == ()


def test_default_has_no_fallback(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert get_llm_config("qwen").fallback_models == ()


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_config("unknown")
