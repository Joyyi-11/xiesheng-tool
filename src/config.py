import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_PROVIDER = "qwen"
LLM_MODELS = {
    "qwen": "qwen3.7-plus",
    "deepseek": "deepseek-v4-flash",
}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    provider: str
    model: str
    fallback_models: tuple[str, ...] = ()  # ordered extra candidates, tried after `model`


def llm_configured() -> bool:
    """Whether an LLM API is configured (needs both key and endpoint)."""
    return bool(os.getenv("LLM_API_KEY")) and bool(os.getenv("LLM_BASE_URL"))


def parse_model_candidates(model: str | None) -> tuple[str, ...]:
    """Split a comma-separated model spec into ordered, deduped candidates.

    ``"--llm-model qwen3.7-plus,gpt-5.2"`` yields ``("qwen3.7-plus", "gpt-5.2")``.
    Empty/whitespace input yields ``()``.
    """
    if not model:
        return ()
    seen: list[str] = []
    for part in model.split(","):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def get_llm_config(
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
) -> LLMConfig:
    if provider not in LLM_MODELS:
        supported = ", ".join(sorted(LLM_MODELS))
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose from: {supported}")
    candidates = parse_model_candidates(model)
    primary = candidates[0] if candidates else LLM_MODELS[provider]
    return LLMConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
        provider=provider,
        model=primary,
        fallback_models=candidates[1:],
    )
