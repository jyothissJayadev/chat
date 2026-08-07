from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # One key authenticates every model role — see app/deepinfra.py, which
    # builds a single AsyncOpenAI client from this key and reuses it for
    # classify/extract/question/answer/vision/embed. The model_* fields below
    # only select which model each role calls, they are not separate credentials.
    # Read from DEEP_INFRA_API to match the account key as provisioned.
    deepinfra_api_key: str = Field(validation_alias="DEEP_INFRA_API")
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"

    mongo_uri: str
    mongo_db_name: str = "interior_design_chat"

    model_intent_classifier: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # Non-reasoning models (with the 'tools' tag instructor's TOOLS mode needs) —
    # GLM-4.7-Flash was previously used here but its reasoning overhead (burning
    # tokens on invisible chain-of-thought) was the dominant source of per-turn
    # latency: 8-23s per call, 3 sequential calls on the update_context branch.
    # mistralai/Mistral-Small-3.2-24B-Instruct-2506 (its non-reasoning
    # replacement) turned out to have its own, unrelated latency problem:
    # live-timed at 11-62s per call regardless of output length, vs. <2s for
    # every "Turbo"-branded model tried — almost certainly served on a
    # cold/shared DeepInfra tier rather than a low-latency one. Reusing the
    # same Turbo model already proven fast for classify_intent fixed it:
    # 2-3s per call, confirmed working with instructor's TOOLS mode too.
    model_extraction: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    model_question_gen: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    model_answer: str = "deepseek-ai/DeepSeek-V4-Pro"
    model_vision: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    model_embedding: str = "Qwen/Qwen3-Embedding-8B"

    session_ttl_days: int = 30

    # Optional — observability stays fully off if these aren't set. Names match
    # the SDK's own env var conventions, but we still read them via Settings
    # (rather than relying on Langfuse's internal os.environ fallback) since
    # pydantic-settings loads .env into this model without exporting it to the
    # real process environment.
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None


settings = Settings()
