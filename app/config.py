from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # One key authenticates every model role — see app/llm.py, which builds a
    # single AsyncOpenAI client from this key and reuses it for
    # classify/extract/question/answer/vision/embed. The model_* fields below
    # only select which model each role calls, they are not separate
    # credentials. Deliberately generic names (not e.g. fireworks_api_key) —
    # the whole point of this field/module naming is that switching providers
    # again later is a config change (this value + llm_base_url + the
    # model_* slugs below), not a rename sweep across the codebase. Was
    # DeepInfra (DEEP_INFRA_API, https://api.deepinfra.com/v1/openai) before
    # the switch to Fireworks AI, 2026-08-13. Accepts either env var name —
    # LLM_API_KEY is the documented/generic one (.env.example), FIRE_WORK_API
    # is also accepted since that's what was already provisioned in the local
    # dev .env before this field even existed; first alias found wins.
    llm_api_key: str = Field(validation_alias=AliasChoices("LLM_API_KEY", "FIRE_WORK_API"))
    llm_base_url: str = "https://api.fireworks.ai/inference/v1"

    mongo_uri: str
    mongo_db_name: str = "interior_design_chat"

    # The knowledge graph (KnowledgeNode/KnowledgeNodeVersion/KnowledgeEdge)
    # lives in Neo4j, not Mongo — see app/graph_store.py. ChatSession/
    # ProjectContext/CatalogItem stay on Mongo above; those are plain
    # documents with no relationship structure, Neo4j buys nothing there.
    # Env var names match Neo4j Desktop's own connection panel convention
    # (NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD), not this file's usual
    # snake_case-of-the-field convention — kept as-is rather than renamed,
    # same reasoning as llm_api_key/LLM_API_KEY above.
    neo4j_uri: str
    neo4j_username: str = Field(default="neo4j", validation_alias="NEO4J_USERNAME")
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # --- Model roles, all on Fireworks AI as of 2026-08-13 ---
    #
    # History (all on DeepInfra at the time): model_intent_classifier was
    # bumped from an 8B Turbo model to Llama-3.3-70B-Instruct-Turbo, then to
    # gpt-oss-120b — splitting a message into several independent,
    # correctly-scoped, graph-grounded operations is a harder task than the
    # single-label classify_intent this role originally replaced.
    # model_extraction/model_question_gen were previously GLM-4.7-Flash (a
    # reasoning model whose invisible chain-of-thought was the dominant
    # source of per-turn latency: 8-23s per call, 3 sequential calls on the
    # update_context branch), then mistralai/Mistral-Small-3.2-24B-Instruct-2506
    # (non-reasoning, but live-timed at 11-62s per call regardless of output
    # length — almost certainly served on a cold/shared DeepInfra tier rather
    # than a low-latency one). Reusing the same fast Turbo model already
    # proven for classify_intent fixed both (2-3s per call).
    #
    # Per user instruction, every text/reasoning role below now shares one
    # model: accounts/fireworks/models/gpt-oss-120b. Unverified against a
    # live Fireworks account — confirm it's actually deployed there before
    # relying on it. It's a REASONING model (like the GLM-4.7-Flash/
    # Mistral-Small history above) — every chat-completion call in app/llm.py
    # for this model already passes extra_body={"reasoning_effort": "low",
    # "reasoning_history": "disabled"} (see _REASONING_KWARGS there) to keep
    # hidden reasoning-token generation down, but this is still worth a live
    # timing check on model_answer in particular (the one streamed
    # token-by-token to the client) before trusting it in production; if it
    # turns out slow the same non-reasoning-model fix that worked twice
    # before is the first thing to try.
    model_intent_classifier: str = "accounts/fireworks/models/gpt-oss-120b"
    model_extraction: str = "accounts/fireworks/models/gpt-oss-120b"
    model_question_gen: str = "accounts/fireworks/models/gpt-oss-120b"
    model_answer: str = "accounts/fireworks/models/gpt-oss-120b"
    # Wasn't covered by the "use gpt-oss-120b for everything" instruction —
    # gpt-oss isn't multimodal. Slug not confirmed live on Fireworks — verify
    # before relying on it.
    model_vision: str = "accounts/fireworks/models/qwen3p7-plus"
    # Same model family as the previous DeepInfra choice (Qwen3-Embedding-8B),
    # just under Fireworks' own hosted namespace — chosen specifically to
    # keep the embedding vector space stable across the provider switch, so
    # existing persisted Label/CatalogItem embeddings should stay
    # cosine-comparable against anything newly embedded here (unlike a switch
    # to a genuinely different embedding model, which would silently produce
    # garbage similarity scores for old vs. new vectors — see
    # app.canonical_mapper._score_candidates, which only backfills an
    # embedding that's actually missing, never one that's merely stale).
    # Slug not confirmed live on Fireworks — verify before relying on it.
    model_embedding: str = "accounts/fireworks/models/qwen3-embedding-8b"

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
