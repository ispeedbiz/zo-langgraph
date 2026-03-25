"""
ZeroOrigine Configuration — Single source of truth.
Reads from Supabase zo_config table + environment variables.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelTier:
    """Claude API model tier configuration."""
    opus: str = "claude-opus-4-6"
    sonnet: str = "claude-sonnet-4-6"
    haiku: str = "claude-haiku-4-5-20251001"

    # Cost per million tokens (USD)
    opus_input_cost: float = 5.0
    opus_output_cost: float = 25.0
    sonnet_input_cost: float = 3.0
    sonnet_output_cost: float = 15.0
    haiku_input_cost: float = 1.0
    haiku_output_cost: float = 5.0

    # Cached input tokens are 90% cheaper
    cache_discount: float = 0.10  # pay 10% of normal input cost

    def get_model(self, tier: str) -> str:
        return getattr(self, tier, self.sonnet)

    def calculate_cost(self, tier: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        input_cost = getattr(self, f"{tier}_input_cost", self.sonnet_input_cost)
        output_cost = getattr(self, f"{tier}_output_cost", self.sonnet_output_cost)

        billable_input = input_tokens - cached_tokens
        cost = (
            (billable_input / 1_000_000) * input_cost
            + (cached_tokens / 1_000_000) * input_cost * self.cache_discount
            + (output_tokens / 1_000_000) * output_cost
        )
        return round(cost, 6)


# Agent → Model Tier mapping
# Opus 4.6 everywhere — full creative freedom → Opus
# Only status_check and json_extract stay on Haiku (simple utility tasks)
AGENT_MODEL_MAP: dict[str, str] = {
    "research_a": "opus",
    "ethics": "opus",
    "research_b": "opus",
    "builder": "opus",
    "builder_opus": "opus",
    "qa": "opus",
    "marketing": "opus",
    "dispatcher": "opus",
    "support": "opus",
    "status_check": "haiku",
    "json_extract": "haiku",
}


@dataclass
class Config:
    """Application configuration."""
    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_service_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    n8n_webhook_base: str = field(default_factory=lambda: os.getenv("N8N_WEBHOOK_BASE", ""))

    # Service
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))

    # Models
    models: ModelTier = field(default_factory=ModelTier)

    def get_agent_model(self, agent_name: str) -> str:
        tier = AGENT_MODEL_MAP.get(agent_name, "sonnet")
        return self.models.get_model(tier)

    def get_agent_tier(self, agent_name: str) -> str:
        return AGENT_MODEL_MAP.get(agent_name, "sonnet")


config = Config()
