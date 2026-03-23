"""
Claude API wrapper with:
  - Tiered model selection (Opus/Sonnet/Haiku)
  - Prompt caching (cache_control: ephemeral)
  - Automatic token usage logging
  - Cost calculation and alerting
"""

import anthropic
import asyncio
import logging
from typing import Any
from .config import config, AGENT_MODEL_MAP
from . import db

logger = logging.getLogger("zo.claude")


class ClaudeClient:
    """Tiered, cost-tracked Claude API client."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
            timeout=600.0,  # 10 min per API call — Builder steps with 20K+ tokens need time
        )

    async def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        project_id: str | None = None,
        workflow: str = "langgraph",
        max_tokens: int = 8000,
        temperature: float = 0.3,
        use_cache: bool = True,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Call Claude with automatic model selection, caching, and cost tracking.

        Args:
            agent_name: Which agent is calling (determines model tier)
            system_prompt: The agent's system prompt
            user_message: The user/pipeline message
            project_id: For cost attribution
            workflow: Which workflow triggered this
            max_tokens: Max output tokens
            temperature: Creativity level (0.0-1.0)
            use_cache: Whether to use prompt caching
            extra_context: Additional context (e.g., ecosystem learnings)
        """
        # Determine model tier
        tier = AGENT_MODEL_MAP.get(agent_name, "sonnet")
        model = config.models.get_model(tier)

        # Build system messages with caching
        system_messages = []
        if use_cache:
            system_messages.append({
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            })
        else:
            system_messages.append({
                "type": "text",
                "text": system_prompt,
            })

        if extra_context:
            system_messages.append({
                "type": "text",
                "text": extra_context,
            })

        # Make the API call (async — does NOT block the event loop)
        logger.info("Calling %s (model=%s, tier=%s)", agent_name, model, tier)
        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_messages,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Calculate cost
        cost = config.models.calculate_cost(tier, input_tokens, output_tokens, cached_tokens)

        # Log to Supabase
        await db.log_token_usage(
            workflow=workflow,
            mind=agent_name,
            model=model,
            model_tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            project_id=project_id,
            notes=f"temp={temperature}",
        )

        # Check cost alert threshold
        threshold = float(await db.get_config("cost_alert_threshold_cad", "5.00"))
        cost_cad = cost * 1.38  # Approximate USD to CAD
        if cost_cad > threshold:
            await db.emit_event(
                event_type="cost_alert",
                project_id=project_id,
                source_agent=agent_name,
                payload={
                    "cost_usd": cost,
                    "cost_cad": round(cost_cad, 2),
                    "threshold_cad": threshold,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return {
            "content": text_content,
            "model": model,
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost,
            "stop_reason": response.stop_reason,
        }


# Singleton
claude = ClaudeClient()
