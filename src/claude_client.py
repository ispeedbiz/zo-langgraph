"""
Claude API wrapper with:
  - Tiered model selection (Opus/Sonnet/Haiku)
  - Prompt caching (cache_control: ephemeral)
  - Adaptive thinking + effort parameter (Claude 4.6)
  - Automatic token usage logging
  - Cost calculation and alerting
  - Error capture to zo_mind_logs + metadata
  - Retry with backoff for transient errors (500, 529)
"""

import anthropic
import asyncio
import logging
import traceback
from typing import Any
from .config import config, AGENT_MODEL_MAP
from . import db

logger = logging.getLogger("zo.claude")

# Effort levels per agent — Builder gets max, others get high
AGENT_EFFORT = {
    "builder": "max",
    "builder_step1": "max",
    "builder_step2": "max",
    "builder_step3": "max",
    "builder_step4": "max",
    "builder_step5": "max",
    "builder_step6": "max",
    "builder_opus": "max",
    "research_a": "high",
    "research_b": "high",
    "ethics": "high",
    "qa": "high",
    "marketing": "high",
    "build-architect": "high",
    "hotfix_diagnose": "high",
    "hotfix_patch": "high",
    "hotfix_verify": "medium",
}

# Retry config: transient errors get retried, client errors don't
RETRYABLE_STATUS_CODES = {500, 502, 503, 529}
MAX_RETRIES = 3
RETRY_DELAYS = [30, 60, 120]  # seconds — exponential backoff


class ClaudeClient:
    """Tiered, cost-tracked Claude API client with adaptive thinking and retry."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
            timeout=600.0,
        )

    async def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        project_id: str | None = None,
        workflow: str = "langgraph",
        max_tokens: int = 16000,
        temperature: float = 0.3,
        use_cache: bool = True,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        """Call Claude with adaptive thinking, effort parameter, retry, and error capture."""

        # Determine model tier and effort
        tier = AGENT_MODEL_MAP.get(agent_name, "sonnet")
        model = config.models.get_model(tier)
        effort = AGENT_EFFORT.get(agent_name, "high")

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

        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_messages,
            "messages": [{"role": "user", "content": user_message}],
            # thinking disabled — custom temperature per step for deterministic code generation
        }

        # Add effort parameter
        if effort:
            api_kwargs["output_config"] = {"effort": effort}

        logger.info(
            "Calling %s (model=%s, tier=%s, effort=%s, max_tokens=%d)",
            agent_name, model, tier, effort, max_tokens,
        )

        # === API call with retry for transient errors ===
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.messages.create(**api_kwargs)
                break  # Success — exit retry loop
            except anthropic.BadRequestError as e:
                # 400 = client error, do NOT retry
                error_msg = f"400 BadRequest for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.RateLimitError as e:
                # 429 = rate limit, retry with backoff
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "429 RateLimit for %s (attempt %d/%d), retrying in %ds...",
                        agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"429: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"429 RateLimit for {agent_name} after {MAX_RETRIES} retries: {str(e)[:200]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "API %d for %s (attempt %d/%d), retrying in %ds...",
                        e.status_code, agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"{e.status_code}: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"API error {e.status_code} for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except Exception as e:
                error_msg = f"Unexpected error for {agent_name}: {str(e)[:300]}\n{traceback.format_exc()[:500]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
        else:
            # All retries exhausted
            error_msg = f"All {MAX_RETRIES} retries exhausted for {agent_name}: {str(last_error)[:300]}"
            logger.error(error_msg)
            await self._log_error(agent_name, workflow, project_id, error_msg)
            return self._error_response(model, tier, error_msg)

        # Extract usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Calculate cost
        cost = config.models.calculate_cost(tier, input_tokens, output_tokens, cached_tokens)

        # Log to zo_cost_logs
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
            notes=f"effort={effort},temp={temperature}",
        )

        # Extract text content (skip thinking blocks)
        text_content = ""
        thinking_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking

        # Log to zo_mind_logs (every Mind call tracked)
        # Column names match actual schema: tokens_used, error_message, status
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": input_tokens + output_tokens,
                "cost_usd": cost,
                "project_id": project_id,
                "status": "success",
                "error_message": None,
                "input_summary": user_message[:200] if user_message else "",
                "output_summary": text_content[:200] if text_content else "",
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log to zo_mind_logs: %s", str(exc)[:100])

        # Cost alert
        try:
            threshold = float(await db.get_config("cost_alert_threshold_cad", "5.00"))
            cost_cad = cost * 1.38
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
                    },
                )
        except Exception:
            pass

        return {
            "content": text_content,
            "thinking": thinking_content,
            "model": model,
            "tier": tier,
            "effort": effort,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost,
            "stop_reason": response.stop_reason,
            "error": None,
        }

    async def _log_retry(self, agent_name: str, workflow: str, project_id: str | None, attempt: int, reason: str):
        """Log retry attempts to zo_mind_logs."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "retry",
                "error_message": f"Retry {attempt}/{MAX_RETRIES}: {reason}",
                "input_summary": f"retry_attempt_{attempt}",
                "output_summary": "",
            }).execute()
        except Exception:
            pass

    async def _log_error(self, agent_name: str, workflow: str, project_id: str | None, error_msg: str):
        """Log API errors to zo_mind_logs and project metadata."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "error",
                "error_message": error_msg[:500],
                "input_summary": "API ERROR",
                "output_summary": error_msg[:200],
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log error to zo_mind_logs: %s", str(exc)[:100])

        # Write error to project metadata
        if project_id:
            try:
                db.get_client().table("zo_projects").update({
                    "metadata": {"error": error_msg[:200]},
                }).eq("project_id", project_id).execute()
            except Exception:
                pass

    def _error_response(self, model: str, tier: str, error_msg: str) -> dict[str, Any]:
        """Return a structured error response so callers can check response['error']."""
        return {
            "content": "",
            "thinking": "",
            "model": model,
            "tier": tier,
            "effort": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0,
            "stop_reason": "error",
            "error": error_msg,
        }


# Singleton
claude = ClaudeClient()
"""
Claude API wrapper with:
  - Tiered model selection (Opus/Sonnet/Haiku)
  - Prompt caching (cache_control: ephemeral)
  - Adaptive thinking + effort parameter (Claude 4.6)
  - Automatic token usage logging
  - Cost calculation and alerting
  - Error capture to zo_mind_logs + metadata
  - Retry with backoff for transient errors (500, 529)
"""

import anthropic
import asyncio
import logging
import traceback
from typing import Any
from .config import config, AGENT_MODEL_MAP
from . import db

logger = logging.getLogger("zo.claude")

# Effort levels per agent — Builder gets max, others get high
AGENT_EFFORT = {
    "builder": "max",
    "builder_step1": "max",
    "builder_step2": "max",
    "builder_step3": "max",
    "builder_step4": "max",
    "builder_step5": "max",
    "builder_step6": "max",
    "builder_opus": "max",
    "research_a": "high",
    "research_b": "high",
    "ethics": "high",
    "qa": "high",
    "marketing": "high",
    "build-architect": "high",
    "hotfix_diagnose": "high",
    "hotfix_patch": "high",
    "hotfix_verify": "medium",
}

# Retry config: transient errors get retried, client errors don't
RETRYABLE_STATUS_CODES = {500, 502, 503, 529}
MAX_RETRIES = 3
RETRY_DELAYS = [30, 60, 120]  # seconds — exponential backoff


class ClaudeClient:
    """Tiered, cost-tracked Claude API client with adaptive thinking and retry."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
            timeout=600.0,
        )

    async def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        project_id: str | None = None,
        workflow: str = "langgraph",
        max_tokens: int = 16000,
        temperature: float = 0.3,
        use_cache: bool = True,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        """Call Claude with adaptive thinking, effort parameter, retry, and error capture."""

        # Determine model tier and effort
        tier = AGENT_MODEL_MAP.get(agent_name, "sonnet")
        model = config.models.get_model(tier)
        effort = AGENT_EFFORT.get(agent_name, "high")

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

        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_messages,
            "messages": [{"role": "user", "content": user_message}],
            # thinking disabled — custom temperature per step for deterministic code gen
        }

        # Add effort parameter
        if effort:
            api_kwargs["output_config"] = {"effort": effort}

        logger.info(
            "Calling %s (model=%s, tier=%s, effort=%s, max_tokens=%d)",
            agent_name, model, tier, effort, max_tokens,
        )

        # === API call with retry for transient errors ===
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.messages.create(**api_kwargs)
                break  # Success — exit retry loop
            except anthropic.BadRequestError as e:
                # 400 = client error, do NOT retry
                error_msg = f"400 BadRequest for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.RateLimitError as e:
                # 429 = rate limit, retry with backoff
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "429 RateLimit for %s (attempt %d/%d), retrying in %ds...",
                        agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"429: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"429 RateLimit for {agent_name} after {MAX_RETRIES} retries: {str(e)[:200]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "API %d for %s (attempt %d/%d), retrying in %ds...",
                        e.status_code, agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"{e.status_code}: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"API error {e.status_code} for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except Exception as e:
                error_msg = f"Unexpected error for {agent_name}: {str(e)[:300]}\n{traceback.format_exc()[:500]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
        else:
            # All retries exhausted
            error_msg = f"All {MAX_RETRIES} retries exhausted for {agent_name}: {str(last_error)[:300]}"
            logger.error(error_msg)
            await self._log_error(agent_name, workflow, project_id, error_msg)
            return self._error_response(model, tier, error_msg)

        # Extract usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Calculate cost
        cost = config.models.calculate_cost(tier, input_tokens, output_tokens, cached_tokens)

        # Log to zo_cost_logs
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
            notes=f"effort={effort},temp={temperature}",
        )

        # Extract text content (skip thinking blocks)
        text_content = ""
        thinking_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking

        # Log to zo_mind_logs (every Mind call tracked)
        # Column names match actual schema: tokens_used, error_message, status
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": input_tokens + output_tokens,
                "cost_usd": cost,
                "project_id": project_id,
                "status": "success",
                "error_message": None,
                "input_summary": user_message[:200] if user_message else "",
                "output_summary": text_content[:200] if text_content else "",
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log to zo_mind_logs: %s", str(exc)[:100])

        # Cost alert
        try:
            threshold = float(await db.get_config("cost_alert_threshold_cad", "5.00"))
            cost_cad = cost * 1.38
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
                    },
                )
        except Exception:
            pass

        return {
            "content": text_content,
            "thinking": thinking_content,
            "model": model,
            "tier": tier,
            "effort": effort,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost,
            "stop_reason": response.stop_reason,
            "error": None,
        }

    async def _log_retry(self, agent_name: str, workflow: str, project_id: str | None, attempt: int, reason: str):
        """Log retry attempts to zo_mind_logs."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "retry",
                "error_message": f"Retry {attempt}/{MAX_RETRIES}: {reason}",
                "input_summary": f"retry_attempt_{attempt}",
                "output_summary": "",
            }).execute()
        except Exception:
            pass

    async def _log_error(self, agent_name: str, workflow: str, project_id: str | None, error_msg: str):
        """Log API errors to zo_mind_logs and project metadata."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "error",
                "error_message": error_msg[:500],
                "input_summary": "API ERROR",
                "output_summary": error_msg[:200],
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log error to zo_mind_logs: %s", str(exc)[:100])

        # Write error to project metadata
        if project_id:
            try:
                db.get_client().table("zo_projects").update({
                    "metadata": {"error": error_msg[:200]},
                }).eq("project_id", project_id).execute()
            except Exception:
                pass

    def _error_response(self, model: str, tier: str, error_msg: str) -> dict[str, Any]:
        """Return a structured error response so callers can check response['error']."""
        return {
            "content": "",
            "thinking": "",
            "model": model,
            "tier": tier,
            "effort": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0,
            "stop_reason": "error",
            "error": error_msg,
        }


# Singleton
claude = ClaudeClient()
"""
Claude API wrapper with:
  - Tiered model selection (Opus/Sonnet/Haiku)
  - Prompt caching (cache_control: ephemeral)
  - Adaptive thinking + effort parameter (Claude 4.6)
  - Automatic token usage logging
  - Cost calculation and alerting
  - Error capture to zo_mind_logs + metadata
  - Retry with backoff for transient errors (500, 529)
"""

import anthropic
import asyncio
import logging
import traceback
from typing import Any
from .config import config, AGENT_MODEL_MAP
from . import db

logger = logging.getLogger("zo.claude")

# Effort levels per agent — Builder gets max, others get high
AGENT_EFFORT = {
    "builder": "max",
    "builder_step1": "max",
    "builder_step2": "max",
    "builder_step3": "max",
    "builder_step4": "max",
    "builder_step5": "max",
    "builder_step6": "max",
    "builder_opus": "max",
    "research_a": "high",
    "research_b": "high",
    "ethics": "high",
    "qa": "high",
    "marketing": "high",
    "build-architect": "high",
    "hotfix_diagnose": "high",
    "hotfix_patch": "high",
    "hotfix_verify": "medium",
}

# Retry config: transient errors get retried, client errors don't
RETRYABLE_STATUS_CODES = {500, 502, 503, 529}
MAX_RETRIES = 3
RETRY_DELAYS = [30, 60, 120]  # seconds — exponential backoff


class ClaudeClient:
    """Tiered, cost-tracked Claude API client with adaptive thinking and retry."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
            timeout=600.0,
        )

    async def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_message: str,
        project_id: str | None = None,
        workflow: str = "langgraph",
        max_tokens: int = 16000,
        temperature: float = 0.3,
        use_cache: bool = True,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        """Call Claude with adaptive thinking, effort parameter, retry, and error capture."""

        # Determine model tier and effort
        tier = AGENT_MODEL_MAP.get(agent_name, "sonnet")
        model = config.models.get_model(tier)
        effort = AGENT_EFFORT.get(agent_name, "high")

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

        # Build API call kwargs
        api_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_messages,
            "messages": [{"role": "user", "content": user_message}],
            "thinking": {"type": "adaptive"},
        }

        # Add effort parameter
        if effort:
            api_kwargs["output_config"] = {"effort": effort}

        logger.info(
            "Calling %s (model=%s, tier=%s, effort=%s, max_tokens=%d)",
            agent_name, model, tier, effort, max_tokens,
        )

        # === API call with retry for transient errors ===
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.messages.create(**api_kwargs)
                break  # Success — exit retry loop
            except anthropic.BadRequestError as e:
                # 400 = client error, do NOT retry
                error_msg = f"400 BadRequest for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.RateLimitError as e:
                # 429 = rate limit, retry with backoff
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "429 RateLimit for %s (attempt %d/%d), retrying in %ds...",
                        agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"429: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"429 RateLimit for {agent_name} after {MAX_RETRIES} retries: {str(e)[:200]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        "API %d for %s (attempt %d/%d), retrying in %ds...",
                        e.status_code, agent_name, attempt + 1, MAX_RETRIES, delay,
                    )
                    await self._log_retry(agent_name, workflow, project_id, attempt + 1, f"{e.status_code}: {str(e)[:100]}")
                    await asyncio.sleep(delay)
                    continue
                error_msg = f"API error {e.status_code} for {agent_name}: {str(e)[:300]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
            except Exception as e:
                error_msg = f"Unexpected error for {agent_name}: {str(e)[:300]}\n{traceback.format_exc()[:500]}"
                logger.error(error_msg)
                await self._log_error(agent_name, workflow, project_id, error_msg)
                return self._error_response(model, tier, error_msg)
        else:
            # All retries exhausted
            error_msg = f"All {MAX_RETRIES} retries exhausted for {agent_name}: {str(last_error)[:300]}"
            logger.error(error_msg)
            await self._log_error(agent_name, workflow, project_id, error_msg)
            return self._error_response(model, tier, error_msg)

        # Extract usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Calculate cost
        cost = config.models.calculate_cost(tier, input_tokens, output_tokens, cached_tokens)

        # Log to zo_cost_logs
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
            notes=f"effort={effort},temp={temperature}",
        )

        # Extract text content (skip thinking blocks)
        text_content = ""
        thinking_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking

        # Log to zo_mind_logs (every Mind call tracked)
        # Column names match actual schema: tokens_used, error_message, status
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": input_tokens + output_tokens,
                "cost_usd": cost,
                "project_id": project_id,
                "status": "success",
                "error_message": None,
                "input_summary": user_message[:200] if user_message else "",
                "output_summary": text_content[:200] if text_content else "",
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log to zo_mind_logs: %s", str(exc)[:100])

        # Cost alert
        try:
            threshold = float(await db.get_config("cost_alert_threshold_cad", "5.00"))
            cost_cad = cost * 1.38
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
                    },
                )
        except Exception:
            pass

        return {
            "content": text_content,
            "thinking": thinking_content,
            "model": model,
            "tier": tier,
            "effort": effort,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost,
            "stop_reason": response.stop_reason,
            "error": None,
        }

    async def _log_retry(self, agent_name: str, workflow: str, project_id: str | None, attempt: int, reason: str):
        """Log retry attempts to zo_mind_logs."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "retry",
                "error_message": f"Retry {attempt}/{MAX_RETRIES}: {reason}",
                "input_summary": f"retry_attempt_{attempt}",
                "output_summary": "",
            }).execute()
        except Exception:
            pass

    async def _log_error(self, agent_name: str, workflow: str, project_id: str | None, error_msg: str):
        """Log API errors to zo_mind_logs and project metadata."""
        try:
            db.get_client().table("zo_mind_logs").insert({
                "mind_name": agent_name,
                "action": workflow,
                "tokens_used": 0,
                "cost_usd": 0,
                "project_id": project_id,
                "status": "error",
                "error_message": error_msg[:500],
                "input_summary": "API ERROR",
                "output_summary": error_msg[:200],
            }).execute()
        except Exception as exc:
            logger.warning("Failed to log error to zo_mind_logs: %s", str(exc)[:100])

        # Write error to project metadata
        if project_id:
            try:
                db.get_client().table("zo_projects").update({
                    "metadata": {"error": error_msg[:200]},
                }).eq("project_id", project_id).execute()
            except Exception:
                pass

    def _error_response(self, model: str, tier: str, error_msg: str) -> dict[str, Any]:
        """Return a structured error response so callers can check response['error']."""
        return {
            "content": "",
            "thinking": "",
            "model": model,
            "tier": tier,
            "effort": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0,
            "stop_reason": "error",
            "error": error_msg,
        }


# Singleton
claude = ClaudeClient()
