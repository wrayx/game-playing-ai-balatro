"""Selection of an LLM provider from the environment."""

import os
from typing import Any, Optional

from .anthropic_provider import API_KEY_ENV_VARS as ANTHROPIC_KEY_VARS
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .openrouter import OpenRouterProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)

#: Overrides the provider's own default model, whichever provider is selected.
MODEL_ENV_VAR = 'BALATRO_LLM_MODEL'

OPENROUTER_KEY_VAR = 'OPENROUTER_API_KEY'


def create_llm_provider(
    model_name: Optional[str] = None, **kwargs: Any
) -> Optional[LLMProvider]:
    """Build the provider whose credentials are present.

    Anthropic is preferred when both are configured: it is the one measured
    against this game's prompts. Returns None when nothing is configured, so
    callers can report the missing key rather than fail on the first request.

    Args:
        model_name: Model id; falls back to BALATRO_LLM_MODEL, then to the
            selected provider's own default
        **kwargs: Passed through to the provider constructor

    Returns:
        An uninitialised provider, or None when no key is available
    """
    model_name = model_name or os.getenv(MODEL_ENV_VAR) or None

    if any(os.getenv(name) for name in ANTHROPIC_KEY_VARS):
        logger.info('Using the Anthropic provider')
        if model_name:
            return AnthropicProvider(model_name=model_name, **kwargs)
        return AnthropicProvider(**kwargs)

    if os.getenv(OPENROUTER_KEY_VAR):
        logger.info('Using the OpenRouter provider')
        if model_name:
            return OpenRouterProvider(model_name=model_name, **kwargs)
        return OpenRouterProvider(**kwargs)

    logger.error(
        'No LLM credentials found. Set one of: %s',
        ', '.join((*ANTHROPIC_KEY_VARS, OPENROUTER_KEY_VAR)),
    )
    return None
