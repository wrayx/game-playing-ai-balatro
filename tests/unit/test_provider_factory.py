"""Tests for selecting a provider from the environment."""

from __future__ import annotations

import pytest

from ai_balatro.ai.providers.anthropic_provider import (
    API_KEY_ENV_VARS as ANTHROPIC_KEY_VARS,
)
from ai_balatro.ai.providers.anthropic_provider import AnthropicProvider
from ai_balatro.ai.providers.factory import (
    MODEL_ENV_VAR,
    OPENROUTER_KEY_VAR,
    create_llm_provider,
)
from ai_balatro.ai.providers.openrouter import OpenRouterProvider

ALL_KEY_VARS = (*ANTHROPIC_KEY_VARS, OPENROUTER_KEY_VAR, MODEL_ENV_VAR)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ALL_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_credentials_returns_none_rather_than_a_broken_provider(monkeypatch):
    """Callers can then report the missing key instead of failing mid-request."""
    assert create_llm_provider() is None


def test_anthropic_key_selects_anthropic(monkeypatch):
    monkeypatch.setenv('BALATRO_LLM_API_KEY', 'k')
    assert isinstance(create_llm_provider(), AnthropicProvider)


def test_openrouter_key_selects_openrouter(monkeypatch):
    monkeypatch.setenv(OPENROUTER_KEY_VAR, 'k')
    assert isinstance(create_llm_provider(), OpenRouterProvider)


def test_anthropic_wins_when_both_are_configured(monkeypatch):
    """Anthropic is the one measured against this game's prompts."""
    monkeypatch.setenv('BALATRO_LLM_API_KEY', 'k')
    monkeypatch.setenv(OPENROUTER_KEY_VAR, 'k')
    assert isinstance(create_llm_provider(), AnthropicProvider)


def test_default_model_is_sonnet(monkeypatch):
    monkeypatch.setenv('BALATRO_LLM_API_KEY', 'k')
    assert create_llm_provider().config.model_name == 'claude-sonnet-5'


def test_env_overrides_the_default_model(monkeypatch):
    monkeypatch.setenv('BALATRO_LLM_API_KEY', 'k')
    monkeypatch.setenv(MODEL_ENV_VAR, 'claude-opus-5')
    assert create_llm_provider().config.model_name == 'claude-opus-5'


def test_explicit_argument_beats_the_env_override(monkeypatch):
    monkeypatch.setenv('BALATRO_LLM_API_KEY', 'k')
    monkeypatch.setenv(MODEL_ENV_VAR, 'claude-opus-5')
    provider = create_llm_provider(model_name='claude-haiku-4-5')
    assert provider.config.model_name == 'claude-haiku-4-5'
