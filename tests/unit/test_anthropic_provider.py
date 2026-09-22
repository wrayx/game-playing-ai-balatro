"""Tests for the Anthropic provider. No network calls.

Covers the two places the provider can silently corrupt things: converting the
shared GAME_ACTIONS schemas, and normalising a response into the shape the
agent reads.
"""

from __future__ import annotations

import copy

import pytest

from ai_balatro.ai.actions.schemas import GAME_ACTIONS
from ai_balatro.ai.providers.anthropic_provider import (
    API_KEY_ENV_VARS,
    AnthropicProvider,
    _key_from_env,
    _to_anthropic_tool,
)


class Block:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Response:
    def __init__(self, content, stop_reason='end_turn', **kwargs):
        self.content = content
        self.stop_reason = stop_reason
        self.model = 'claude-opus-5'
        self.usage = Block(input_tokens=10, output_tokens=20)
        self.__dict__.update(kwargs)


def provider() -> AnthropicProvider:
    return AnthropicProvider(api_key='test-key')


class TestSchemaConversion:
    def test_parameters_become_input_schema(self):
        tool = _to_anthropic_tool(
            {'name': 'x', 'description': 'd', 'parameters': {'type': 'object'}},
            strict=False,
        )
        assert tool['name'] == 'x'
        assert tool['input_schema'] == {'type': 'object'}
        assert 'parameters' not in tool

    def test_strict_strips_numeric_bounds(self):
        """Anthropic's strict subset rejects 'minimum' on an integer."""
        tool = _to_anthropic_tool(
            {
                'name': 'play_cards',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'indices': {
                            'type': 'array',
                            'items': {'type': 'integer', 'minimum': 0},
                        }
                    },
                },
            },
            strict=True,
        )
        items = tool['input_schema']['properties']['indices']['items']
        assert items == {'type': 'integer'}

    def test_strict_closes_objects(self):
        tool = _to_anthropic_tool(
            {'name': 'x', 'parameters': {'type': 'object', 'properties': {}}},
            strict=True,
        )
        assert tool['input_schema']['additionalProperties'] is False
        assert tool['strict'] is True

    def test_conversion_does_not_mutate_the_shared_schemas(self):
        """GAME_ACTIONS is a module constant the OpenRouter provider also reads."""
        before = copy.deepcopy(GAME_ACTIONS)
        for action in GAME_ACTIONS:
            _to_anthropic_tool(action, strict=True)
        assert GAME_ACTIONS == before

    @pytest.mark.parametrize('action', GAME_ACTIONS)
    def test_every_game_action_converts(self, action):
        tool = _to_anthropic_tool(action, strict=True)
        assert tool['name'] == action['name']
        assert tool['input_schema']['type'] == 'object'


class TestResponseNormalisation:
    def test_text_blocks_join_into_content(self):
        result = provider()._normalise(
            Response([Block(type='text', text='one'), Block(type='text', text='two')])
        )
        assert result.success is True
        assert result.data['content'] == 'one\ntwo'
        assert result.data['function_calls'] == []

    def test_tool_use_becomes_a_function_call(self):
        result = provider()._normalise(
            Response(
                [
                    Block(type='text', text='playing'),
                    Block(
                        type='tool_use',
                        id='tu_1',
                        name='play_cards',
                        input={'indices': [0, 1]},
                    ),
                ],
                stop_reason='tool_use',
            )
        )
        assert result.data['function_calls'] == [
            {'id': 'tu_1', 'name': 'play_cards', 'arguments': {'indices': [0, 1]}}
        ]
        assert result.data['finish_reason'] == 'tool_use'

    def test_thinking_blocks_are_ignored(self):
        """Adaptive thinking is on, so responses carry blocks we do not read."""
        result = provider()._normalise(
            Response(
                [Block(type='thinking', thinking='...'), Block(type='text', text='hi')]
            )
        )
        assert result.data['content'] == 'hi'

    def test_refusal_is_a_failure_not_empty_content(self):
        result = provider()._normalise(
            Response([], stop_reason='refusal', stop_details=Block(category='cyber'))
        )
        assert result.success is False
        assert 'declined' in result.errors[0]

    def test_usage_is_reported(self):
        result = provider()._normalise(Response([Block(type='text', text='x')]))
        assert result.data['usage'] == {'input_tokens': 10, 'output_tokens': 20}


class TestKeyResolution:
    def test_project_variable_wins_over_the_shared_one(self, monkeypatch):
        """The project key is kept out of ANTHROPIC_API_KEY, which other
        Anthropic tooling on the same machine also reads."""
        monkeypatch.setenv('BALATRO_LLM_API_KEY', 'project-key')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'shared-key')
        assert _key_from_env() == 'project-key'

    def test_falls_back_to_the_standard_variable(self, monkeypatch):
        monkeypatch.delenv('BALATRO_LLM_API_KEY', raising=False)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'shared-key')
        assert _key_from_env() == 'shared-key'

    def test_no_key_anywhere(self, monkeypatch):
        for name in API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        assert _key_from_env() is None


def test_uninitialised_provider_fails_cleanly(monkeypatch):
    for name in API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    result = AnthropicProvider().generate_text('hi')
    assert result.success is False
    assert result.errors == ['Provider not initialized']


class TestAdaptiveThinkingSupport:
    """Haiku rejects adaptive thinking with a 400; Opus and Sonnet require it."""

    @pytest.mark.parametrize('model', ['claude-opus-5', 'claude-sonnet-5'])
    def test_current_models_get_modern_controls(self, model):
        assert AnthropicProvider(
            model_name=model, api_key='k'
        )._supports_modern_controls()

    @pytest.mark.parametrize('model', ['claude-haiku-4-5', 'CLAUDE-HAIKU-4-5'])
    def test_haiku_does_not(self, model):
        assert not AnthropicProvider(
            model_name=model, api_key='k'
        )._supports_modern_controls()


class TestSystemPromptHandling:
    """The agent builds OpenAI-style history; Anthropic wants system separately."""

    def _request(self, monkeypatch, context):
        captured = {}

        class StubMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return Response([Block(type='text', text='ok')])

        p = provider()
        p.client = type('C', (), {'messages': StubMessages()})()
        p.is_initialized = True
        p.generate_text('the prompt', context=context)
        return captured

    def test_system_turn_is_hoisted_out_of_history(self, monkeypatch):
        captured = self._request(
            monkeypatch,
            {'history': [{'role': 'system', 'content': 'you are an agent'}]},
        )
        assert captured['system'] == 'you are an agent'
        assert all(m['role'] != 'system' for m in captured['messages'])
        assert captured['messages'][-1] == {'role': 'user', 'content': 'the prompt'}

    def test_non_system_history_is_preserved_in_order(self, monkeypatch):
        captured = self._request(
            monkeypatch,
            {
                'history': [
                    {'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'first'},
                    {'role': 'assistant', 'content': 'reply'},
                ]
            },
        )
        assert [m['role'] for m in captured['messages']] == [
            'user',
            'assistant',
            'user',
        ]

    def test_explicit_system_message_is_combined(self, monkeypatch):
        captured = self._request(
            monkeypatch,
            {
                'history': [{'role': 'system', 'content': 'from history'}],
                'system_message': 'explicit',
            },
        )
        assert captured['system'] == 'from history\n\nexplicit'

    def test_no_system_anywhere_omits_the_parameter(self, monkeypatch):
        captured = self._request(monkeypatch, {})
        assert 'system' not in captured
