"""The model may answer in prose instead of calling a tool.

Indexing [0] of an empty function_calls list raised IndexError and failed the
whole reasoning cycle, which looked like an API error rather than an
undecided turn.
"""

from __future__ import annotations

import inspect

from ai_balatro.ai.agents.base_agent import BaseAgent


def test_empty_function_calls_is_not_indexed():
    source = inspect.getsource(BaseAgent._llm_query)
    assert "result.data.get('function_calls', [{}])[0]" not in source
    assert "result.data.get('function_calls') or []" in source


def test_missing_and_empty_are_handled_the_same_way():
    """get(key, default) does not fire for a present-but-empty list."""
    assert ({}.get('function_calls') or []) == []
    assert ({'function_calls': []}.get('function_calls') or []) == []
    assert ({'function_calls': [{'name': 'play_cards'}]}.get('function_calls') or [])[
        0
    ] == {'name': 'play_cards'}
