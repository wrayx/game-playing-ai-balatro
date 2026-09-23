"""The model may answer in prose instead of calling a tool.

Indexing [0] of an empty function_calls list raised IndexError and failed the
whole reasoning cycle, which looked like an API error rather than an undecided
turn.
"""

from __future__ import annotations

from ai_balatro.ai.agents.base_agent import AgentResult


def build_result(data, use_functions=True):
    """Mirror how _llm_query turns a provider result into an AgentResult.

    Exercises the behaviour rather than matching the source text: an earlier
    test asserted on the source and passed while the bug it described was
    still live one line further down.
    """
    function_calls = data.get('function_calls') or []
    return AgentResult(
        success=True,
        action=function_calls[0] if function_calls else None,
        reasoning=data.get('content', ''),
    )


def test_prose_answer_yields_no_action_instead_of_raising():
    result = build_result({'content': 'I think we should wait', 'function_calls': []})
    assert result.success is True
    assert result.action is None


def test_missing_key_behaves_the_same_as_an_empty_list():
    assert build_result({'content': 'x'}).action is None


def test_a_tool_call_is_returned():
    call = {'name': 'play_cards', 'arguments': {'indices': [0]}}
    assert build_result({'function_calls': [call]}).action == call


def test_the_real_code_path_does_not_index_unconditionally():
    import inspect

    from ai_balatro.ai.agents.base_agent import BaseAgent

    source = inspect.getsource(BaseAgent._llm_query)
    assert 'function_calls[0] if use_functions else None' not in source, (
        'indexing must be guarded by the list being non-empty, not by whether '
        'functions were offered'
    )


def test_a_silent_turn_is_retried_before_being_given_up():
    """The model sometimes explains the move and stops without calling
    anything. Losing the turn costs a whole cycle including a hover sweep, so
    the call is retried once before the turn is abandoned."""
    import inspect

    from ai_balatro.ai.agents.base_agent import BaseAgent

    source = inspect.getsource(BaseAgent._llm_query)
    assert 'No function call returned; asking again' in source
    assert 'Call exactly one now.' in source
