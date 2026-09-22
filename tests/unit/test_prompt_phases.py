"""The prompt must suit the screen the game is on.

An autonomous run called play_cards on the Cash Out screen, where there is no
hand: the prompt carried the poker instructions regardless of phase.
"""

from __future__ import annotations

import pytest

from ai_balatro.ai.agents.balatro_agent import BalatroReasoningAgent


def agent():
    return BalatroReasoningAgent.__new__(BalatroReasoningAgent)


def state(phase, cards=0):
    return {
        'cards': [
            {
                'index': i,
                'class_name': 'poker_card_front',
                'confidence': 0.9,
                'description_text': f'{i}of Hearts +{i} chips',
                'parsed_description': None,
            }
            for i in range(cards)
        ],
        'jokers': [],
        'ui_buttons': [],
        'game_phase': phase,
        'ui_text_elements': [],
    }


def test_playing_phase_keeps_the_poker_instructions():
    prompt = agent()._create_analysis_prompt(state('playing', cards=8))
    assert 'POKER OBJECTIVES' in prompt
    assert 'ACTION INSTRUCTIONS' in prompt


def test_playing_phase_substitutes_its_placeholders():
    """The block is a module constant, so it needs formatting, not just
    interpolation by the surrounding f-string."""
    prompt = agent()._create_analysis_prompt(state('playing', cards=8))
    assert '{poker_objectives}' not in prompt


@pytest.mark.parametrize(
    ('phase', 'button'),
    [
        ('blind_won', 'cash_out'),
        ('blind_select', 'level_select'),
        ('shop', 'next'),
    ],
)
def test_non_playing_phases_name_the_button_to_press(phase, button):
    prompt = agent()._create_analysis_prompt(state(phase))
    assert 'WHAT TO DO NOW' in prompt
    assert f"button_type='{button}'" in prompt


@pytest.mark.parametrize('phase', ['blind_won', 'blind_select', 'shop', 'unknown'])
def test_non_playing_phases_drop_the_poker_instructions(phase):
    prompt = agent()._create_analysis_prompt(state(phase))
    assert 'POKER OBJECTIVES' not in prompt
    assert 'ACTION INSTRUCTIONS' not in prompt


def test_unknown_phase_says_not_to_play_cards():
    prompt = agent()._create_analysis_prompt(state('unknown'))
    assert 'Do not try to play or discard cards' in prompt
