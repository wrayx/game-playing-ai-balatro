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


def shop_state(items=None, jokers=None, cash='14'):
    return {
        'cards': [],
        'jokers': jokers or [],
        'ui_buttons': [{'class_name': 'button_store_next_round'}],
        'game_phase': 'shop',
        'ui_text_elements': [{'class_name': 'ui_data_cash', 'text': cash}],
        'shop_items': items or [],
    }


def shop_item(index, class_name, price, text=''):
    return {
        'index': index,
        'class_name': class_name,
        'price': price,
        'description_text': text,
        'description_detected': bool(text),
    }


class TestShopPrompt:
    def test_lists_each_item_with_its_price_and_effect(self):
        prompt = agent()._create_analysis_prompt(
            shop_state([shop_item(0, 'joker_card', '5', 'Blueprint Copies ability')])
        )
        assert 'Item 0: joker_card costs $5' in prompt
        assert 'Blueprint Copies ability' in prompt

    def test_says_when_a_description_could_not_be_read(self):
        prompt = agent()._create_analysis_prompt(
            shop_state([shop_item(0, 'card_pack', '4')])
        )
        assert 'description unreadable' in prompt

    def test_shows_cash_and_owned_jokers(self):
        prompt = agent()._create_analysis_prompt(
            shop_state(
                jokers=[
                    {
                        'index': 0,
                        'class_name': 'joker_card',
                        'confidence': 0.95,
                        'description_text': 'Raised Fist adds double the rank',
                    }
                ],
                cash='23',
            )
        )
        assert 'You have $23' in prompt
        assert '1 of 5 slots' in prompt
        assert 'Raised Fist' in prompt

    def test_offers_both_buying_and_leaving(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'buy_item(index=N)' in prompt
        assert "click_button(button_type='next')" in prompt

    def test_explains_interest_so_it_does_not_spend_to_zero(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert '$1 for every $5' in prompt

    def test_warns_about_run_specific_stickers(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'Perishable' in prompt
        assert 'Rental' in prompt

    def test_steers_away_from_packs_we_cannot_open(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'do not buy them' in prompt

    def test_shop_prompt_drops_the_poker_instructions(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'POKER OBJECTIVES' not in prompt


def test_pack_opening_tells_it_to_skip_not_play():
    prompt = agent()._create_analysis_prompt(state('pack_opening'))
    assert "button_type='skip'" in prompt
    assert 'not your hand' in prompt
    assert 'POKER OBJECTIVES' not in prompt
