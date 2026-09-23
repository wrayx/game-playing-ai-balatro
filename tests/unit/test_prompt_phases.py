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

    def test_packs_are_presented_as_buyable(self):
        """Packs were refused while their selection screen was unhandled."""
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'do not buy them' not in prompt
        assert 'Booster packs can be opened now' in prompt

    def test_shop_prompt_drops_the_poker_instructions(self):
        prompt = agent()._create_analysis_prompt(shop_state())
        assert 'POKER OBJECTIVES' not in prompt


def test_pack_opening_offers_choosing_and_skipping():
    st = state('pack_opening')
    st['pack_items'] = [
        {
            'index': 0,
            'class_name': 'joker_card',
            'description_text': 'Blueprint copies',
        },
        {
            'index': 1,
            'class_name': 'planet_card',
            'description_text': 'Earth levels Full House',
        },
    ]
    prompt = agent()._create_analysis_prompt(st)
    flat = ' '.join(prompt.split())

    assert 'choose_from_pack(index=N)' in prompt
    assert "button_type='skip'" in prompt
    assert 'not your hand' in flat
    assert 'Item 0: joker_card' in prompt
    assert 'Earth levels Full House' in prompt
    assert 'POKER OBJECTIVES' not in prompt


def test_pack_prompt_reports_slot_pressure():
    """A joker cannot be taken with five already held."""
    st = state('pack_opening')
    st['pack_items'] = [
        {'index': 0, 'class_name': 'joker_card', 'description_text': 'x'}
    ]
    st['jokers'] = [
        {
            'index': i,
            'class_name': 'joker_card',
            'confidence': 0.9,
            'description_text': 'j',
        }
        for i in range(5)
    ]
    prompt = agent()._create_analysis_prompt(st)
    assert '5 of 5 jokers' in prompt


def test_playing_prompt_shows_what_each_joker_does():
    """A joker's rules are re-read from its tooltip every turn, so the model
    need not remember them -- but only if the prompt actually carries them.
    'Ride the Bus' as a bare class name says nothing about avoiding face cards.
    """
    st = state('playing', cards=3)
    st['jokers'] = [
        {
            'index': 0,
            'class_name': 'joker_card',
            'confidence': 0.95,
            'description_text': 'Ride the Bus gains +1 Mult per consecutive '
            'hand played without a scoring face card',
        }
    ]
    prompt = agent()._create_analysis_prompt(st)
    assert 'without a scoring face card' in prompt


def test_unreadable_joker_says_so_rather_than_looking_informative():
    st = state('playing', cards=3)
    st['jokers'] = [
        {
            'index': 0,
            'class_name': 'joker_card',
            'confidence': 0.9,
            'description_text': '',
        }
    ]
    prompt = agent()._create_analysis_prompt(st)
    assert 'effect unreadable' in prompt


def test_shop_prompt_names_the_interest_threshold():
    prompt = agent()._create_analysis_prompt(shop_state())
    assert '$25' in prompt


def test_shop_prompt_warns_when_consumable_slots_are_full():
    """Balatro refuses to sell a consumable with no slot free, yet still shows
    a Buy button that does nothing -- so the agent retried the same purchase
    and only saw that its money never moved."""
    st = shop_state([shop_item(0, 'tarot_card', '3', 'Strength')])
    st['consumables'] = [{'index': 0}, {'index': 1}]
    prompt = agent()._create_analysis_prompt(st)
    assert '2 of 2 slots' in prompt
    assert 'Both slots are full' in prompt


def test_shop_prompt_does_not_warn_with_a_free_slot():
    st = shop_state([shop_item(0, 'tarot_card', '3', 'Strength')])
    st['consumables'] = [{'index': 0}]
    prompt = agent()._create_analysis_prompt(st)
    assert '1 of 2 slots' in prompt
    assert 'Both slots are full' not in prompt
