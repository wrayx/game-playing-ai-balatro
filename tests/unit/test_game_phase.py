"""Tests for naming the screen the game is showing.

The agent needs the phase to know whether to play cards or press a button; a
wrong phase is why an autonomous run tried to play cards on the Cash Out screen.
"""

from __future__ import annotations

import pytest

from ai_balatro.core.detection import Detection
from ai_balatro.services.game_state_extraction import GameStateExtractionService


def service() -> GameStateExtractionService:
    return GameStateExtractionService.__new__(GameStateExtractionService)


def buttons(*names):
    return [{'class_name': n} for n in names]


def card():
    return Detection(0, 'poker_card_front', 0.9, (100, 400, 190, 520))


def test_cards_in_hand_mean_playing():
    """Play and Discard only render once cards are selected, so a fresh hand
    shows neither -- the cards themselves are the signal."""
    phase = service()._infer_game_phase(
        buttons('button_sort_hand_rank', 'button_options'), [card()]
    )
    assert phase == 'playing'


def test_cash_out_screen():
    assert (
        service()._infer_game_phase(buttons('button_cash_out', 'button_options'), [])
        == 'blind_won'
    )


def test_blind_select_screen():
    assert (
        service()._infer_game_phase(
            buttons('button_level_select', 'button_level_skip'), []
        )
        == 'blind_select'
    )


@pytest.mark.parametrize(
    'button', ['button_store_next_round', 'button_store_reroll', 'button_purchase']
)
def test_shop_screen(button):
    assert service()._infer_game_phase(buttons(button, 'button_options'), []) == 'shop'


def test_sort_button_alone_is_not_playing():
    """'button_sort_hand_rank' contains 'hand'; substring matching read that as
    the playing phase on screens with no hand at all."""
    assert (
        service()._infer_game_phase(buttons('button_sort_hand_rank'), []) != 'playing'
    )


def test_unrecognised_screen():
    assert service()._infer_game_phase(buttons('button_options'), []) == 'unknown'


def test_an_opened_pack_is_not_the_playing_phase():
    """A booster pack shows cards to choose from that look exactly like a hand.

    Read as 'playing', the agent was told to play them and the executor refused
    a selection it could never make.
    """
    phase = service()._infer_game_phase(
        buttons('button_card_pack_skip', 'button_options'), [card(), card()]
    )
    assert phase == 'pack_opening'


def test_a_blind_skip_is_not_a_pack():
    """button_level_skip also maps to 'skip', so match the exact class."""
    assert (
        service()._infer_game_phase(
            buttons('button_level_skip', 'button_level_select'), []
        )
        == 'blind_select'
    )


def test_a_priced_item_identifies_the_shop():
    """More reliable than the shop's buttons: the UI model misses Next Round
    and Reroll in some states, leaving nothing to identify the screen."""
    stock = [(card(), card())]  # (item, price tag) pairs
    phase = service()._infer_game_phase(buttons('button_options'), [], stock)
    assert phase == 'shop'


def test_stock_outranks_a_hand_on_screen():
    """Owned cards can be visible in the shop; a price tag cannot."""
    stock = [(card(), card())]
    assert service()._infer_game_phase(buttons(), [card()], stock) == 'shop'


def test_no_stock_leaves_the_other_phases_alone():
    assert (
        service()._infer_game_phase(buttons('button_cash_out'), [], []) == 'blind_won'
    )
