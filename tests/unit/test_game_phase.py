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
