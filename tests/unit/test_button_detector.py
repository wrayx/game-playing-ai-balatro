"""Tests for distinguishing buttons from UI readouts.

The UI model emits both 'button_*' controls and 'ui_*' readouts. A readout that
happens to contain a button word is not clickable, and clicking one would look
like a successful action while doing nothing.
"""

from __future__ import annotations

import pytest

from ai_balatro.ai.actions.button_detector import ButtonDetector
from ai_balatro.core.detection import Detection


def det(class_name: str) -> Detection:
    return Detection(0, class_name, 0.95, (10, 10, 90, 50))


@pytest.mark.parametrize(
    'class_name',
    [
        'button_discard',
        'button_play',
        'button_store_next_round',
        'button_sort_hand_rank',
    ],
)
def test_button_classes_are_buttons(class_name):
    assert ButtonDetector()._is_button(det(class_name)) is True


@pytest.mark.parametrize(
    'class_name',
    [
        'ui_data_discards_left',  # contains 'discard'
        'ui_data_hands_left',
        'ui_score_target_score',
        'ui_round_ante_current',
        'poker_card_front',
    ],
)
def test_readouts_are_not_buttons(class_name):
    assert ButtonDetector()._is_button(det(class_name)) is False


@pytest.mark.parametrize('class_name', ['play_button', 'discard_button', 'shop_button'])
def test_invented_alias_names_are_not_buttons(class_name):
    """The UI model emits 'button_play', never 'play_button'.

    A hand-maintained alias table used to accept both. It carried its own
    naming scheme, drifted from BUTTON_CONFIG, and was the route by which
    button_main_menu_play resolved to 'play'. The map is now derived from the
    config, so only real class names resolve.
    """
    assert ButtonDetector()._is_button(det(class_name)) is False
