"""Tests for resolving a requested button type to a UI-model class.

Substring matching against class names collided: 'shop' matched
button_store_reroll, 'play' matched button_new_run_play, and 'discard' matched
the ui_data_discards_left counter.
"""

from __future__ import annotations

import pytest

from ai_balatro.ai.actions.schemas import (
    AGENT_BUTTON_TYPES,
    BUTTON_CONFIG,
    GAME_ACTIONS,
)

UI_MODEL_BUTTON_CLASSES = {
    'button_back',
    'button_card_pack_skip',
    'button_cash_out',
    'button_discard',
    'button_level_select',
    'button_level_skip',
    'button_main_menu',
    'button_main_menu_play',
    'button_new_run',
    'button_new_run_play',
    'button_options',
    'button_play',
    'button_purchase',
    'button_run_info',
    'button_sell',
    'button_sort_hand_rank',
    'button_sort_hand_suits',
    'button_store_next_round',
    'button_store_reroll',
    'button_use',
}


def _enum():
    action = next(a for a in GAME_ACTIONS if a['name'] == 'click_button')
    return action['parameters']['properties']['button_type']['enum']


def test_every_offered_button_is_executable():
    """The enum and the executor's validation set must not drift apart.

    They did: the schema offered sort_hand_rank while the config keyed it as
    button_sort_hand_rank, so every such call was rejected.
    """
    for button_type in _enum():
        assert button_type in BUTTON_CONFIG


def test_enum_matches_the_declared_agent_subset():
    assert _enum() == AGENT_BUTTON_TYPES


@pytest.mark.parametrize('button_type', sorted(BUTTON_CONFIG))
def test_classes_are_real_ui_model_classes(button_type):
    for class_name in BUTTON_CONFIG[button_type]['classes']:
        assert class_name in UI_MODEL_BUTTON_CLASSES


def test_no_class_serves_two_button_types():
    """An ambiguous class would make the resolved button order-dependent."""
    seen = {}
    for button_type, config in BUTTON_CONFIG.items():
        for class_name in config['classes']:
            assert class_name not in seen, (
                f'{class_name} claimed by {seen.get(class_name)} and {button_type}'
            )
            seen[class_name] = button_type


def test_run_can_be_advanced():
    """Without these the agent stalls the moment a blind resolves."""
    assert 'cash_out' in _enum()
    assert 'level_select' in _enum()


def test_shop_entry_is_not_offered():
    """Balatro has no shop-entry button; 'shop' used to reroll the shop."""
    assert 'shop' not in _enum()
    assert 'shop' not in BUTTON_CONFIG


def test_every_ui_button_class_resolves_to_a_type():
    """A class resolving to 'unknown' is a button the code cannot act on."""
    from ai_balatro.ai.actions.button_detector import ButtonDetector
    from ai_balatro.core.detection import Detection

    detector = ButtonDetector()
    for class_name in UI_MODEL_BUTTON_CLASSES:
        detection = Detection(0, class_name, 0.9, (0, 0, 10, 10))
        assert detector._get_button_type(detection) != 'unknown', class_name


@pytest.mark.parametrize('class_name', ['button_main_menu_play', 'button_new_run_play'])
def test_menu_play_buttons_do_not_resolve_to_play(class_name):
    """Asking to play a hand must never find a button that starts a new run."""
    from ai_balatro.ai.actions.button_detector import ButtonDetector
    from ai_balatro.core.detection import Detection

    detection = Detection(0, class_name, 0.9, (0, 0, 10, 10))
    assert ButtonDetector()._get_button_type(detection) != 'play'


def test_button_class_map_agrees_with_the_config():
    """The map used to be hand-maintained and drifted out of sync."""
    from ai_balatro.ai.actions.button_detector import ButtonDetector

    expected = {
        class_name.lower(): button_type
        for button_type, config in BUTTON_CONFIG.items()
        for class_name in config['classes']
    }
    assert ButtonDetector().button_class_map == expected
