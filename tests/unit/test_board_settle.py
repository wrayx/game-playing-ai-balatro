"""Tests for waiting out the game's animations after an action.

Capturing mid-animation is how an agent ends up reading a partial hand whose
tooltips are blank, or a screen whose only buttons are Options and Run Info.
"""

from __future__ import annotations

import numpy as np

from ai_balatro.ai.actions.card_action_engine import CardActionEngine
from ai_balatro.ai.actions.card_detector import CardPositionDetector
from ai_balatro.core.detection import Detection

FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


class StubDetector:
    """Returns a different board on each call, scripted by the test."""

    def __init__(self, entity_frames, ui_frames):
        self._entities = list(entity_frames)
        self._ui = list(ui_frames)

    def detect_entities(self, _frame, *a, **k):
        return self._entities.pop(0) if len(self._entities) > 1 else self._entities[0]

    def detect_ui(self, _frame, *a, **k):
        return self._ui.pop(0) if len(self._ui) > 1 else self._ui[0]


def card(x):
    return Detection(0, 'poker_card_front', 0.9, (x, 400, x + 90, 520))


def button(name):
    return Detection(0, name, 0.9, (0, 0, 10, 10))


def engine(entity_frames, ui_frames):
    instance = CardActionEngine.__new__(CardActionEngine)
    instance.multi_detector = StubDetector(entity_frames, ui_frames)
    instance.yolo_detector = None
    instance.position_detector = CardPositionDetector()
    instance.screen_capture = type(
        'C', (), {'capture_once': staticmethod(lambda: FRAME)}
    )()
    return instance


def test_settles_once_the_hand_stops_changing():
    dealing = [[card(100)], [card(100), card(200)], [card(100), card(200)]]
    e = engine(dealing, [[button('button_play')]])
    assert e._wait_until_settled(timeout=3.0, interval=0.01) is True


def test_settles_on_a_screen_change_with_no_hand():
    """Winning a blind empties the hand, so the hand count alone never settles;
    the Cash Out button appearing is the signal."""
    ui = [[], [button('button_cash_out')], [button('button_cash_out')]]
    e = engine([[]], ui)
    assert e._wait_until_settled(timeout=3.0, interval=0.01) is True


def test_reports_failure_when_the_board_keeps_changing():
    flickering = [[card(100)], [card(100), card(200)]] * 200
    e = engine(flickering, [[button('button_play')]])
    assert e._wait_until_settled(timeout=0.3, interval=0.01) is False


def test_signature_includes_both_hand_and_buttons():
    e = engine([[card(100)]], [[button('button_cash_out'), button('button_options')]])
    count, buttons = e._board_signature(FRAME)
    assert count == 1
    assert buttons == frozenset({'button_cash_out', 'button_options'})
