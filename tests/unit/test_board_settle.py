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
    instance.ui_text_service = None  # score unreadable in tests
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


def test_signature_includes_hand_buttons_and_score():
    e = engine([[card(100)]], [[button('button_cash_out'), button('button_options')]])
    count, buttons, score = e._board_signature(FRAME)
    assert count == 1
    assert buttons == frozenset({'button_cash_out', 'button_options'})
    assert score == ''  # no OCR service wired in the test


def test_a_changing_score_keeps_the_board_unsettled():
    """The hand does not change while a hand scores, so the score is the only
    signal that the animation is still running."""
    import itertools

    scores = itertools.cycle(['100', '232', '410', '590'])
    e = engine([[card(100)]], [[button('button_play')]])
    e.ui_text_service = object()
    e._round_score = lambda *a, **k: next(scores)

    assert e._wait_until_settled(timeout=0.3, interval=0.001) is False


def test_does_not_settle_on_only_the_permanent_buttons():
    """Options and Run Info are on screen in every phase. A board showing just
    those is mid-transition -- accepting it is how an agent read the Cash Out
    screen before Cash Out had rendered, and guessed the wrong button."""
    persistent = [button('button_options'), button('button_run_info')]
    e = engine([[]], [persistent, persistent, persistent])
    assert e._wait_until_settled(timeout=0.3, interval=0.01) is False


def test_settles_once_a_real_button_appears():
    persistent = [button('button_options'), button('button_run_info')]
    ready = persistent + [button('button_cash_out')]
    e = engine([[]], [persistent, ready, ready])
    assert e._wait_until_settled(timeout=3.0, interval=0.01) is True


def test_a_single_repeat_is_not_enough():
    """While a hand scores, the cards left behind sit still long enough to
    match twice; the board looked ready while the blind was ending.

    This board plateaus for exactly two samples before changing again, over
    and over, so it must never be called settled.
    """
    import itertools

    one = [card(100)]
    two = [card(100), card(200)]
    cycling = itertools.cycle([one, one, two, two])

    e = engine([one], [[button('button_play')]])
    # Cycle forever rather than exhausting a scripted list, which would end up
    # repeating its last frame and settle for the wrong reason.
    e.multi_detector.detect_entities = lambda *a, **k: next(cycling)

    assert e._wait_until_settled(timeout=0.3, interval=0.001) is False


def test_settles_after_enough_consecutive_matches():
    steady = [[card(100), card(200)]]
    e = engine(steady, [[button('button_play')]])
    assert e._wait_until_settled(timeout=3.0, interval=0.01) is True
