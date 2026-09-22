"""Tests for reading which cards the game shows as selected.

Selecting a card lifts it; a click that macOS swallowed leaves it where it was.
Pressing Play one card short spends a hand on a hand nobody chose, so this
reading is what makes the action path safe to commit.
"""

from __future__ import annotations

import numpy as np

from ai_balatro.ai.actions.card_action_engine import CardActionEngine
from ai_balatro.ai.actions.card_detector import CardPositionDetector
from ai_balatro.core.detection import Detection

FRAME = np.zeros((10, 10, 3), dtype=np.uint8)
RESTING_Y = 400
LIFT = 25


class StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect_entities(self, _frame, *args, **kwargs):
        return list(self._detections)


def card(x: int, y: int) -> Detection:
    return Detection(0, 'poker_card_front', 0.95, (x, y, x + 90, y + 120))


def hand(*lifts: int):
    """Eight cards left to right, each raised by the given amount."""
    return [card(240 + i * 65, RESTING_Y - lift) for i, lift in enumerate(lifts)]


def engine_for(current):
    instance = CardActionEngine.__new__(CardActionEngine)
    instance.multi_detector = StubDetector(current)
    instance.yolo_detector = None
    instance.position_detector = CardPositionDetector()
    return instance


def test_nothing_selected_when_all_cards_rest_level():
    baseline = hand(0, 0, 0, 0)
    assert engine_for(baseline)._selected_indices(baseline, FRAME) == []


def test_reports_the_lifted_cards():
    baseline = hand(0, 0, 0, 0)
    current = hand(LIFT, 0, LIFT, 0)
    assert engine_for(current)._selected_indices(baseline, FRAME) == [0, 2]


def test_hover_lift_is_not_mistaken_for_selection():
    """Hovering raises a card a few pixels; only selection clears the threshold."""
    baseline = hand(0, 0, 0, 0)
    current = hand(0, 5, 0, 0)
    assert engine_for(current)._selected_indices(baseline, FRAME) == []


def test_detects_a_card_that_was_already_selected():
    """The reading is absolute, so a leftover selection is still seen."""
    baseline = hand(LIFT, 0, 0, 0)
    assert engine_for(baseline)._selected_indices(baseline, FRAME) == [0]


def test_ignores_a_card_that_is_no_longer_detected():
    baseline = hand(0, 0, 0, 0)
    current = [c for c in hand(0, LIFT, 0, 0) if c.bbox[0] != 240]
    assert engine_for(current)._selected_indices(baseline, FRAME) == [1]


def test_empty_detection_reports_nothing_selected():
    baseline = hand(0, 0)
    assert engine_for([])._selected_indices(baseline, FRAME) == []


class TestCorrectingInBothDirections:
    """A stray selection must be cleared, not just missing ones added.

    Clicking only the missing cards leaves an extra selected, so the check that
    follows refuses again -- and keeps refusing, since nothing ever clears it.
    """

    def test_extra_selection_is_visible_to_the_check(self):
        baseline = hand(0, 0, 0, 0)
        current = hand(LIFT, 0, 0, LIFT)
        selected = engine_for(current)._selected_indices(baseline, FRAME)
        wanted = [3]
        assert selected == [0, 3]
        assert [i for i in selected if i not in wanted] == [0]

    def test_nothing_to_correct_when_the_selection_already_matches(self):
        baseline = hand(0, 0, 0, 0)
        current = hand(0, LIFT, 0, LIFT)
        selected = engine_for(current)._selected_indices(baseline, FRAME)
        wanted = [1, 3]
        assert selected == wanted
        assert [i for i in wanted if i not in selected] == []
        assert [i for i in selected if i not in wanted] == []


class TestArcCurvature:
    """The hand is fanned, so a central card sits above the outer edges.

    Measured live: five selected cards were lifted 23-32px while an unselected
    central card sat 14px above the low right edge. Reading that as a sixth
    selection made the executor refuse a legal play.
    """

    def test_arc_lift_does_not_produce_a_sixth_selection(self):
        baseline = hand(0, 0, 0, 0, 0, 0, 0, 0)
        current = hand(23, 25, 28, 32, 14, 25, 0, 1)
        assert engine_for(current)._selected_indices(baseline, FRAME) == [0, 1, 2, 3, 5]

    def test_never_reports_more_than_the_game_allows(self):
        baseline = hand(0, 0, 0, 0, 0, 0, 0, 0)
        current = hand(30, 29, 28, 27, 26, 25, 24, 23)
        selected = engine_for(current)._selected_indices(baseline, FRAME)
        assert len(selected) <= CardActionEngine.MAX_SELECTED_CARDS

    def test_five_genuine_selections_are_all_kept(self):
        baseline = hand(0, 0, 0, 0, 0, 0, 0, 0)
        current = hand(25, 0, 25, 0, 25, 0, 25, 25)
        assert engine_for(current)._selected_indices(baseline, FRAME) == [0, 2, 4, 6, 7]
