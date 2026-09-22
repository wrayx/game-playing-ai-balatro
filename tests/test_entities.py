"""Tests for the shared entity taxonomy.

The regression these lock down: the prompt builder and the action executor must
derive hand card indices from the same filter. When they disagree, the LLM asks
to play index 2 and the mouse clicks a different card.
"""

import pytest

from ai_balatro.ai.actions.card_detector import CardPositionDetector
from ai_balatro.core import entities
from ai_balatro.core.detection import Detection

# Every class the v2 entities model emits.
ENTITIES_MODEL_CLASSES = [
    'card_description',
    'card_pack',
    'joker_card',
    'planet_card',
    'poker_card_back',
    'poker_card_description',
    'poker_card_front',
    'poker_card_stack',
    'spectral_card',
    'tarot_card',
]


def det(class_name: str, x: int, y: int = 700) -> Detection:
    """Build a detection at a given horizontal position."""
    return Detection(0, class_name, 0.9, (x, y, x + 100, y + 140))


def typical_frame() -> list:
    """A frame as it appears mid-blind.

    Jokers along the top, the hand along the bottom, the deck pile at the right,
    and one tooltip raised by the hover sweep.
    """
    return [
        det('joker_card', 100, 100),
        det('joker_card', 220, 100),
        det('poker_card_front', 400),
        det('card_description', 420, 480),
        det('poker_card_front', 530),
        det('poker_card_front', 660),
        det('poker_card_stack', 1700),
        det('poker_card_back', 1700, 660),
    ]


class TestTaxonomy:
    """Classification of the entities model's class names."""

    def test_every_model_class_is_recognised(self):
        detections = [det(name, 0) for name in ENTITIES_MODEL_CLASSES]
        assert entities.unknown_classes(detections) == set()

    @pytest.mark.parametrize('class_name', ENTITIES_MODEL_CLASSES)
    def test_class_falls_in_exactly_one_bucket(self, class_name):
        detection = det(class_name, 0)
        buckets = [
            entities.is_hand_card(detection),
            entities.is_joker(detection),
            entities.is_consumable(detection),
            entities.is_description(detection),
            entities.is_pile(detection),
            class_name in entities.PACK_CLASSES,
        ]
        assert sum(buckets) == 1

    def test_only_poker_card_front_is_playable(self):
        playable = [
            name
            for name in ENTITIES_MODEL_CLASSES
            if entities.is_hand_card(det(name, 0))
        ]
        assert playable == ['poker_card_front']

    def test_jokers_are_reported_separately(self):
        frame = typical_frame()
        assert len(entities.jokers(frame)) == 2
        assert all(not entities.is_hand_card(d) for d in entities.jokers(frame))

    def test_unknown_class_is_excluded_not_guessed(self):
        frame = [det('poker_card_front', 100), det('enhanced_card_v3', 200)]
        assert len(entities.hand_cards(frame)) == 1
        assert entities.unknown_classes(frame) == {'enhanced_card_v3'}

    def test_hand_cards_are_sorted_left_to_right(self):
        frame = [det('poker_card_front', x) for x in (900, 100, 500)]
        assert [c.bbox[0] for c in entities.hand_cards(frame)] == [100, 500, 900]


class TestIndexSpacesAgree:
    """The prompt side and the click side must index the same cards."""

    def test_executor_filter_matches_taxonomy(self):
        frame = typical_frame()
        assert CardPositionDetector().get_hand_cards(frame) == entities.hand_cards(
            frame
        )

    def test_tooltip_does_not_shift_indices(self):
        """The hover sweep raises tooltips deliberately; they must not renumber."""
        frame = typical_frame()
        without_tooltip = [d for d in frame if d.class_name != 'card_description']
        assert entities.hand_cards(frame) == entities.hand_cards(without_tooltip)

    def test_jokers_and_pile_do_not_shift_indices(self):
        frame = typical_frame()
        hand_only = [d for d in frame if d.class_name == 'poker_card_front']
        assert entities.hand_cards(frame) == entities.hand_cards(hand_only)
