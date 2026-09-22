"""Canonical taxonomy for entities-model detections.

Every class the entities model emits contains the substring ``card``, so
substring matching silently folds jokers, tooltips and the deck pile into the
player's hand. Call sites used to carry their own ad-hoc filter, which let the
index space the LLM reasons about drift from the one the mouse clicks --
``play_cards(indices=[2])`` could address a different card than the executor
selected.

All card classification goes through this module so those index spaces are
identical by construction. Matching is exact on the lowercased class name; an
unrecognised class is deliberately classified as nothing rather than guessed
into the hand.
"""

from typing import Iterable, List, Sequence, Set

from .detection import Detection
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Balatro's hand holds poker cards only. Jokers and consumables render in their
# own rows and cannot be played or discarded, so they are not hand cards.
HAND_CARD_CLASSES = frozenset({'poker_card_front'})
JOKER_CLASSES = frozenset({'joker_card'})
CONSUMABLE_CLASSES = frozenset({'planet_card', 'spectral_card', 'tarot_card'})
DESCRIPTION_CLASSES = frozenset({'card_description', 'poker_card_description'})
PILE_CLASSES = frozenset({'poker_card_back', 'poker_card_stack'})
PACK_CLASSES = frozenset({'card_pack'})

KNOWN_CLASSES = (
    HAND_CARD_CLASSES
    | JOKER_CLASSES
    | CONSUMABLE_CLASSES
    | DESCRIPTION_CLASSES
    | PILE_CLASSES
    | PACK_CLASSES
)

# Entities that raise a hover tooltip worth running OCR over.
DESCRIBABLE_CLASSES = HAND_CARD_CLASSES | JOKER_CLASSES | CONSUMABLE_CLASSES

# Class names already reported as unrecognised, so a retrained model that adds
# classes warns once per name instead of once per frame.
_warned_unknown: Set[str] = set()


def _name(detection: Detection) -> str:
    """Get a detection's class name normalised for comparison."""
    return detection.class_name.strip().lower()


def is_hand_card(detection: Detection) -> bool:
    """Check whether a detection is a card the player can play or discard."""
    return _name(detection) in HAND_CARD_CLASSES


def is_joker(detection: Detection) -> bool:
    """Check whether a detection is a joker."""
    return _name(detection) in JOKER_CLASSES


def is_consumable(detection: Detection) -> bool:
    """Check whether a detection is a tarot, planet or spectral card."""
    return _name(detection) in CONSUMABLE_CLASSES


def is_description(detection: Detection) -> bool:
    """Check whether a detection is a hover tooltip rather than an entity."""
    return _name(detection) in DESCRIPTION_CLASSES


def is_pile(detection: Detection) -> bool:
    """Check whether a detection is the deck or draw pile."""
    return _name(detection) in PILE_CLASSES


def is_describable(detection: Detection) -> bool:
    """Check whether hovering a detection is expected to raise a tooltip."""
    return _name(detection) in DESCRIBABLE_CLASSES


def sort_left_to_right(detections: Iterable[Detection]) -> List[Detection]:
    """Sort detections by left edge, the order the player sees them in."""
    return sorted(detections, key=lambda detection: detection.bbox[0])


def hand_cards(detections: Iterable[Detection]) -> List[Detection]:
    """Extract playable hand cards, ordered left to right.

    This ordering defines the index space used by ``play_cards`` and
    ``discard_cards`` and by the game state handed to the LLM. Both sides must
    call this function rather than filtering themselves.

    Args:
        detections: Entities-model detections for a single frame

    Returns:
        Hand card detections sorted left to right
    """
    detections = list(detections)
    _warn_unknown(detections)
    return sort_left_to_right(d for d in detections if is_hand_card(d))


def jokers(detections: Iterable[Detection]) -> List[Detection]:
    """Extract joker detections, ordered left to right."""
    return sort_left_to_right(d for d in detections if is_joker(d))


def consumables(detections: Iterable[Detection]) -> List[Detection]:
    """Extract consumable detections, ordered left to right."""
    return sort_left_to_right(d for d in detections if is_consumable(d))


def descriptions(detections: Iterable[Detection]) -> List[Detection]:
    """Extract hover tooltip detections, ordered left to right."""
    return sort_left_to_right(d for d in detections if is_description(d))


def unknown_classes(detections: Sequence[Detection]) -> Set[str]:
    """Get class names that this taxonomy does not recognise."""
    return {_name(d) for d in detections} - KNOWN_CLASSES


def _warn_unknown(detections: Sequence[Detection]) -> None:
    """Warn once per unrecognised class name so model drift is visible."""
    for class_name in unknown_classes(detections) - _warned_unknown:
        _warned_unknown.add(class_name)
        logger.warning(
            f'Unrecognised entity class {class_name!r}; it will be excluded from '
            f'the hand. Add it to ai_balatro.core.entities if the model changed.'
        )
