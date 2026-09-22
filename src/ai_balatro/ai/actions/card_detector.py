"""Card position detection and sorting functionality."""

from typing import List

from ...core import entities
from ...core.detection import Detection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class CardPositionDetector:
    """Detects and sorts hand cards by position from left to right."""

    def get_hand_cards(self, detections: List[Detection]) -> List[Detection]:
        """
        Extract hand cards from detection results and sort them from left to right.

        Delegates to ``ai_balatro.core.entities`` so the indices returned here are
        the same ones the LLM is shown in the game state. Jokers, consumables and
        the deck pile are excluded: none of them can be played or discarded, and
        counting them would shift every index after them.

        Args:
            detections: YOLO detection results

        Returns:
            Sorted list of hand card Detection objects
        """
        hand_cards = entities.hand_cards(detections)

        if not hand_cards:
            logger.warning('No playable hand cards detected')
            return []

        logger.info(f'Detected {len(hand_cards)} hand cards:')
        for i, card in enumerate(hand_cards):
            logger.info(
                f'  Position {i}: {card.class_name} at {card.center} '
                f'(confidence: {card.confidence:.3f})'
            )

        return hand_cards
