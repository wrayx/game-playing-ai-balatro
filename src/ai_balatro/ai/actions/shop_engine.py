"""Buying from the Balatro shop."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core import entities
from ...core.detection import Detection
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...services.ui_text_service import UITextExtractionService
from ...utils.logger import get_logger
from .button_detector import ButtonDetector
from .mouse_controller import MouseController

logger = get_logger(__name__)


class ShopActionEngine:
    """Selects and buys shop items, verifying each step against the screen."""

    #: How long to wait for the shop to stop animating after a click.
    SETTLE_TIMEOUT = 6.0

    #: How far the Buy button's centre may sit outside the item it belongs to.
    BUTTON_ALIGNMENT_SLACK = 40

    def __init__(
        self,
        screen_capture: ScreenCapture,
        multi_detector: MultiYOLODetector,
        mouse_controller: MouseController,
        button_detector: Optional[ButtonDetector] = None,
        ui_text_service: Optional[UITextExtractionService] = None,
    ):
        self.screen_capture = screen_capture
        self.multi_detector = multi_detector
        self.mouse_controller = mouse_controller
        self.button_detector = button_detector or ButtonDetector(multi_detector)
        self.ui_text_service = ui_text_service

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def shop_items(self, frame: np.ndarray) -> List[Tuple[Detection, Detection]]:
        """Items currently for sale, left to right, each with its price tag."""
        priced = entities.pair_with_prices(
            entities.shop_item_candidates(self.multi_detector.detect_entities(frame)),
            entities.price_tags(self.multi_detector.detect_ui(frame)),
        )
        return [(item, tag) for item, tag in priced if tag is not None]

    #: Where the Next Round button sits, as a fraction of the game window.
    #: Only used when the UI model fails to detect it, which it does in some
    #: shop states -- measured on two otherwise similar frames with the cursor
    #: clear of any item, the button was found at 0.99 confidence with $7 in
    #: hand and not at all at any threshold with $1, when Reroll is greyed out.
    #: A hardcoded position is a poor substitute for detection; the real fix is
    #: labelled shop frames covering that state.
    NEXT_ROUND_FALLBACK = (0.356, 0.448)

    def next_round_fallback_position(self) -> Optional[Tuple[int, int]]:
        """Screen position of the Next Round button from the shop's layout."""
        region = self.screen_capture.get_capture_region()
        if not region:
            return None

        x_fraction, y_fraction = self.NEXT_ROUND_FALLBACK
        return (
            int(region['left'] + region['width'] * x_fraction),
            int(region['top'] + region['height'] * y_fraction),
        )

    def looks_like_shop(self, frame: np.ndarray) -> bool:
        """Whether this frame is a shop, judged by something being for sale."""
        return bool(self.shop_items(frame))

    def read_cash(self, frame: np.ndarray) -> Optional[int]:
        """Current money, or None when it cannot be read.

        Used to confirm a purchase actually happened: the shop redraws either
        way, but money only leaves when something is bought.
        """
        if self.ui_text_service is None:
            return None

        cash = [
            d
            for d in self.multi_detector.detect_ui(frame)
            if d.class_name.lower() == 'ui_data_cash'
        ]
        if not cash:
            return None

        try:
            for extraction in self.ui_text_service.extract(frame, cash):
                if extraction.text.isdigit():
                    return int(extraction.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'Could not read cash: {exc}')
        return None

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def execute_buy(self, index: int, description: str = '') -> Dict[str, Any]:
        """Buy the shop item at the given index.

        Args:
            index: Position in the shop, left to right, zero based
            description: Why the caller wants it, for the log

        Returns:
            Result dict with success, action_executed and error_message
        """
        result: Dict[str, Any] = {
            'success': False,
            'action_executed': False,
            'index': index,
            'description': description,
            'error_message': '',
        }

        if not self.mouse_controller.is_game_foreground():
            result['error_message'] = (
                'Refusing to buy: another window is in front of the game'
            )
            logger.error(result['error_message'])
            return result

        frame = self.screen_capture.capture_once()
        if frame is None:
            result['error_message'] = 'Screen capture failed'
            return result

        items = self.shop_items(frame)
        if not items:
            result['error_message'] = 'No items for sale were detected'
            logger.error(result['error_message'])
            return result

        if not 0 <= index < len(items):
            result['error_message'] = (
                f'Invalid shop index {index}: {len(items)} items for sale'
            )
            logger.error(result['error_message'])
            return result

        item, _tag = items[index]

        # Opening a pack leads to a selection screen that can only be skipped,
        # so buying one spends money for nothing. The prompt asks the model to
        # avoid them, and it bought one anyway when the description came back
        # unreadable and it guessed at what the item was.
        if entities.is_pack(item):
            result['error_message'] = (
                f'Refusing to buy item {index}: booster packs open a selection '
                f'screen that is not supported, so the purchase would be wasted'
            )
            logger.error(result['error_message'])
            return result

        cash_before = self.read_cash(frame)
        logger.info(
            f'Buying shop item {index} ({item.class_name}) - {description}; '
            f'cash before: {cash_before}'
        )

        if not self._click(item):
            result['error_message'] = f'Could not click shop item {index}'
            logger.error(result['error_message'])
            return result

        self._wait_until_settled()

        frame = self.screen_capture.capture_once()
        button = self._purchase_button(frame) if frame is not None else None

        if button is None:
            # One retry: a click into a window that was not active is sometimes
            # swallowed, exactly as it is on the card table.
            logger.warning('No Buy button after selecting; clicking the item again')
            self._click(item)
            self._wait_until_settled()
            frame = self.screen_capture.capture_once()
            button = self._purchase_button(frame) if frame is not None else None

        if button is None:
            result['error_message'] = (
                f'Refusing to buy: selecting item {index} produced no Buy button'
            )
            logger.error(result['error_message'])
            return result

        if not self._button_belongs_to(button, item):
            result['error_message'] = (
                f'Refusing to buy: the Buy button at x={button.center[0]} does '
                f'not line up with item {index} spanning '
                f'{item.bbox[0]}-{item.bbox[2]}, so a different item is selected'
            )
            logger.error(result['error_message'])
            return result

        if not self.mouse_controller.click_at(*self._to_screen(button.center, frame)):
            result['error_message'] = 'Could not click the Buy button'
            logger.error(result['error_message'])
            return result

        self._wait_until_settled()

        after = self.screen_capture.capture_once()
        cash_after = self.read_cash(after) if after is not None else None

        # The shop redraws whether or not anything was bought, so the money is
        # the signal that it happened.
        if cash_before is not None and cash_after is not None:
            if cash_after >= cash_before:
                result['error_message'] = (
                    f'Buy did not take effect: cash is still {cash_after}'
                )
                logger.error(result['error_message'])
                return result
            logger.info(f'Bought item {index}; cash {cash_before} -> {cash_after}')
        else:
            logger.warning('Could not read cash; purchase not confirmed')

        result['success'] = True
        result['action_executed'] = True
        result['cash_before'] = cash_before
        result['cash_after'] = cash_after
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _purchase_button(self, frame: np.ndarray) -> Optional[Detection]:
        """The Buy button, if the shop is showing one."""
        found = self.button_detector.find_best_button(frame, 'purchase')
        return found

    def _button_belongs_to(self, button: Any, item: Detection) -> bool:
        """Whether the Buy button sits under the item it is meant to buy.

        Balatro draws it beneath the selected item, so its horizontal position
        identifies what is selected. Without this check a stale selection from
        an earlier click would be bought instead.
        """
        centre_x = button.center[0]
        return (
            item.bbox[0] - self.BUTTON_ALIGNMENT_SLACK
            <= centre_x
            <= item.bbox[2] + self.BUTTON_ALIGNMENT_SLACK
        )

    def _to_screen(self, centre: Tuple[int, int], frame: np.ndarray) -> Tuple[int, int]:
        """Convert a frame coordinate to a screen coordinate."""
        region = self.screen_capture.get_capture_region()
        if not region:
            return int(centre[0]), int(centre[1])

        height, width = frame.shape[:2]
        return (
            int(region['left'] + centre[0] * region['width'] / width),
            int(region['top'] + centre[1] * region['height'] / height),
        )

    def _click(self, item: Detection) -> bool:
        frame = self.screen_capture.capture_once()
        if frame is None:
            return False
        return self.mouse_controller.click_at(*self._to_screen(item.center, frame))

    def _wait_until_settled(self, interval: float = 0.25) -> bool:
        """Block until the shop stops changing, so the next read is of the
        finished screen rather than a redraw in progress."""
        deadline = time.time() + self.SETTLE_TIMEOUT
        previous = None

        while time.time() < deadline:
            frame = self.screen_capture.capture_once()
            if frame is not None:
                signature = (
                    len(self.shop_items(frame)),
                    frozenset(
                        d.class_name.lower()
                        for d in self.multi_detector.detect_ui(frame)
                        if d.class_name.lower().startswith('button_')
                    ),
                )
                if signature == previous:
                    return True
                previous = signature
            time.sleep(interval)

        logger.warning('Shop did not settle; the next read may be mid-redraw')
        return False
