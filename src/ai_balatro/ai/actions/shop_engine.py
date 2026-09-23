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

    def _settled_frame(self) -> Optional[np.ndarray]:
        """A frame with the cursor moved off anything that raises a tooltip.

        The engine takes its own captures, so it needs the same parking that
        capture_state does: a cursor left on a card hides whatever sits behind
        its tooltip, and a pack read that way reported one item when two were
        on offer.
        """
        self.mouse_controller.park_cursor()
        time.sleep(0.3)
        return self.screen_capture.capture_once()

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

        frame = self._settled_frame()
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
                hint = ''
                if entities.is_consumable(item):
                    hint = (
                        ' -- a tarot, planet or spectral card cannot be bought '
                        'while both consumable slots are full, and the game '
                        'shows a Buy button regardless'
                    )
                result['error_message'] = (
                    f'Buy did not take effect: cash is still {cash_after}{hint}'
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

    #: Anything above this fraction of the window is the row of jokers and
    #: consumables you already own, which is on screen while a pack is open.
    #: Measured: owned row at 0.11-0.12, pack contents at 0.58-0.62.
    OWNED_ROW_MAX_FRACTION = 0.35

    #: The Select button is drawn just under the chosen card. The UI model has
    #: no class for it, so it is clicked relative to the card instead -- which
    #: is anchored to a real detection rather than a fixed window position.
    #:
    #: Measured as a fraction of the CARD's height, not the window's: on a
    #: Buffoon pack the selected joker was 99px tall with Select centred 9px
    #: below it, and on a Standard Pack a 118px card had it 11px below. A
    #: window-relative offset put the click just under the button and the
    #: choice silently did not register.
    SELECT_OFFSET_FRACTION = 0.09

    def pack_contents(self, frame: np.ndarray) -> List[Detection]:
        """The cards or items an opened pack is offering, left to right.

        Excludes the row you already own, which stays on screen, and the deck.
        A Buffoon pack offers jokers that share a class with the ones you hold,
        so they can only be told apart by where they sit.
        """
        height = frame.shape[0]
        cutoff = height * self.OWNED_ROW_MAX_FRACTION

        offered = [
            d
            for d in self.multi_detector.detect_entities(frame)
            if d.bbox[1] > cutoff
            and not entities.is_pile(d)
            and not entities.is_description(d)
        ]
        return entities.sort_left_to_right(offered)

    def _is_selected(self, item: Detection, offered: List[Detection]) -> bool:
        """Whether this item is the one lifted above the rest.

        A pack lifts the chosen item, exactly as the table lifts a selected
        card. Measured on a Buffoon pack: the chosen joker sat at y=363 while
        the other sat at y=379.
        """
        if len(offered) < 2:
            return False
        others = [o for o in offered if o is not item]
        return all(item.bbox[1] < other.bbox[1] - 8 for other in others)

    def choose_from_pack(self, index: int, description: str = '') -> Dict[str, Any]:
        """Take one item from an opened booster pack.

        Args:
            index: Which offered item to take, left to right, zero based
            description: Why, for the log
        """
        result: Dict[str, Any] = {
            'success': False,
            'action_executed': False,
            'index': index,
            'error_message': '',
        }

        if not self.mouse_controller.is_game_foreground():
            result['error_message'] = (
                'Refusing to choose: another window is in front of the game'
            )
            logger.error(result['error_message'])
            return result

        frame = self._settled_frame()
        if frame is None:
            result['error_message'] = 'Screen capture failed'
            return result

        offered = self.pack_contents(frame)
        if not offered:
            result['error_message'] = 'No pack contents were detected'
            logger.error(result['error_message'])
            return result

        if not 0 <= index < len(offered):
            result['error_message'] = (
                f'Invalid pack index {index}: the pack offers {len(offered)} items'
            )
            logger.error(result['error_message'])
            return result

        item = offered[index]
        logger.info(f'Taking pack item {index} ({item.class_name}) - {description}')

        # Selecting lifts the item above the others, so an already-selected one
        # must not be clicked again: that deselects it and the Select button
        # disappears before it can be pressed.
        if not self._is_selected(item, offered):
            if not self.mouse_controller.click_at(*self._to_screen(item.center, frame)):
                result['error_message'] = f'Could not click pack item {index}'
                return result
            time.sleep(0.6)

        # Re-read: the chosen item has moved, and Select is drawn under where
        # it sits now, not where it sat before the click.
        frame = self._settled_frame()
        if frame is None:
            result['error_message'] = 'Screen capture failed after selecting'
            return result

        current = self.pack_contents(frame)
        match = [c for c in current if abs(c.bbox[0] - item.bbox[0]) < 40]
        if not match:
            result['error_message'] = f'Pack item {index} vanished after clicking'
            logger.error(result['error_message'])
            return result

        chosen = match[0]
        if not self._is_selected(chosen, current):
            result['error_message'] = (
                f'Refusing to confirm: pack item {index} did not stay selected'
            )
            logger.error(result['error_message'])
            return result

        card_height = chosen.bbox[3] - chosen.bbox[1]
        select_y = chosen.bbox[3] + card_height * self.SELECT_OFFSET_FRACTION
        centre_x = (chosen.bbox[0] + chosen.bbox[2]) // 2
        if not self.mouse_controller.click_at(
            *self._to_screen((centre_x, int(select_y)), frame)
        ):
            result['error_message'] = 'Could not click the Select button'
            return result

        self._wait_until_settled()

        # The pack closing is the evidence it was taken; the screen redraws
        # either way, so nothing else proves it.
        after = self._settled_frame()
        if after is not None and self.pack_is_open(after):
            result['error_message'] = (
                'Choice did not take effect: the pack is still open'
            )
            logger.error(result['error_message'])
            return result

        logger.info(f'Took pack item {index}')
        result['success'] = True
        result['action_executed'] = True
        return result

    def pack_is_open(self, frame: np.ndarray) -> bool:
        """Whether a booster pack is showing its contents."""
        return any(
            d.class_name.lower() == 'button_card_pack_skip'
            for d in self.multi_detector.detect_ui(frame)
        )

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
