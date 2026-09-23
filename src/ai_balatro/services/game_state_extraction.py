"""Fast game state extraction pipeline with batched capture and OCR."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import entities
from ..core.detection import Detection
from ..core.multi_yolo_detector import MultiYOLODetector
from ..core.screen_capture import ScreenCapture
from ..utils.logger import get_logger
from .ui_text_service import UITextExtractionService
from .card_tooltip_service import CardTooltipService
from ..ai.actions.mouse_controller import MouseController

logger = get_logger(__name__)


@dataclass
class CardHoverFrame:
    """Stores data collected during a hover sweep."""

    card_index: int
    frame: np.ndarray


class GameStateExtractionService:
    """Service responsible for capturing and enriching Balatro game state snapshots."""

    def __init__(
        self,
        screen_capture: ScreenCapture,
        multi_detector: MultiYOLODetector,
        mouse_controller: Optional[MouseController] = None,
        card_tooltip_service: Optional[CardTooltipService] = None,
        ui_text_service: Optional[UITextExtractionService] = None,
        hover_dwell_time: float = 0.25,
        sweep_move_duration: float = 0.2,
        max_workers: int = 4,
    ) -> None:
        self.screen_capture = screen_capture
        self.multi_detector = multi_detector
        self.mouse_controller = mouse_controller
        self.card_tooltip_service = card_tooltip_service
        self.ui_text_service = ui_text_service or UITextExtractionService()
        self.hover_dwell_time = hover_dwell_time
        self.sweep_move_duration = sweep_move_duration
        self._detector_lock = threading.Lock()

    def capture_state(
        self,
        *,
        frame: Optional[np.ndarray] = None,
        entities_detection: Optional[Sequence[Detection]] = None,
        ui_detection: Optional[Sequence[Detection]] = None,
        capture_card_descriptions: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Capture a full game state snapshot with optional card description OCR."""
        if frame is None and self.mouse_controller is not None:
            # A cursor left on a card or shop item raises a tooltip that covers
            # whatever is behind it, and that tooltip is captured as part of the
            # board. Park before looking.
            self.mouse_controller.park_cursor()
            time.sleep(0.25)

        frame = frame if frame is not None else self.screen_capture.capture_once()
        if frame is None:
            logger.error('Game state capture failed: no frame available')
            return None

        entities = (
            list(entities_detection)
            if entities_detection is not None
            else self.multi_detector.detect_entities(frame)
        )
        ui_elements = (
            list(ui_detection)
            if ui_detection is not None
            else self.multi_detector.detect_ui(frame)
        )

        game_state, hand_cards = self._build_base_state(frame, entities, ui_elements)

        self._enrich_ui_text(frame, ui_elements, game_state)

        if capture_card_descriptions and hand_cards:
            self._enrich_card_descriptions(
                frame, entities, ui_elements, hand_cards, game_state
            )

        if capture_card_descriptions:
            self._enrich_owned_jokers(frame, game_state)
            if game_state['game_phase'] == 'shop':
                self._enrich_shop(frame, ui_elements, game_state)

        return game_state

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _build_base_state(
        self,
        frame: np.ndarray,
        entities_detection: Sequence[Detection],
        ui_detection: Sequence[Detection],
    ) -> Tuple[Dict[str, Any], List[Detection]]:
        """Create baseline game state payload from detections."""
        timestamp = time.time()
        buttons: List[Dict[str, Any]] = []

        # Hand cards define the index space the LLM addresses cards by, so they
        # come from the shared taxonomy rather than a local filter. Jokers and
        # the deck pile are not part of the hand and must not shift its indices.
        hand_cards = entities.hand_cards(entities_detection)
        cards = [
            {
                'index': idx,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'position': list(detection.bbox),
                'center': detection.center,
                'width': detection.width,
                'height': detection.height,
                'description_text': '',
                'description_detected': False,
                'parsed_description': None,
                'ocr_confidence': 0.0,
            }
            for idx, detection in enumerate(hand_cards)
        ]

        # A price tag above an item is what marks it as for sale. Owned jokers
        # and shop jokers share a class, and both are on screen at once in the
        # shop, so position alone cannot separate them.
        priced = entities.pair_with_prices(
            entities.shop_item_candidates(entities_detection),
            entities.price_tags(ui_detection),
        )
        shop_stock = [(item, tag) for item, tag in priced if tag is not None]
        owned_jokers = [
            item for item, tag in priced if tag is None and entities.is_joker(item)
        ]

        # Consumables you already hold. Balatro refuses to sell another
        # when both slots are full, and shows a Buy button that silently
        # does nothing -- so without this the agent retries the same
        # purchase forever, seeing only that its money never moved.
        owned_consumables = [
            item for item, tag in priced if tag is None and entities.is_consumable(item)
        ]

        jokers = [
            {
                'index': idx,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'position': list(detection.bbox),
                'center': detection.center,
                'width': detection.width,
                'height': detection.height,
                'description_text': '',
                'description_detected': False,
            }
            for idx, detection in enumerate(owned_jokers)
        ]

        for detection in ui_detection:
            if 'button' in detection.class_name.lower():
                x1, y1, x2, y2 = detection.bbox
                buttons.append(
                    {
                        'class_name': detection.class_name,
                        'confidence': detection.confidence,
                        'position': [x1, y1, x2, y2],
                        'center': detection.center,
                        'width': detection.width,
                        'height': detection.height,
                    }
                )

        game_phase = self._infer_game_phase(buttons, hand_cards, shop_stock)

        game_state = {
            'timestamp': timestamp,
            'entities_raw': list(entities_detection),
            'ui_elements_raw': list(ui_detection),
            'cards': cards,
            'jokers': jokers,
            'ui_buttons': buttons,
            'game_phase': game_phase,
            'card_descriptions': [],
            'ui_text_elements': [],
            'ui_text_values': {},
            'shop_items': [],
            'consumables': [
                {
                    'index': position,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'position': list(detection.bbox),
                }
                for position, detection in enumerate(owned_consumables)
            ],
            '_owned_joker_detections': owned_jokers,
            '_shop_stock_detections': shop_stock,
        }

        return game_state, hand_cards

    def _infer_game_phase(
        self,
        buttons: Sequence[Dict[str, Any]],
        hand_cards: Sequence[Detection] = (),
        shop_stock: Sequence[Any] = (),
    ) -> str:
        """Name the screen the game is showing.

        Cards in hand decide the playing phase rather than the Play button:
        Balatro only renders Play and Discard once cards are selected, so a
        button-only test reads a freshly dealt hand as 'unknown'. Keyword
        matching had the mirror problem, since 'button_sort_hand_rank'
        contains 'hand'.

        The other screens are identified by a button that appears only there,
        matched on the exact class rather than a substring.
        """
        classes = {btn['class_name'].lower() for btn in buttons}

        # Checked before the hand, because an opened booster pack shows cards
        # to choose from that look exactly like a hand. Its Skip button is the
        # only thing that distinguishes the two.
        if 'button_card_pack_skip' in classes:
            return 'pack_opening'

        # Something priced is for sale, so this is a shop. More reliable than
        # its buttons: the UI model misses Next Round and Reroll in some
        # states, and then nothing identifies the screen at all.
        if shop_stock:
            return 'shop'

        if hand_cards:
            return 'playing'

        if 'button_cash_out' in classes:
            return 'blind_won'
        if 'button_level_select' in classes:
            return 'blind_select'
        if classes & {
            'button_store_next_round',
            'button_store_reroll',
            'button_purchase',
        }:
            return 'shop'
        return 'unknown'

    def _enrich_ui_text(
        self,
        frame: np.ndarray,
        ui_detection: Sequence[Detection],
        game_state: Dict[str, Any],
    ) -> None:
        if not ui_detection:
            return

        try:
            ui_text_results = self.ui_text_service.extract(frame, ui_detection)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'UI text extraction failed: {exc}')
            return

        if not ui_text_results:
            return

        text_entries: List[Dict[str, Any]] = []
        value_map: Dict[str, Dict[str, Any]] = {}

        for result in ui_text_results:
            entry = {
                'class_name': result.class_name,
                'text': result.text,
                'position': list(result.bbox),
                'detection_confidence': result.detection_confidence,
                'ocr_success': result.ocr_success,
                'ocr_confidence': result.ocr_confidence,
            }
            text_entries.append(entry)

            existing = value_map.get(result.class_name)
            if (
                existing is None
                or entry['ocr_confidence'] > existing.get('ocr_confidence', 0.0)
                or (
                    entry['ocr_confidence'] == existing.get('ocr_confidence', 0.0)
                    and entry['detection_confidence']
                    > existing.get('detection_confidence', 0.0)
                )
            ):
                value_map[result.class_name] = {
                    'text': entry['text'],
                    'ocr_confidence': entry['ocr_confidence'],
                    'detection_confidence': entry['detection_confidence'],
                    'position': entry['position'],
                }

        game_state['ui_text_elements'] = text_entries
        game_state['ui_text_values'] = value_map

    def _enrich_owned_jokers(
        self, base_frame: np.ndarray, game_state: Dict[str, Any]
    ) -> None:
        """Read the name and effect of each joker the player owns.

        Without this a joker reaches the model as 'joker_card (confidence
        0.95)', which says nothing about what it does, so the model cannot
        reason about synergy with cards it cannot identify.
        """
        owned = game_state.get('_owned_joker_detections') or []
        if not owned:
            return

        descriptions = self.sweep_descriptions(base_frame, owned)
        for idx, desc in enumerate(descriptions):
            if idx >= len(game_state['jokers']):
                continue
            entry = game_state['jokers'][idx]
            entry['description_text'] = desc.get('description_text', '')
            entry['description_detected'] = desc.get('description_detected', False)

    def _enrich_shop(
        self,
        base_frame: np.ndarray,
        ui_detection: Sequence[Detection],
        game_state: Dict[str, Any],
    ) -> None:
        """Read what the shop is selling, for how much, and what each one does.

        The tooltip is the only place the game states a joker's name, effect,
        rarity and stickers, and stickers are run-specific: an external table
        cannot know this copy is Perishable with five rounds left or Rental at
        $3 a round, which on higher stakes decides whether it is worth buying.
        """
        stock = game_state.get('_shop_stock_detections') or []
        if not stock:
            return

        prices: Dict[tuple, str] = {}
        tags = [tag for _, tag in stock]
        if tags and self.ui_text_service is not None:
            try:
                for extraction in self.ui_text_service.extract(base_frame, tags):
                    prices[tuple(extraction.bbox)] = extraction.text
            except Exception as exc:  # noqa: BLE001
                logger.warning(f'Shop price OCR failed: {exc}')

        items = [item for item, _ in stock]
        descriptions = self.sweep_descriptions(base_frame, items)

        entries: List[Dict[str, Any]] = []
        for idx, (item, tag) in enumerate(stock):
            desc = descriptions[idx] if idx < len(descriptions) else {}
            entries.append(
                {
                    'index': idx,
                    'class_name': item.class_name,
                    'confidence': item.confidence,
                    'position': list(item.bbox),
                    'center': item.center,
                    'price': prices.get(tuple(tag.bbox), ''),
                    'description_text': desc.get('description_text', ''),
                    'description_detected': desc.get('description_detected', False),
                }
            )

        game_state['shop_items'] = entries

    def sweep_descriptions(
        self,
        base_frame: np.ndarray,
        targets: Sequence[Detection],
    ) -> List[Dict[str, Any]]:
        """Hover each target in turn and OCR the tooltip it raises.

        Works for anything the game describes on hover -- hand cards, the
        jokers you own, shop stock -- because the tooltip is the only place the
        game states a joker's name, effect and stickers. Reading it beats any
        external table, which cannot know that this particular copy is
        Perishable with five rounds left or Rental at $3 a round.

        Args:
            base_frame: A settled frame used to plan hover positions
            targets: Detections to hover, in the order results are wanted

        Returns:
            One description dict per target, in the same order, where entries
            for targets that raised no readable tooltip are present but empty.
            An empty list means the sweep could not run at all, which is a
            different thing from running and reading nothing.
        """
        empty = {
            'description_text': '',
            'description_detected': False,
            'ocr_confidence': 0.0,
            'parsed_description': None,
        }

        ordered = list(targets)
        if not ordered:
            return []
        if not self.card_tooltip_service or not self.mouse_controller:
            logger.debug('Tooltip service or mouse controller missing; skipping sweep')
            return []

        sweep_plan = self._plan_card_sweep(base_frame, ordered)
        if not sweep_plan:
            logger.debug('Unable to plan sweep; skipping hover')
            return []

        hover_frames: Dict[int, np.ndarray] = {}
        index_lookup = [index for index, _ in sweep_plan]
        positions = [coords for _, coords in sweep_plan]

        def capture_callback(step_index: int, _x: int, _y: int) -> None:
            index = index_lookup[min(step_index, len(index_lookup) - 1)]
            frame = self.screen_capture.capture_once()
            if frame is not None:
                hover_frames[index] = frame

        swept = self.mouse_controller.sweep_path(
            positions,
            dwell_time=self.hover_dwell_time,
            move_duration=self.sweep_move_duration,
            capture_callback=capture_callback,
        )

        if not swept or not hover_frames:
            logger.debug('Hover sweep produced no frames; skipping description OCR')
            return []

        descriptions = [dict(empty) for _ in ordered]
        for index, frame in hover_frames.items():
            if index < len(ordered):
                descriptions[index] = self._process_hover_frame(
                    ordered[index], frame, index
                )
        return descriptions

    def _enrich_card_descriptions(
        self,
        base_frame: np.ndarray,
        entities_detection: Sequence[Detection],
        ui_detection: Sequence[Detection],
        hand_cards: Sequence[Detection],
        game_state: Dict[str, Any],
    ) -> None:
        descriptions = self.sweep_descriptions(base_frame, hand_cards)
        if not descriptions:
            return

        game_state['card_descriptions'] = descriptions

        for idx, desc in enumerate(descriptions):
            if idx >= len(game_state['cards']):
                continue

            card_entry = game_state['cards'][idx]
            card_entry['description_text'] = desc.get('description_text', '')
            card_entry['description_detected'] = desc.get('description_detected', False)
            card_entry['parsed_description'] = desc.get('parsed_description')
            card_entry['ocr_confidence'] = desc.get('ocr_confidence', 0.0)

    def _plan_card_sweep(
        self, frame: np.ndarray, hand_cards: Sequence[Detection]
    ) -> List[Tuple[int, Tuple[int, int]]]:
        if not self.card_tooltip_service:
            return []

        plan: List[Tuple[int, Tuple[int, int]]] = []
        for idx, detection in enumerate(hand_cards):
            try:
                info = self.card_tooltip_service._prepare_card_info(detection, idx)
                hover_x, hover_y = info['hover_position']
                screen_x, screen_y = (
                    self.card_tooltip_service._frame_to_screen_coordinates(
                        hover_x, hover_y, frame.shape
                    )
                )
                plan.append((idx, (screen_x, screen_y)))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f'Failed to prepare sweep position for card {idx}: {exc}')

        plan.sort(key=lambda item: item[1][0])
        return plan

    def _process_hover_frame(
        self, card: Detection, frame: np.ndarray, position_index: int
    ) -> Dict[str, Any]:
        if not self.card_tooltip_service:
            return {
                'description_text': '',
                'description_detected': False,
                'ocr_confidence': 0.0,
                'parsed_description': None,
            }

        with self._detector_lock:
            try:
                return self.card_tooltip_service._process_hovered_card_frame(
                    frame, card, position_index, save_debug_image=False
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f'Failed to process hovered frame for card {position_index}: {exc}'
                )
                return {
                    'description_text': '',
                    'description_detected': False,
                    'ocr_confidence': 0.0,
                    'parsed_description': None,
                }

    def _fallback_missing_cards(
        self,
        frame: np.ndarray,
        hand_cards: Sequence[Detection],
        detections: Sequence[Detection],
        card_descriptions: List[Dict[str, Any]],
    ) -> None:
        if not self.card_tooltip_service:
            return

        missing_indices = [
            idx
            for idx, desc in enumerate(card_descriptions)
            if not desc.get('description_detected')
        ]

        if not missing_indices:
            return

        try:
            fallback_infos = self.card_tooltip_service.collect_card_infos(
                frame,
                list(hand_cards),
                detections=list(detections),
                auto_hover_missing=True,
                save_debug_images=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'Fallback card description collection failed: {exc}')
            return

        for idx in missing_indices:
            if idx < len(fallback_infos):
                card_descriptions[idx] = fallback_infos[idx]
