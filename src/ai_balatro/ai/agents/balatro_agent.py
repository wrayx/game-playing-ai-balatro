"""Balatro game playing agent with LLM reasoning and strategic planning."""

import time
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from ..providers.base import LLMProvider
from ..actions.executor import ActionExecutor
from ..actions.schemas import GAME_ACTIONS
from ...core.multi_yolo_detector import MultiYOLODetector
from ...core.screen_capture import ScreenCapture
from ...core.detection import Detection
from ...services.ui_text_service import UITextExtractionService
from ...services.game_state_extraction import GameStateExtractionService
from ...utils.logger import get_logger

logger = get_logger(__name__)

#: Appended to the prompt only while cards are on the table.
PLAYING_DECISION_SECTION = """Based on the card descriptions and game state, make the optimal strategic decision:
{poker_objectives}

ACTION INSTRUCTIONS:
1. Analyze the cards listed above (Card 0, Card 1, Card 2, etc.) with their descriptions
2. Identify the best poker hand you can form from these cards
3. Discarding cards may help you draw better cards to improve your hand, forming stronger poker hands
4. Choose ONE action:

   a) If you have a strong playable hand:
      - Use play_cards(indices=[...]) with the indices of cards to play
      - Example: play_cards(indices=[0, 1, 2, 3, 4]) to play first 5 cards
      - Example: play_cards(indices=[0, 2, 4, 6, 7]) to play specific cards
      - If you are holding 8, 8, 9, 2, 5, 10, J, Q, you should play 8, 9, 10, J, Q to form a straight, and the indices would be [0, 2, 5, 6, 7]

   b) If you need better cards:
      - Use discard_cards(indices=[...]) with the indices of cards to discard
      - Example: discard_cards(indices=[5, 6, 7]) to discard last 3 cards
      - Example: discard_cards(indices=[1, 3]) to discard specific unwanted cards
      - If you are holding 8, 10, 2, 5, 5, J, Q, K, since 8, 10, J, Q, K forms with one addition 9 a better straight, try discarding 2 and 5, the indices would be [2, 3]

   c) If you need to interact with UI:
      - Use click_button(button_type='...') for UI actions

Suits can be counted too, if you hold 3, 7, 8, J, 9, 2, 2, 4, suggesting high card but if you could identify the suits of the card, say
3 of Hearts, 7 of Hearts, 8 of Hearts, J of Hearts, 9 of Hearts, 2 of Diamonds, 2 of Clubs, 4 of Spades, this would form a flush with Hearts suit, so playing these 5 cards would be a better option.

Remember:
- Cards are indexed from 0 (Card 0 is the leftmost)
- You can only play OR discard in one action, not both
- Provide clear reasoning for your choice

Execute the best action immediately and explain your strategic reasoning."""


class BalatroReasoningAgent(BaseAgent):
    """
    Balatro-specific reasoning agent that combines vision processing,
    LLM strategic planning, and action execution.
    """

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        screen_capture: ScreenCapture,
        multi_detector: Optional[MultiYOLODetector] = None,
        **kwargs,
    ):
        """
        Initialize Balatro reasoning agent.

        Args:
            name: Agent identifier
            llm_provider: LLM provider for reasoning
            screen_capture: Screen capture for game state observation
            multi_detector: Multi-YOLO detector for entity detection
        """
        super().__init__(name, llm_provider, **kwargs)

        self.screen_capture = screen_capture
        self.multi_detector = multi_detector or MultiYOLODetector()

        # Initialize action executor
        self.action_executor = ActionExecutor(
            screen_capture=screen_capture, multi_detector=self.multi_detector
        )
        self.action_executor.initialize()

        self.ui_text_service = UITextExtractionService()
        self.game_state_extractor = GameStateExtractionService(
            screen_capture=self.screen_capture,
            multi_detector=self.multi_detector,
            mouse_controller=self.action_executor.card_engine.mouse_controller,
            card_tooltip_service=self.action_executor.card_engine.card_tooltip_service,
            ui_text_service=self.ui_text_service,
        )

        # Game state tracking
        self.current_game_state: Dict[str, Any] = {}
        self.game_history: List[Dict[str, Any]] = []

        logger.info(f"BalatroReasoningAgent '{name}' initialized")

    def run(self, context: AgentContext) -> AgentResult:
        """Run a complete reasoning cycle with integrated decision making."""
        logger.info(f'Running {self.name} - integrated decision cycle')

        try:
            result = self.analyze_situation(context)

            if result.success and result.action:
                execution_result = self.execute_action(result.action, result.context)
                return execution_result
            elif result.success:
                return result
            else:
                return result

        except Exception as e:
            logger.error(f'Integrated run cycle failed: {e}')
            return AgentResult(
                success=False, errors=[f'Run cycle failed: {e}'], state=AgentState.ERROR
            )

    def _get_system_message(self) -> str:
        """Get Balatro-specific system message."""
        return """You are a Balatro expert AI agent. You analyze game states and make immediate strategic decisions to maximize scores.

Your capabilities:
- Analyze detected cards with full OCR text descriptions (card names, ranks, suits)
- Understand poker hands and scoring mechanics
- Make strategic card plays and discards based on card indices
- Execute actions through simple function calls

Available actions:
- play_cards(indices=[0,1,2]): Play selected cards by their index numbers (0-based)
  Example: play_cards(indices=[0,1,2]) plays the first three cards
- discard_cards(indices=[3,4]): Discard selected cards by their index numbers (0-based)
  Example: discard_cards(indices=[3,4]) discards cards at positions 3 and 4
- click_button(button_type='...'): Press UI buttons (play, discard, skip, shop, next)

Card indexing:
- Cards are numbered starting from 0 (leftmost card is index 0)
- Each card in your hand has a unique index
- Use the provided card information (Card 0, Card 1, etc.) to select cards

Make immediate, optimal decisions based on the complete card information provided."""

    #: How many times to look again when the screen is not recognisable.
    UNKNOWN_PHASE_RETRIES = 4

    #: How long to wait between those looks.
    UNKNOWN_PHASE_DELAY = 1.5

    def _capture_recognisable_state(self) -> Optional[Dict[str, Any]]:
        """Capture the board, waiting out a screen that is mid-transition.

        Balatro animates in stages, so a capture can land between them and see
        no cards and no actionable buttons. Asking the model to decide from
        that produces a guess -- it picked 'next' on the Cash Out screen more
        than once -- and no amount of settling after an action prevents it,
        because the next screen can arrive later still.

        Waiting is free and a wrong button is not, so look again rather than
        reason about a screen that is not there yet.
        """
        game_state = None

        for attempt in range(self.UNKNOWN_PHASE_RETRIES + 1):
            game_state = self.game_state_extractor.capture_state(
                capture_card_descriptions=True
            )
            if game_state is None:
                return None
            if game_state.get('game_phase') != 'unknown':
                return game_state

            if attempt < self.UNKNOWN_PHASE_RETRIES:
                logger.info(
                    f'Screen not recognisable yet; looking again '
                    f'({attempt + 1}/{self.UNKNOWN_PHASE_RETRIES})'
                )
                time.sleep(self.UNKNOWN_PHASE_DELAY)

        logger.warning('Screen still not recognisable; reasoning about it anyway')
        return game_state

    def analyze_situation(self, context: AgentContext) -> AgentResult:
        """Analyze current Balatro game state and make immediate strategic decision."""
        try:
            logger.info('Analyzing Balatro game situation and planning action...')

            game_state = self._capture_recognisable_state()

            if game_state is None:
                return AgentResult(
                    success=False,
                    errors=['Failed to capture screen'],
                    state=AgentState.ERROR,
                )

            entities_detection = game_state.get('entities_raw', [])
            ui_detection = game_state.get('ui_elements_raw', [])

            # Update context with current state
            context.game_state.update(game_state)
            self.current_game_state = game_state

            # Create strategic decision prompt
            decision_prompt = self._create_analysis_prompt(game_state)

            # Query LLM with function calling for immediate action decision
            result = self._llm_query(
                decision_prompt, use_functions=True, functions=GAME_ACTIONS
            )

            if result.success:
                logger.info(f'Strategic decision: {result.reasoning}')
                return AgentResult(
                    success=True,
                    action=result.action,  # Action is planned immediately
                    reasoning=result.reasoning,
                    context=context,
                    metadata={
                        'game_state': game_state,
                        'entities_count': len(entities_detection),
                        'ui_elements_count': len(ui_detection),
                    },
                )
            else:
                return AgentResult(
                    success=False, errors=result.errors, state=AgentState.ERROR
                )

        except Exception as e:
            logger.error(f'Failed to analyze situation and plan action: {e}')
            return AgentResult(
                success=False,
                errors=[f'Analysis/planning failed: {e}'],
                state=AgentState.ERROR,
            )

    def plan_action(self, context: AgentContext) -> AgentResult:
        """Plan strategic action - now integrated into analyze_situation for efficiency."""
        logger.info(
            'Action planning is now integrated into analyze_situation for efficiency'
        )

        # Check if action was already planned during analysis
        if hasattr(context, 'planned_action') and context.planned_action:
            return AgentResult(
                success=True,
                action=context.planned_action,
                reasoning='Action already planned during analysis phase',
                context=context,
                metadata={'reused_planned_action': True},
            )

        # Fallback: re-run analysis if no action planned
        return self.analyze_situation(context)

    def execute_action(
        self, action: Dict[str, Any], context: AgentContext
    ) -> AgentResult:
        """Execute the planned action using the action executor."""
        try:
            logger.info(f'Executing action: {action}')

            # Execute action through action executor
            execution_result = self.action_executor.process({'function_call': action})

            # Record action in game history
            action_record = {
                'timestamp': time.time(),
                'action': action,
                'game_state_before': self.current_game_state.copy(),
                'success': execution_result.success,
                'errors': execution_result.errors
                if not execution_result.success
                else [],
            }
            self.game_history.append(action_record)

            if execution_result.success:
                logger.info('Action executed successfully')
                return AgentResult(
                    success=True,
                    action=action,
                    reasoning=f'Successfully executed {action.get("name", "unknown action")}',
                    context=context,
                    metadata={
                        'execution_data': execution_result.data,
                        'execution_time': time.time(),
                    },
                )
            else:
                logger.error(f'Action execution failed: {execution_result.errors}')
                return AgentResult(
                    success=False,
                    action=action,
                    errors=execution_result.errors,
                    state=AgentState.ERROR,
                )

        except Exception as e:
            logger.error(f'Failed to execute action: {e}')
            return AgentResult(
                success=False,
                action=action,
                errors=[f'Execution failed: {e}'],
                state=AgentState.ERROR,
            )

    def _extract_game_state(
        self,
        entities_detection: Optional[List[Detection]] = None,
        ui_detection: Optional[List[Detection]] = None,
        frame=None,
        capture_card_descriptions: bool = True,
    ) -> Dict[str, Any]:
        """Compatibility wrapper delegating to game state extractor."""
        snapshot = self.game_state_extractor.capture_state(
            frame=frame,
            entities_detection=entities_detection,
            ui_detection=ui_detection,
            capture_card_descriptions=capture_card_descriptions,
        )

        return snapshot or {
            'timestamp': time.time(),
            'entities_raw': entities_detection or [],
            'ui_elements_raw': ui_detection or [],
            'cards': [],
            'jokers': [],
            'ui_buttons': [],
            'game_phase': 'unknown',
            'card_descriptions': [],
            'ui_text_elements': [],
            'ui_text_values': {},
        }

    def _ui_value(self, game_state: Dict[str, Any], class_name: str) -> str:
        """One OCR'd UI readout by class name, or '' when it was not read."""
        for entry in game_state.get('ui_text_elements', []):
            if entry.get('class_name') == class_name:
                return str(entry.get('text', '')).strip()
        return ''

    def _create_shop_section(self, game_state: Dict[str, Any]) -> str:
        """Prompt for the shop screen.

        The shop is where a run is won or lost: an agent that never buys has no
        jokers and no scaling, and stalls around ante 2 no matter how well it
        plays the cards it is dealt.
        """
        items = game_state.get('shop_items', [])
        cash = self._ui_value(game_state, 'ui_data_cash') or 'unknown'

        lines = []
        for entry in items:
            text = ' '.join(str(entry.get('description_text', '')).split())
            if len(text) > 160:
                text = text[:160] + '...'
            price = entry.get('price') or '?'
            lines.append(
                f'  Item {entry["index"]}: {entry["class_name"]} costs ${price}'
                + (f' -> {text}' if text else ' -> (description unreadable)')
            )

        owned = []
        for joker in game_state.get('jokers', []):
            text = ' '.join(str(joker.get('description_text', '')).split())
            owned.append(f'  {text[:120]}' if text else '  (unreadable joker)')

        return f"""WHAT TO DO NOW:
You are in the shop. There are no cards to play here.

FOR SALE ({len(items)} items):
{chr(10).join(lines) if lines else '  Nothing detected for sale'}

You have ${cash}.

Jokers you already own ({len(owned)} of 5 slots):
{chr(10).join(owned) if owned else '  None'}

Buy something with buy_item(index=N), using the item numbers above.
Leave the shop with click_button(button_type='next').

HOW TO DECIDE:
- Jokers are where scoring comes from. A run holding no jokers stalls
  around ante 2, because blind targets roughly triple while an unimproved
  deck does not. Buying a decent joker early usually beats saving.
- Money earns interest between rounds: $1 for every $5 you hold, capped at
  $5 per round, so $25 is the point beyond which holding more earns
  nothing extra. Spending down to nothing costs future income, so leaving a
  few dollars is worth something, but not at the cost of an empty joker
  slot in the early antes.
- Read the sticker lines in each description. 'Perishable' means it stops
  working after a few rounds. 'Rental' means it drains $3 every round.
  Both are worth much less than the same joker without them.
- Booster packs open a selection screen this agent cannot yet handle, so
  do not buy them. Prefer jokers, then consumables.
- Leaving without buying is a legitimate choice when nothing is worth its
  price, or when everything on offer is a pack.

Choose one action now and explain your reasoning."""

    def _create_analysis_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create prompt for game state analysis."""
        cards_info = []

        for i, card in enumerate(game_state.get('cards', [])):
            base_line = (
                f'Card {i}: {card["class_name"]} (confidence: {card["confidence"]:.2f})'
            )

            # Always show the raw OCR alongside any parse. The parser drops the
            # suit whenever OCR garbles it, and a confident wrong rank used to
            # replace the raw text entirely -- hiding the evidence that would
            # have corrected it. '+11 chips' identifies an Ace even when the
            # rank word is unreadable.
            descriptors = []

            parsed = card.get('parsed_description') or {}
            if parsed.get('english_name'):
                descriptor = parsed['english_name']
                short_code = parsed.get('short_code')
                if short_code:
                    descriptor += f' [{short_code}]'
                descriptors.append(descriptor)

            desc_text = ' '.join((card.get('description_text') or '').split())
            if len(desc_text) > 100:
                desc_text = desc_text[:100] + '...'
            if desc_text:
                descriptors.append(f'raw: {desc_text}')

            if descriptors:
                cards_info.append(f'{base_line} -> {" | ".join(descriptors)}')
            else:
                cards_info.append(base_line)

        jokers_info = []
        for joker in game_state.get('jokers', []):
            # The effect text, not the class name. A joker's rules are read
            # off its tooltip every turn, so the model does not have to
            # remember them -- but only if they are actually shown here. Sent
            # as a bare class name, 'Ride the Bus' says nothing about avoiding
            # face cards, which is the whole reason to own it.
            text = ' '.join(str(joker.get('description_text', '')).split())
            if len(text) > 160:
                text = text[:160] + '...'
            jokers_info.append(
                f'Joker: {text}'
                if text
                else f'Joker: {joker["class_name"]} '
                f'(effect unreadable, confidence: {joker["confidence"]:.2f})'
            )

        buttons_info = []
        for button in game_state.get('ui_buttons', []):
            buttons_info.append(f'Button: {button["class_name"]}')

        ui_text_info = []
        for ui_item in game_state.get('ui_text_elements', []):
            value_text = ui_item.get('text', '')
            value_compact = ' '.join(value_text.split()) if value_text else '(no text)'
            confidence = ui_item.get('ocr_confidence')
            if confidence is not None and confidence > 0:
                ui_text_info.append(
                    f'{ui_item["class_name"]}: {value_compact} (OCR {confidence:.2f})'
                )
            else:
                ui_text_info.append(f'{ui_item["class_name"]}: {value_compact}')

        # Add OCR capture status
        ocr_status = ''
        if game_state.get('cards'):
            captured_count = sum(
                1 for card in game_state['cards'] if card.get('description_detected')
            )
            total_cards = len(game_state['cards'])
            ocr_status = (
                f'\nCARD DESCRIPTIONS CAPTURED: {captured_count}/{total_cards} cards'
            )

        screen_actions = {
            'blind_won': (
                'You have beaten this blind, so there are no cards to play.\n'
                "Collect the reward with click_button(button_type='cash_out')."
            ),
            'blind_select': (
                'You are choosing the next blind, so there are no cards to '
                'play.\n'
                "Start it with click_button(button_type='level_select'), or "
                "pass it up with click_button(button_type='skip') to take the "
                'tag instead.'
            ),
            'shop': (
                'You are in the shop, so there are no cards to play. Buying is '
                'not wired up yet, so leave with '
                "click_button(button_type='next')."
            ),
            'pack_opening': (
                "You opened a booster pack. The cards shown are the pack's "
                'contents, not your hand, so they cannot be played.\n'
                'Choosing from a pack is not supported yet, so move on with '
                "click_button(button_type='skip')."
            ),
            'unknown': (
                'This is not a screen you can play cards on, and it may still '
                'be animating.\nUse click_button with whichever button listed '
                'above advances the game. Do not try to play or discard cards.'
            ),
        }

        poker_objectives = (
            '\nPOKER OBJECTIVES:\n'
            '- Form the strongest five-card poker hand from the detected cards.\n'
            '- Favor high-ranking combinations (pairs, straights, flushes, full houses, etc.).\n'
            '- Discard low-value cards that do not contribute to potential strong hands.'
        )

        phase = game_state.get('game_phase', 'unknown')
        if phase == 'playing':
            decision_section = PLAYING_DECISION_SECTION.format(
                poker_objectives=poker_objectives
            )
        elif phase == 'shop':
            decision_section = self._create_shop_section(game_state)
        else:
            # Suppress the poker instructions entirely off the table. Leaving
            # them in is what led an agent to call play_cards on the Cash Out
            # screen, where there is no hand at all.
            decision_section = f'WHAT TO DO NOW:\n{screen_actions.get(phase, screen_actions["unknown"])}'

        return f"""You are playing a game called Balatro, a game borrowed the concept of Texas Hold'em Poker and enhanced the gameplay with rogue-like level setup, and many different joker cards to manipulate the game rules.
Most strategy comes from understanding the Texas Hold'em poker rules and making optimal plays based on the current hand and game phase.

<game_rules>
In Texas Hold'em, the poker hand rankings from highest to lowest are:

1. Royal Flush (e.g. A, K, Q, J, 10 of the same suit)
2. Straight Flush (e.g. 5, 6, 7, 8, 9 of the same suit)
3. Four of a Kind (e.g. four Aces, four Kings, etc.)
4. Full House (e.g. three of a kind plus a pair)
5. Flush (e.g. any five cards of the same suit)
6. Straight (e.g. five consecutive cards of mixed suits)
7. Three of a Kind (e.g. three Aces, three Kings, etc.)
8. Two Pair (e.g. two Aces and two Kings)
9. One Pair (e.g. two Aces)
10. High Card (e.g. the highest card in hand)

It's the same in Balatro, but instead of gaming with other opponents, you are playing against a score target set by the current blind.
The deck holds 52 playing cards by default; you may add or modify cards as the run progresses.
You hold a hand of 8 cards by default and may select at most 5 of them for any single action.
Playing cards scores them and then draws replacements back up to your hand size.
Discarding cards also draws replacements back up to your hand size - that is what discards are for, they let you dig for a better hand at no score cost.
Each blind gives you a limited number of hands and discards (4 and 3 by default). Spending your last hand without reaching the target ends the run.
Scoring is not simply the rank of your hand: the hand type contributes base chips and a multiplier that grow each time that hand type is levelled up, the individual scoring cards add their own chip values, and joker cards modify chips or multiplier further. The round score is chips multiplied by the multiplier.
Your goal is therefore to reach the target score before running out of hands - not to form the most impressive poker hand. A frequently levelled modest hand backed by jokers often outscores a rarer one.

Card descriptions are read off the screen with OCR and are often garbled ('Hce of Diamonds' is the Ace of Diamonds). Each description ends with that card's chip value, which survives a bad read and narrows the rank: 11 chips is an Ace, 10 chips is a Ten or a face card, and any other number is that card's rank. Use the chip value and the suit line to recover a card whose rank word is unreadable, rather than treating it as unknown.
</game_rules>

Here is your current known game state:

<game_state>
Current cards in hand ({len(game_state.get('cards', []))} cards):
{chr(10).join(cards_info) if cards_info else 'No cards detected'}{ocr_status}

Current Joker cards ({len(game_state.get('jokers', []))} active) enabled:
{chr(10).join(jokers_info) if jokers_info else 'No jokers detected'}

UI elements you can interact with ({len(game_state.get('ui_buttons', []))} buttons):
{chr(10).join(buttons_info) if buttons_info else 'No buttons detected'}

Dynamic UI values ({len(game_state.get('ui_text_elements', []))} tracked):
{chr(10).join(ui_text_info) if ui_text_info else 'No dynamic UI text detected'}

GAME PHASE: {game_state.get('game_phase', 'unknown')}
</game_state>

{decision_section}"""

    def _create_decision_prompt_legacy(self, game_state: Dict[str, Any]) -> str:
        """Legacy planning prompt - replaced by integrated decision making."""
        cards_count = len(game_state.get('cards', []))

        # This method is kept for backward compatibility but not used
        position_example = 'Example position arrays:\n'
        position_example += (
            '- Play first 3 cards: [1, 1, 1' + ', 0' * max(0, cards_count - 3) + ']\n'
        )
        position_example += (
            '- Discard last 2 cards: [0' * max(0, cards_count - 2) + ', -1, -1]\n'
        )
        position_example += (
            '- Play cards 0 and 2: [1, 0, 1' + ', 0' * max(0, cards_count - 3) + ']'
        )

        return f"""Based on the game state analysis, plan your next strategic action.

CURRENT SITUATION:
- Hand size: {cards_count} cards
- Game phase: {game_state.get('game_phase', 'unknown')}
- Available buttons: {[btn['class_name'] for btn in game_state.get('ui_buttons', [])]}

STRATEGIC OPTIONS:
1. **Play cards**: Use position array with 1s for cards to play
2. **Discard cards**: Use position array with -1s for cards to discard
3. **Hover card**: Examine specific card details first
4. **Click button**: Use available UI buttons

{position_example}

Choose the action that maximizes your score potential. Consider:
- Poker hand strength and scoring
- Joker synergies and multipliers
- Long-term strategy vs immediate gains
- Risk vs reward of different plays

Make your strategic decision and call the appropriate function."""

    def _should_stop(self, result: AgentResult) -> bool:
        """Determine if agent should continue or stop."""
        # Continue playing unless there's a critical error
        if not result.success:
            return True

        # Stop if we've reached a game over state
        if result.metadata and result.metadata.get('game_over'):
            return True

        # Stop if no more valid actions are available
        if result.metadata and result.metadata.get('no_actions_available'):
            return True

        return False

    def get_game_history(self) -> List[Dict[str, Any]]:
        """Get the complete game action history."""
        return self.game_history.copy()

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current game state."""
        return self.current_game_state.copy()

    def reset_game_state(self):
        """Reset game state tracking for a new game."""
        self.current_game_state = {}
        self.game_history = []
        logger.info('Game state reset')
