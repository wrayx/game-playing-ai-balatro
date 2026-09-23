"""Action schemas and data structures for Balatro game actions."""

from enum import Enum
from typing import List
from dataclasses import dataclass


class ActionType(Enum):
    """Types of game actions."""

    SELECT_CARDS = 'select_cards'
    PLAY_CARDS = 'play_cards'
    DISCARD_CARDS = 'discard_cards'
    HOVER_CARD = 'hover_card'
    CLICK_BUTTON = 'click_button'


@dataclass
class CardAction:
    """Represents an action on cards using position array."""

    action_type: ActionType
    positions: List[int]  # Array like [1, 1, 1, 0] or [-1, -1, 0, 0]
    description: str = ''

    @property
    def selected_indices(self) -> List[int]:
        """Get indices of cards to select (1 values)."""
        return [i for i, val in enumerate(self.positions) if val == 1]

    @property
    def discard_indices(self) -> List[int]:
        """Get indices of cards to discard (-1 values)."""
        return [i for i, val in enumerate(self.positions) if val == -1]

    @property
    def is_play_action(self) -> bool:
        """Check if this is a play action (has positive values)."""
        return any(val > 0 for val in self.positions)

    @property
    def is_discard_action(self) -> bool:
        """Check if this is a discard action (has negative values)."""
        return any(val < 0 for val in self.positions)

    @classmethod
    def from_array(cls, positions: List[int], description: str = '') -> 'CardAction':
        """Create CardAction from position array."""
        if any(val > 0 for val in positions):
            action_type = ActionType.PLAY_CARDS
        elif any(val < 0 for val in positions):
            action_type = ActionType.DISCARD_CARDS
        else:
            action_type = ActionType.SELECT_CARDS

        return cls(
            action_type=action_type, positions=positions, description=description
        )


# Maps a button type to the exact UI-model classes that satisfy it.
#
# These were keyword lists matched as substrings against class names, which
# collided badly: 'shop' matched button_store_reroll (rerolling the shop at $5
# a time), 'play' matched button_new_run_play (starting a new run from a menu),
# and 'discard' matched ui_data_discards_left, the counter. Exact class names
# cannot collide. Anything not listed here is rejected rather than guessed at.
BUTTON_CONFIG = {
    # In a blind
    'play': {'classes': ['button_play'], 'description': 'Play the selected cards'},
    'discard': {
        'classes': ['button_discard'],
        'description': 'Discard the selected cards',
    },
    'sort_hand_rank': {
        'classes': ['button_sort_hand_rank'],
        'description': 'Sort the hand by rank',
    },
    'sort_hand_suits': {
        'classes': ['button_sort_hand_suits'],
        'description': 'Sort the hand by suit',
    },
    # Advancing the run
    'cash_out': {
        'classes': ['button_cash_out'],
        'description': 'Collect the reward after beating a blind',
    },
    'level_select': {
        'classes': ['button_level_select'],
        'description': 'Choose the next blind',
    },
    'skip': {
        'classes': ['button_level_skip', 'button_card_pack_skip'],
        'description': 'Skip this blind or booster pack',
    },
    'next': {
        'classes': ['button_store_next_round'],
        'description': 'Leave the shop and start the next round',
    },
    # Shop controls. Present so the executor can reach them once shop reasoning
    # exists; deliberately absent from the LLM schema until then.
    'purchase': {'classes': ['button_purchase'], 'description': 'Buy the item'},
    'sell': {'classes': ['button_sell'], 'description': 'Sell the item'},
    'reroll': {
        'classes': ['button_store_reroll'],
        'description': 'Reroll the shop stock',
    },
    'use': {'classes': ['button_use'], 'description': 'Use the consumable'},
    # Menu screens. Config-only: the agent has no business pressing these, and
    # button_main_menu_play / button_new_run_play used to resolve to 'play',
    # so asking to play a hand could start a new run.
    'main_menu': {'classes': ['button_main_menu'], 'description': 'Main menu'},
    'main_menu_play': {
        'classes': ['button_main_menu_play'],
        'description': 'Play from the main menu',
    },
    'new_run': {'classes': ['button_new_run'], 'description': 'Start a new run'},
    'new_run_play': {
        'classes': ['button_new_run_play'],
        'description': 'Confirm the new run',
    },
    # Informational
    'run_info': {'classes': ['button_run_info'], 'description': 'Show run info'},
    'options': {'classes': ['button_options'], 'description': 'Open options'},
    'back': {'classes': ['button_back'], 'description': 'Go back'},
}

#: Button types the agent may request. A subset of BUTTON_CONFIG: the shop
#: controls are reachable by code but not offered to the model, which has no
#: reasoning for buying or selling yet.
AGENT_BUTTON_TYPES = [
    'play',
    'discard',
    'sort_hand_rank',
    'sort_hand_suits',
    'cash_out',
    'level_select',
    'skip',
    'next',
]


# Function calling schemas for LLM integration
GAME_ACTIONS = [
    {
        'name': 'play_cards',
        'description': 'Play selected cards from your hand by their index numbers (0-based)',
        'parameters': {
            'type': 'object',
            'properties': {
                'indices': {
                    'type': 'array',
                    'items': {'type': 'integer', 'minimum': 0},
                    'description': 'Array of card indices to play (e.g., [0, 1, 2] plays cards 0, 1, and 2)',
                },
                'description': {
                    'type': 'string',
                    'description': 'Strategy explanation for this play',
                },
            },
            'required': ['indices'],
        },
    },
    {
        'name': 'discard_cards',
        'description': 'Discard selected cards from your hand by their index numbers (0-based)',
        'parameters': {
            'type': 'object',
            'properties': {
                'indices': {
                    'type': 'array',
                    'items': {'type': 'integer', 'minimum': 0},
                    'description': 'Array of card indices to discard (e.g., [3, 4] discards cards 3 and 4)',
                },
                'description': {
                    'type': 'string',
                    'description': 'Reason for discarding these cards',
                },
            },
            'required': ['indices'],
        },
    },
    {
        'name': 'buy_item',
        'description': (
            'Buy one item from the shop by its position, counting from the '
            'left starting at 0'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'index': {
                    'type': 'integer',
                    'description': 'Which shop item to buy, 0 for the leftmost',
                },
                'description': {
                    'type': 'string',
                    'description': 'Why this item is worth its price',
                },
            },
            'required': ['index'],
        },
    },
    {
        'name': 'choose_from_pack',
        'description': (
            'Take one item from an opened booster pack, by its position '
            'counting from the left starting at 0'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'index': {
                    'type': 'integer',
                    'description': 'Which offered item to take, 0 for the leftmost',
                },
                'description': {
                    'type': 'string',
                    'description': 'Why this one is the best of what is offered',
                },
            },
            'required': ['index'],
        },
    },
    {
        'name': 'click_button',
        'description': 'Click game interface button',
        'parameters': {
            'type': 'object',
            'properties': {
                'button_type': {
                    'type': 'string',
                    'enum': AGENT_BUTTON_TYPES,
                    'description': 'Button type to click',
                }
            },
            'required': ['button_type'],
        },
    },
]
