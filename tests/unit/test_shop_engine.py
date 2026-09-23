"""Tests for buying from the shop.

Every step is verified against the screen, because a shop click that does not
land spends money on nothing or buys the wrong item -- and the shop redraws
either way, so the redraw itself proves nothing.
"""

from __future__ import annotations

import numpy as np

from ai_balatro.ai.actions.shop_engine import ShopActionEngine
from ai_balatro.core.detection import Detection

FRAME = np.zeros((602, 932, 3), dtype=np.uint8)


def item(x, class_name='joker_card', y=240):
    return Detection(0, class_name, 0.95, (x, y, x + 70, y + 110))


def tag(x, y=210):
    return Detection(0, 'ui_card_value', 0.95, (x, y, x + 30, y + 30))


def button(name, x):
    return Detection(0, name, 0.95, (x, 380, x + 60, 410))


class StubDetector:
    def __init__(self, entities_list, ui_list):
        self.entities_list = entities_list
        self.ui_list = ui_list

    def detect_entities(self, _f, *a, **k):
        return list(self.entities_list)

    def detect_ui(self, _f, *a, **k):
        return list(self.ui_list)


def engine(entities_list, ui_list, foreground=True):
    instance = ShopActionEngine.__new__(ShopActionEngine)
    instance.multi_detector = StubDetector(entities_list, ui_list)
    instance.screen_capture = type(
        'C',
        (),
        {
            'capture_once': staticmethod(lambda: FRAME),
            'get_capture_region': staticmethod(
                lambda: {'left': 0, 'top': 0, 'width': 932, 'height': 602}
            ),
        },
    )()
    instance.mouse_controller = type(
        'M',
        (),
        {
            'is_game_foreground': staticmethod(lambda: foreground),
            # Clicks fail in tests; the buy path should report that rather
            # than raise, and the pack guard must run before any of it.
            'click_at': staticmethod(lambda *a, **k: False),
        },
    )()
    instance.ui_text_service = None
    instance.button_detector = None
    return instance


class TestReadingTheShop:
    def test_lists_only_items_with_a_price(self):
        e = engine(
            [item(100), item(300), item(600, class_name='joker_card', y=60)],
            [tag(120), tag(320)],
        )
        assert len(e.shop_items(FRAME)) == 2

    def test_orders_left_to_right(self):
        e = engine([item(500), item(100)], [tag(520), tag(120)])
        assert [i.bbox[0] for i, _ in e.shop_items(FRAME)] == [100, 500]


class TestGuards:
    def test_refuses_when_another_window_is_in_front(self):
        e = engine([item(100)], [tag(120)], foreground=False)
        result = e.execute_buy(0)
        assert result['success'] is False
        assert 'in front of the game' in result['error_message']

    def test_refuses_an_index_that_is_not_for_sale(self):
        e = engine([item(100)], [tag(120)])
        result = e.execute_buy(3)
        assert result['success'] is False
        assert 'Invalid shop index 3' in result['error_message']

    def test_refuses_when_nothing_is_for_sale(self):
        e = engine([item(100)], [])
        result = e.execute_buy(0)
        assert result['success'] is False
        assert 'No items for sale' in result['error_message']


class TestButtonAlignment:
    """Balatro draws Buy under the selected item, so its position says which
    item is selected. Without the check a stale selection would be bought."""

    def test_button_under_the_item_belongs_to_it(self):
        e = engine([], [])
        assert e._button_belongs_to(button('button_purchase', 110), item(100)) is True

    def test_button_under_a_different_item_does_not(self):
        e = engine([], [])
        assert e._button_belongs_to(button('button_purchase', 700), item(100)) is False

    def test_slight_offset_is_tolerated(self):
        """The check reads the button's centre, so position it by centre."""
        e = engine([], [])
        target_centre = item(100).bbox[2] + ShopActionEngine.BUTTON_ALIGNMENT_SLACK - 5
        just_inside = button('button_purchase', target_centre - 30)
        assert just_inside.center[0] == target_centre
        assert e._button_belongs_to(just_inside, item(100)) is True

    def test_offset_beyond_the_slack_is_rejected(self):
        e = engine([], [])
        target_centre = item(100).bbox[2] + ShopActionEngine.BUTTON_ALIGNMENT_SLACK + 5
        assert (
            e._button_belongs_to(
                button('button_purchase', target_centre - 30), item(100)
            )
            is False
        )


class TestNextRoundFallback:
    """The UI model misses the shop's Next Round button in some states, which
    would otherwise strand the agent in the shop after any purchase that
    leaves it short of the reroll price."""

    def test_position_is_inside_the_window(self):
        e = engine([], [])
        x, y = e.next_round_fallback_position()
        assert 0 < x < 932
        assert 0 < y < 602

    def test_looks_like_shop_needs_something_for_sale(self):
        assert engine([item(100)], [tag(120)]).looks_like_shop(FRAME) is True
        assert engine([item(100)], []).looks_like_shop(FRAME) is False


class TestPackPurchases:
    """Packs were refused while their selection screen was unhandled. Now that
    contents can be read and chosen from, buying one is allowed again."""

    def test_buying_a_pack_is_no_longer_refused_outright(self):
        e = engine([item(100, class_name='card_pack')], [tag(120)])
        result = e.execute_buy(0)
        assert 'booster packs' not in result['error_message']

    def test_still_buys_a_joker(self):
        e = engine([item(100, class_name='joker_card')], [tag(120)])
        result = e.execute_buy(0)
        # Fails later for want of a real screen, but not on the pack guard.
        assert 'booster packs' not in result['error_message']


class TestPackSelectOffset:
    """Select has no UI class, so it is clicked relative to the chosen card.

    Scaled by the card's height rather than the window's: measured at 9px below
    a 99px joker and 11px below a 118px playing card. A window-relative offset
    put the click just under the button and the choice did not register.
    """

    def test_offset_lands_inside_the_button_for_a_joker(self):
        card_top, card_bottom = 361, 460
        height = card_bottom - card_top
        y = card_bottom + height * ShopActionEngine.SELECT_OFFSET_FRACTION
        assert 458 <= y <= 480, f'{y} is outside the measured Select button'

    def test_offset_lands_inside_the_button_for_a_playing_card(self):
        card_top, card_bottom = 352, 470
        height = card_bottom - card_top
        y = card_bottom + height * ShopActionEngine.SELECT_OFFSET_FRACTION
        assert 468 <= y <= 492, f'{y} is outside the measured Select button'
