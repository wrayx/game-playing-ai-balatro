"""Tests for reading the shop's stock and prices.

Owned jokers and shop jokers share a class, so they cannot be told apart by
class alone. A price tag above the item is what marks it as for sale.
"""

from __future__ import annotations

from ai_balatro.core import entities
from ai_balatro.core.detection import Detection


def item(class_name: str, x: int, y: int = 240) -> Detection:
    return Detection(0, class_name, 0.95, (x, y, x + 70, y + 110))


def price(x: int, y: int = 205) -> Detection:
    return Detection(0, 'ui_card_value', 0.95, (x, y, x + 30, y + 22))


def test_candidates_cover_everything_the_shop_sells():
    found = {
        d.class_name
        for d in entities.shop_item_candidates(
            [
                item('joker_card', 100),
                item('tarot_card', 200),
                item('planet_card', 300),
                item('spectral_card', 400),
                item('card_pack', 500),
                item('poker_card_front', 600),
            ]
        )
    }
    assert found == {
        'joker_card',
        'tarot_card',
        'planet_card',
        'spectral_card',
        'card_pack',
    }


def test_pairs_each_item_with_the_tag_above_it():
    jokers = [item('joker_card', 100), item('joker_card', 300)]
    tags = [price(120), price(320)]
    paired = entities.pair_with_prices(jokers, tags)
    assert [p is not None for _, p in paired] == [True, True]
    assert paired[0][1].bbox[0] == 120
    assert paired[1][1].bbox[0] == 320


def test_an_owned_joker_has_no_price():
    """Owned jokers sit in their own row with no tag, which is the signal."""
    owned = item('joker_card', 100, y=60)
    for_sale = item('joker_card', 300)
    paired = entities.pair_with_prices([owned, for_sale], [price(320)])
    assert paired[0][1] is None
    assert paired[1][1] is not None


def test_a_tag_far_above_is_not_matched():
    """Guards against pairing across rows."""
    paired = entities.pair_with_prices([item('joker_card', 100)], [price(120, y=10)])
    assert paired[0][1] is None


def test_a_tag_below_the_item_is_not_matched():
    paired = entities.pair_with_prices([item('joker_card', 100)], [price(120, y=400)])
    assert paired[0][1] is None


def test_horizontally_offset_tag_is_not_matched():
    paired = entities.pair_with_prices([item('joker_card', 100)], [price(900)])
    assert paired[0][1] is None


def test_results_are_ordered_left_to_right():
    items = [item('joker_card', 500), item('tarot_card', 100), item('card_pack', 300)]
    ordered = [i.bbox[0] for i, _ in entities.pair_with_prices(items, [])]
    assert ordered == [100, 300, 500]


def test_a_tag_touching_the_item_is_matched():
    """The game draws the tag against the card's top edge; measured at one
    pixel of overlap, which a non-negative-gap rule dropped."""
    joker = item('joker_card', 489, y=251)
    tag = Detection(0, 'ui_card_value', 0.96, (507, 221, 554, 252))
    assert entities.pair_with_prices([joker], [tag])[0][1] is tag


def test_a_tag_overlapping_far_into_the_item_is_not_matched():
    joker = item('joker_card', 489, y=251)
    deep = Detection(0, 'ui_card_value', 0.96, (507, 300, 554, 340))
    assert entities.pair_with_prices([joker], [deep])[0][1] is None
