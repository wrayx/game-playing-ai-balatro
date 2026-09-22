import pytest

from ai_balatro.utils.card_text_parser import parse_card_description


@pytest.mark.parametrize(
    'raw_text, expected_rank, expected_suit, expected_short',
    [
        ('方片K\n+10筹码', 'K', 'diamonds', 'KD'),
        ('黑桃7\n+7筹码', '7', 'spades', '7S'),
        ('海化2\n+2筹码', '2', 'clubs', '2C'),
        ('红挑A', 'A', 'hearts', 'AH'),
    ],
)
def test_parse_card_description_honors_rank_and_suit(
    raw_text: str, expected_rank: str, expected_suit: str, expected_short: str
) -> None:
    result = parse_card_description(raw_text)
    assert result is not None
    assert result['valid'] is True
    assert result['rank'] == expected_rank
    assert result['suit'] == expected_suit
    assert result['short_code'] == expected_short
    assert expected_suit.title() in result['english_name']


def test_parse_card_description_gracefully_handles_empty_text() -> None:
    assert parse_card_description('') is None
    assert parse_card_description('   ') is None


@pytest.mark.parametrize(
    'raw_text, expected_rank, expected_suit',
    [
        # English tooltips: rank on the first line, suit on the second.
        ('10of\nHearts\n+10 chips', '10', 'hearts'),
        ('9of\nHearts\n+9 chips', '9', 'hearts'),
        ('Aceof\nDiamonds\n+11 chips', 'A', 'diamonds'),
        ('Kingof\nClubs\n+10 chips', 'K', 'clubs'),
        # Suit garbled by OCR: keep the rank, report no suit rather than guess.
        ('6of\nsapedc\n+6 chips', '6', None),
    ],
)
def test_parses_english_two_line_tooltips(
    raw_text: str, expected_rank: str, expected_suit: str
) -> None:
    result = parse_card_description(raw_text)
    assert result is not None
    assert result['rank'] == expected_rank
    assert result['suit'] == expected_suit


@pytest.mark.parametrize(
    'raw_text',
    [
        'Rce ot\nDiamonds\n+11 chips',  # 'Ace of' misread; the 't' used to mean Ten
        'Mueenor\nHearts\n+10 chips',  # 'Queen of' misread
        'Bueen\n101\nWamonos\n+10 chips',
    ],
)
def test_garbled_rank_yields_no_rank_rather_than_a_wrong_one(raw_text: str) -> None:
    """A wrong rank is worse than none: it used to hide the raw text entirely."""
    result = parse_card_description(raw_text)
    assert result is not None
    assert result['rank'] is None
    assert result['english_name'] is None
