"""Tests for carrying what has already happened in this blind.

The screen shows the board but not the path to it: how many hands have gone,
what each scored, whether discards were spent. That is the one thing a turn
cannot re-read, and a boss like The Hook -- which debuffs cards played earlier
in the ante -- makes it matter.
"""

from __future__ import annotations

from ai_balatro.ai.agents.balatro_agent import BalatroReasoningAgent


def agent(history=None):
    instance = BalatroReasoningAgent.__new__(BalatroReasoningAgent)
    instance.game_history = history or []
    return instance


def state(target='300', score='0'):
    return {
        'ui_text_elements': [
            {'class_name': 'ui_score_target_score', 'text': target},
            {'class_name': 'ui_score_round_score', 'text': score},
        ]
    }


def record(name='play_cards', count=5, target='300', before='0', after='296', ok=True):
    return {
        'action_name': name,
        'card_count': count,
        'target': target,
        'score_before': before,
        'score_after': after,
        'success': ok,
    }


def test_no_summary_before_anything_has_happened():
    assert agent()._blind_progress_summary(state()) == ''


def test_lists_each_hand_with_what_it_scored():
    summary = agent([record()])._blind_progress_summary(state())
    assert 'THIS BLIND SO FAR' in summary
    assert '1. play_cards 5 cards, score 0 to 296' in summary


def test_numbers_the_actions_in_order():
    summary = agent(
        [record(after='296'), record(name='discard_cards', count=3, after='296')]
    )._blind_progress_summary(state())
    assert '1. play_cards' in summary
    assert '2. discard_cards 3 cards' in summary


def test_a_new_blind_starts_a_fresh_list():
    """Entries are scoped by target score, so the transition needs no detecting."""
    history = [record(target='300'), record(target='450', after='120')]
    summary = agent(history)._blind_progress_summary(state(target='450'))
    assert summary.count('play_cards') == 1
    assert 'score 0 to 120' in summary


def test_failed_actions_are_not_reported_as_history():
    assert agent([record(ok=False)])._blind_progress_summary(state()) == ''


def test_no_summary_when_the_target_cannot_be_read():
    assert agent([record()])._blind_progress_summary(state(target='')) == ''


def test_outcome_is_attributed_on_the_next_capture():
    """The score animates past the action returning, so it can only be read
    on the next settled board."""
    pending = record(after=None)
    a = agent([pending])
    a._record_outcome_of_last_action(state(score='296'))
    assert pending['score_after'] == '296'


def test_an_already_scored_action_is_not_overwritten():
    done = record(after='296')
    a = agent([done])
    a._record_outcome_of_last_action(state(score='999'))
    assert done['score_after'] == '296'
