"""The agent should look again rather than reason about a screen mid-transition.

Balatro animates in stages, so a capture can land between them and see no cards
and no actionable buttons. Asked to decide from that, the model guesses -- it
chose 'next' on the Cash Out screen repeatedly.
"""

from __future__ import annotations

from ai_balatro.ai.agents.balatro_agent import BalatroReasoningAgent


class StubExtractor:
    def __init__(self, phases):
        self.phases = list(phases)
        self.calls = 0

    def capture_state(self, **_kwargs):
        self.calls += 1
        phase = self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]
        return None if phase is None else {'game_phase': phase}


def agent(phases):
    instance = BalatroReasoningAgent.__new__(BalatroReasoningAgent)
    instance.game_state_extractor = StubExtractor(phases)
    instance.UNKNOWN_PHASE_DELAY = 0.0
    return instance


def test_a_recognisable_screen_is_used_immediately():
    a = agent(['playing'])
    assert a._capture_recognisable_state()['game_phase'] == 'playing'
    assert a.game_state_extractor.calls == 1


def test_waits_for_the_screen_to_become_recognisable():
    a = agent(['unknown', 'unknown', 'blind_won'])
    assert a._capture_recognisable_state()['game_phase'] == 'blind_won'
    assert a.game_state_extractor.calls == 3


def test_gives_up_after_the_retries_and_returns_what_it_has():
    a = agent(['unknown'])
    state = a._capture_recognisable_state()
    assert state['game_phase'] == 'unknown'
    assert (
        a.game_state_extractor.calls == BalatroReasoningAgent.UNKNOWN_PHASE_RETRIES + 1
    )


def test_a_failed_capture_is_not_retried_as_unknown():
    a = agent([None])
    assert a._capture_recognisable_state() is None
    assert a.game_state_extractor.calls == 1
