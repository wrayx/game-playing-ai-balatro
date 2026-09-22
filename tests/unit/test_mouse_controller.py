"""Tests for mouse movement verification.

pynput accepts a position assignment and raises nothing when macOS denies
Accessibility, so a blocked sweep used to report success at every step.
"""

from __future__ import annotations

from ai_balatro.ai.actions.mouse_controller import MouseController


class StubMouse:
    """Mouse stub that optionally ignores position assignments."""

    def __init__(self, start=(0, 0), frozen: bool = False) -> None:
        self._position = start
        self.frozen = frozen

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if not self.frozen:
            self._position = value


def _controller(mouse: StubMouse) -> MouseController:
    controller = MouseController.__new__(MouseController)
    controller.mouse = mouse
    controller.mouse_move_duration = 0.0
    controller.mouse_move_steps = 2
    return controller


def test_reports_failure_when_cursor_cannot_move():
    """A frozen cursor means the OS blocked us; that is not success."""
    controller = _controller(StubMouse(start=(600, -286), frozen=True))

    assert controller.smooth_move_to(-239, -680) is False


def test_reports_success_when_cursor_reaches_target():
    controller = _controller(StubMouse(start=(600, -286)))

    assert controller.smooth_move_to(-239, -680) is True


def test_small_move_is_also_verified():
    """The short-distance shortcut must verify too, not assume.

    The offset here stays under smooth_move_to's 5px shortcut threshold while
    exceeding MOVE_TOLERANCE, which is the only band where a blocked move is
    still distinguishable from an arrival.
    """
    frozen = _controller(StubMouse(start=(100, 100), frozen=True))
    moving = _controller(StubMouse(start=(100, 100)))

    assert frozen.smooth_move_to(103, 103) is False
    assert moving.smooth_move_to(103, 103) is True


def test_tolerance_allows_small_rounding_drift():
    """Easing rounds to ints, so exact equality would be too strict."""
    mouse = StubMouse(start=(0, 0))
    controller = _controller(mouse)
    controller.smooth_move_to(50, 50)
    mouse._position = (50 + MouseController.MOVE_TOLERANCE, 50)

    assert controller._arrived(50, 50) is True

    mouse._position = (50 + MouseController.MOVE_TOLERANCE + 1, 50)
    assert controller._arrived(50, 50) is False
