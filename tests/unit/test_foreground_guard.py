"""Tests for refusing to act when something covers the game.

Capture takes a screen rectangle rather than a window, so a browser over the
board is analysed as if it were the board -- YOLO has reported cards from video
thumbnails that way. Acting on that clicks inside another application.
"""

from __future__ import annotations

from ai_balatro.ai.actions.mouse_controller import MouseController

GAME_PID = 16205
OTHER_PID = 28530


def controller(front_pid, game_pid=GAME_PID):
    instance = MouseController.__new__(MouseController)
    instance._front_window_pid = lambda: front_pid
    instance._game_pid = lambda: game_pid
    return instance


def test_foreground_when_the_game_owns_the_front_window():
    assert controller(GAME_PID).is_game_foreground() is True


def test_not_foreground_when_another_window_is_in_front():
    assert controller(OTHER_PID).is_game_foreground() is False


def test_not_foreground_when_no_window_was_detected():
    assert controller(GAME_PID, game_pid=None).is_game_foreground() is False


def test_not_foreground_when_nothing_is_in_front():
    assert controller(None).is_game_foreground() is False


def test_is_game_active_reads_window_order():
    """NSWorkspace caches per process and never refreshes without a run loop,
    which made a working activation look like a failure."""
    assert controller(GAME_PID)._is_game_active(GAME_PID) is True
    assert controller(OTHER_PID)._is_game_active(GAME_PID) is False


class TestCursorParking:
    """A resting cursor raises a tooltip, and the tooltip is captured too."""

    def _controller(self, region):
        instance = MouseController.__new__(MouseController)
        instance.screen_capture = type(
            'C', (), {'get_capture_region': staticmethod(lambda: region)}
        )()
        moved = []
        instance.smooth_move_to = lambda x, y: moved.append((x, y)) or True
        instance._moved = moved
        return instance

    def test_parks_inside_the_window_above_the_content(self):
        region = {'left': 8, 'top': 49, 'width': 932, 'height': 602}
        controller = self._controller(region)
        assert controller.park_cursor() is True
        x, y = controller._moved[0]
        assert region['left'] < x < region['left'] + region['width']
        assert region['top'] < y < region['top'] + region['height'] * 0.4

    def test_no_capture_region_is_not_an_error(self):
        assert self._controller(None).park_cursor() is False
