"""Tests for re-detecting the game window.

The region is found once at construction and cached, so a moved or resized
window -- or a game restarted since -- leaves captures pointing at a stale
rectangle, reading whatever now occupies it rather than the game.
"""

from __future__ import annotations

from ai_balatro.core.screen_capture import ScreenCapture


def capture(found, new_region=None):
    instance = ScreenCapture.__new__(ScreenCapture)
    instance.capture_region = {'left': 8, 'top': 49, 'width': 932, 'height': 602}

    def detect():
        if found and new_region is not None:
            instance.capture_region = new_region
        return found

    instance._detect_balatro_window = detect
    return instance


def test_updates_the_region_when_the_window_has_moved():
    moved = {'left': 400, 'top': 100, 'width': 932, 'height': 602}
    instance = capture(True, moved)
    assert instance.refresh_window_region() is True
    assert instance.capture_region == moved


def test_reports_failure_when_no_window_is_found():
    instance = capture(False)
    assert instance.refresh_window_region() is False


def test_an_unmoved_window_is_left_alone():
    same = {'left': 8, 'top': 49, 'width': 932, 'height': 602}
    instance = capture(True, same)
    assert instance.refresh_window_region() is True
    assert instance.capture_region == same


def test_no_prior_region_is_not_an_error():
    instance = capture(True, {'left': 0, 'top': 0, 'width': 10, 'height': 10})
    instance.capture_region = None
    assert instance.refresh_window_region() is True
