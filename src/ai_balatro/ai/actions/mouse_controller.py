"""Mouse control and movement functionality for card actions."""

import time
import sys
import subprocess
from typing import Optional, Sequence, Callable, Tuple
from pynput import mouse
from ...core.screen_capture import ScreenCapture
from ...utils.logger import get_logger

logger = get_logger(__name__)


class MouseController:
    """Mouse controller for handling smooth movement, clicking, and window focus management."""

    #: Pixels of slack when confirming arrival. The final step assigns the exact
    #: target, so this only absorbs the sub-pixel drift macOS reports back on a
    #: scaled display. Keep it below the short-move threshold in smooth_move_to,
    #: or a blocked tiny move looks identical to a successful one.
    MOVE_TOLERANCE = 2

    def __init__(self, screen_capture: Optional[ScreenCapture] = None):
        """
        Initialize mouse controller.

        Args:
            screen_capture: Screen capture instance for window focus management
        """
        self.screen_capture = screen_capture

        # Mouse control
        self.mouse = mouse.Controller()
        self.mouse_button = mouse.Button  # Save Button reference

        # Click interval settings
        self.click_interval = 0.3  # Click interval (seconds)
        self.action_delay = 0.5  # Action delay (seconds)

        # Mouse movement animation settings (optimized speed)
        self.mouse_move_duration = 0.3  # Mouse movement duration (seconds)
        self.mouse_move_steps = 15  # Movement steps
        self.click_hold_duration = 0.08  # Click hold time (seconds)

        # Window focus settings
        self.ensure_window_focus = (
            True  # Whether to ensure window focus before clicking
        )
        self.focus_method = 'auto'  # Focus method: auto, click, applescript

        logger.info('MouseController initialized')
        logger.info(
            f'Mouse movement settings: duration={self.mouse_move_duration}s, steps={self.mouse_move_steps}, click hold={self.click_hold_duration}s'
        )
        logger.info(
            f'Window focus settings: enabled={self.ensure_window_focus}, method={self.focus_method}'
        )

    def set_animation_params(
        self,
        move_duration: float = 0.5,
        move_steps: int = 20,
        click_hold_duration: float = 0.1,
    ) -> None:
        """
        Set mouse movement animation parameters.

        Args:
            move_duration: Mouse movement duration (seconds)
            move_steps: Movement steps (more steps = smoother)
            click_hold_duration: Click hold time (seconds)
        """
        self.mouse_move_duration = move_duration
        self.mouse_move_steps = move_steps
        self.click_hold_duration = click_hold_duration

        logger.info(
            f'Updated mouse movement settings: duration={move_duration}s, steps={move_steps}, click hold={click_hold_duration}s'
        )

    def sweep_path(
        self,
        positions: Sequence[Tuple[int, int]],
        dwell_time: float = 0.25,
        move_duration: Optional[float] = None,
        capture_callback: Optional[Callable[[int, int, int], None]] = None,
        restore_position: bool = True,
    ) -> bool:
        """Sweep mouse cursor through positions with optional capture callback."""
        if not positions:
            return False

        original_position = self.mouse.position
        original_move_duration = self.mouse_move_duration
        original_move_steps = self.mouse_move_steps

        try:
            if move_duration is not None:
                # Adjust movement parameters for faster sweep
                self.mouse_move_duration = move_duration
                self.mouse_move_steps = max(5, int(move_duration / 0.01))

            for idx, (x, y) in enumerate(positions):
                if not self.smooth_move_to(x, y):
                    logger.warning(f'Failed to reach sweep position {idx}: ({x}, {y})')
                    continue

                time.sleep(max(dwell_time, 0.0))

                if capture_callback:
                    try:
                        capture_callback(idx, x, y)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f'Sweep capture callback failed at {idx}: {exc}')

            return True

        finally:
            if move_duration is not None:
                self.mouse_move_duration = original_move_duration
                self.mouse_move_steps = original_move_steps

            if restore_position:
                try:
                    self.mouse.position = original_position
                except Exception:
                    logger.debug(
                        'Failed to restore original mouse position after sweep'
                    )

    def smooth_move_to(self, target_x: int, target_y: int) -> bool:
        """
        Smoothly move mouse to target position.

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate

        Returns:
            Whether movement was successful
        """
        try:
            import math

            # Get current mouse position
            start_x, start_y = self.mouse.position

            logger.info(
                f'Smooth mouse move: ({start_x}, {start_y}) -> ({target_x}, {target_y})'
            )

            # Calculate movement distance
            distance_x = target_x - start_x
            distance_y = target_y - start_y
            total_distance = math.sqrt(distance_x**2 + distance_y**2)

            if total_distance < 5:  # If distance is small, move directly
                self.mouse.position = (target_x, target_y)
                time.sleep(0.1)
                return self._arrived(target_x, target_y)

            # Calculate step movement amount
            step_delay = self.mouse_move_duration / self.mouse_move_steps

            for i in range(self.mouse_move_steps + 1):
                # Use easing function for more natural movement (slow-fast-slow)
                progress = i / self.mouse_move_steps
                # Use ease-in-out easing function
                eased_progress = self._ease_in_out(progress)

                current_x = int(start_x + distance_x * eased_progress)
                current_y = int(start_y + distance_y * eased_progress)

                self.mouse.position = (current_x, current_y)

                # Last step ensures precise arrival at target
                if i == self.mouse_move_steps:
                    self.mouse.position = (target_x, target_y)

                time.sleep(step_delay)

            return self._arrived(target_x, target_y)

        except Exception as e:
            logger.error(f'Smooth mouse movement failed: {e}')
            # fallback: direct movement
            try:
                self.mouse.position = (target_x, target_y)
                time.sleep(0.2)
                return self._arrived(target_x, target_y)
            except Exception:
                return False

    def _arrived(self, target_x: int, target_y: int) -> bool:
        """Confirm the cursor actually reached the target.

        pynput raises nothing when macOS denies Accessibility -- the assignment
        is accepted and the cursor simply never moves. Without this check an
        entire hover sweep no-ops while every step reports success, which looks
        identical to the game not rendering tooltips.
        """
        try:
            final_x, final_y = self.mouse.position
        except Exception:  # noqa: BLE001
            logger.warning('Could not read mouse position to confirm movement')
            return False

        if (
            abs(final_x - target_x) <= self.MOVE_TOLERANCE
            and abs(final_y - target_y) <= self.MOVE_TOLERANCE
        ):
            logger.info(f'Mouse movement complete: ({final_x}, {final_y})')
            return True

        logger.warning(
            f'Mouse did not reach ({target_x}, {target_y}); cursor is still at '
            f'({final_x}, {final_y}). On macOS this usually means Accessibility '
            f'permission has not been granted to this terminal or IDE.'
        )
        return False

    def click_at(self, x: int, y: int, move_first: bool = True) -> bool:
        """
        Click at specified position.

        Args:
            x: X coordinate
            y: Y coordinate
            move_first: Whether to move to target position first

        Returns:
            Whether click was successful
        """
        try:
            if move_first:
                if not self.smooth_move_to(x, y):
                    logger.error('Failed to move to target position')
                    return False
                time.sleep(0.2)  # Brief pause after movement

            # Execute click operation (press-hold-release)
            logger.info(f'Executing click at position ({x}, {y})')
            self.mouse.press(self.mouse_button.left)
            time.sleep(self.click_hold_duration)  # Hold pressed state
            self.mouse.release(self.mouse_button.left)

            # Brief wait after click
            time.sleep(0.1)
            return True

        except Exception as e:
            logger.error(f'Click operation failed: {e}')
            return False

    def ensure_game_window_focus(self) -> bool:
        """
        Ensure game window has focus.

        Returns:
            Whether focus was successfully obtained
        """
        if not self.ensure_window_focus or not self.screen_capture:
            return True  # Return success if focus check is disabled

        try:
            # Get capture region information
            capture_region = self.screen_capture.get_capture_region()
            if not capture_region:
                logger.warning('Cannot get capture region, skipping focus handling')
                return True

            # Detect current system
            if sys.platform == 'darwin':  # macOS
                return self._ensure_focus_macos(capture_region)
            elif sys.platform == 'win32':  # Windows
                return self._ensure_focus_windows(capture_region)
            else:  # Linux/others
                return self._ensure_focus_linux(capture_region)

        except Exception as e:
            logger.warning(f'Window focus handling failed: {e}')
            return True  # Don't block subsequent operations on failure

    #: Where to rest the cursor, as a fraction of the window. Empty felt on
    #: every screen: above the shop panel, above the hand, below the score row.
    PARK_POSITION = (0.5, 0.22)

    def park_cursor(self) -> bool:
        """Move the cursor somewhere it raises no tooltip.

        Whatever the cursor rests on renders a tooltip, and that tooltip is
        captured along with everything else. Left on a shop item it covers the
        Next Round and Reroll buttons entirely, so the UI model reports only
        Options and Run Info and the agent cannot leave the shop.
        """
        if not self.screen_capture:
            return False

        region = self.screen_capture.get_capture_region()
        if not region:
            return False

        x_fraction, y_fraction = self.PARK_POSITION
        return self.smooth_move_to(
            int(region['left'] + region['width'] * x_fraction),
            int(region['top'] + region['height'] * y_fraction),
        )

    def _game_pid(self) -> Optional[int]:
        """PID of the detected game window, if the capture layer found one."""
        if not self.screen_capture:
            return None
        info = self.screen_capture.get_window_info() or {}
        pid = info.get('pid')
        return int(pid) if pid else None

    def _is_game_active(self, pid: int) -> bool:
        """Whether the game owns the frontmost ordinary window.

        Read from Quartz window order rather than NSWorkspace. NSWorkspace's
        frontmostApplication() is cached per process and only refreshes on a
        run loop, so polling it in a script returns whatever was true at first
        access -- it reported a stale app through an entire focus attempt and
        made working activation look broken.

        The tradeoff is that an app holding activation without any windows, such
        as Finder, is not detected. That does not matter for either caller: an
        app with no windows occludes nothing, and a click still lands on the
        game's window.
        """
        return self._front_window_pid() == pid

    def _wait_for_activation(self, pid: int, timeout: float = 2.0) -> bool:
        """Poll until the OS reports the game active, or give up.

        Activation is asynchronous: osascript returns before NSWorkspace
        reflects the change. A single short sleep reported failure while the
        switch was still in flight, which made every focus attempt look broken.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_game_active(pid):
                return True
            time.sleep(0.1)
        return False

    def _front_window_pid(self) -> Optional[int]:
        """PID owning the frontmost ordinary window, per Quartz window order."""
        try:
            import Quartz

            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
        except Exception:  # noqa: BLE001
            return None

        for window in windows:
            # Front to back; the first ordinary window is the frontmost one.
            if window.get('kCGWindowLayer', 0) != 0:
                continue
            return window.get('kCGWindowOwnerPID')
        return None

    def is_game_foreground(self) -> bool:
        """Whether the game owns the window on top of its own capture region.

        Capture takes a screen rectangle, not a window, so anything sitting over
        the game is captured in its place -- YOLO has read cards out of video
        thumbnails that way, and an agent run issued a click request while a
        browser covered the board. Every action path checks this before
        clicking, so a covered game is refused rather than clicked through.
        """
        pid = self._game_pid()
        if pid is None:
            logger.warning('No game window recorded; cannot confirm foreground')
            return False

        front = self._front_window_pid()
        if front != pid:
            logger.warning(
                f'Another window (pid {front}) is in front of the game (pid {pid})'
            )
            return False
        return True

    def _ensure_focus_macos(self, capture_region: dict) -> bool:
        """Ensure the game window is frontmost, verifying rather than assuming.

        Balatro runs on LOVE, so System Events reports the process as 'love'
        while Quartz reports the window owner as 'Balatro'. The previous script
        compared the frontmost process against "Balatro", so it never matched,
        then tried to activate an application of that name -- which does not
        resolve. Every call hit the 2s timeout, and the timeout handler skipped
        the click fallback and returned True anyway. Focus never worked, and the
        first card click was silently consumed activating the window, leaving
        one fewer card selected than the model asked for.
        """
        pid = self._game_pid()

        if pid and self._is_game_active(pid):
            return True

        if pid and self.focus_method in ('applescript', 'auto'):
            script = (
                'tell application "System Events" to set frontmost of '
                f'(first application process whose unix id is {pid}) to true'
            )
            try:
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode != 0:
                    logger.warning(
                        f'AppleScript activation failed: {result.stderr.strip()}'
                    )
            except subprocess.TimeoutExpired:
                logger.warning('AppleScript execution timeout')
            except Exception as e:  # noqa: BLE001
                logger.warning(f'macOS focus handling failed: {e}')

            if self._wait_for_activation(pid):
                logger.info('Activated game window using AppleScript')
                return True

        if self.focus_method in ('click', 'auto'):
            self._click_to_focus(capture_region)
            if pid is None:
                time.sleep(0.2)
                return True  # nothing to verify against
            if self._wait_for_activation(pid):
                logger.info('Activated game window by clicking its title bar')
                return True

        logger.warning(
            'Could not confirm the game is the active application. The next click '
            'will likely be consumed activating it, selecting one fewer card '
            'than requested.'
        )
        return False

    def _ensure_focus_windows(self, capture_region: dict) -> bool:
        """Ensure window focus on Windows system."""
        try:
            # Use click activation on Windows
            return self._click_to_focus(capture_region)
        except Exception as e:
            logger.warning(f'Windows focus handling failed: {e}')
            return True

    def _ensure_focus_linux(self, capture_region: dict) -> bool:
        """Ensure window focus on Linux system."""
        try:
            # Use click activation on Linux
            return self._click_to_focus(capture_region)
        except Exception as e:
            logger.warning(f'Linux focus handling failed: {e}')
            return True

    def _click_to_focus(self, capture_region: dict) -> bool:
        """Get focus by clicking on window."""
        try:
            # Calculate click position for window title bar (window top center)
            title_bar_x = capture_region['left'] + capture_region['width'] // 2
            title_bar_y = capture_region['top'] + 10  # Title bar area

            current_pos = self.mouse.position

            # Quick click on title bar to activate window
            self.mouse.position = (title_bar_x, title_bar_y)
            time.sleep(0.05)
            self.mouse.click(self.mouse_button.left)
            time.sleep(0.1)

            # Restore mouse position
            self.mouse.position = current_pos

            logger.info(f'Activated window by clicking: ({title_bar_x}, {title_bar_y})')
            return True

        except Exception as e:
            logger.warning(f'Failed to activate window by clicking: {e}')
            return False

    def _ease_in_out(self, t: float) -> float:
        """
        Easing function: slow-fast-slow movement effect.

        Args:
            t: Progress value (0.0 to 1.0)

        Returns:
            Eased progress value
        """
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
