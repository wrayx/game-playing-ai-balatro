"""Burst-capture the Balatro window and answer the open question:
when a tooltip is raised over the hand, does it shift card indices?

Captures only the game window region. No mouse control, no clicks.
"""

import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[5] / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import cv2
from ai_balatro.core import entities
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector
from ai_balatro.core.screen_capture import ScreenCapture

SP, SECONDS, INTERVAL = sys.argv[1], float(sys.argv[2]), 0.5
LEGACY_PLAYABLE = {
    'poker_card_front',
    'joker_card',
    'tarot_card',
    'planet_card',
    'spectral_card',
}


def old_prompt(ds):  # every class contains 'card', so all pass
    return sorted(
        [
            d
            for d in ds
            if 'card' in d.class_name.lower() and 'tooltip' not in d.class_name.lower()
        ],
        key=lambda d: d.bbox[0],
    )


def old_exec(ds):
    return sorted(
        [
            d
            for d in ds
            if d.class_name.lower() in LEGACY_PLAYABLE
            and 'description' not in d.class_name.lower()
            and 'back' not in d.class_name.lower()
        ],
        key=lambda d: d.bbox[0],
    )


cap = ScreenCapture()
if not cap._detect_balatro_window():
    print('FAIL: Balatro window not found')
    sys.exit(1)
det = MultiYOLODetector()

print(f'capturing for {SECONDS:.0f}s -- hover a card and hold\n')
best = None
t_end = time.time() + SECONDS
n = 0
while time.time() < t_end:
    frame = cap.capture_once()
    if frame is None:
        continue
    n += 1
    ds = det.detect_entities(frame)
    hand = entities.hand_cards(ds)
    tips = entities.descriptions(ds)
    if tips and hand:
        p, e = old_prompt(ds), old_exec(ds)
        shifted = [c for c in hand if p.index(c) != e.index(c)]
        tip_x = [t.bbox[0] for t in tips]
        hand_x = [c.bbox[0] for c in hand]
        inside = any(min(hand_x) < tx < max(hand_x) for tx in tip_x)
        print(
            f'frame {n}: hand={len(hand)} tooltip={len(tips)} @x={tip_x} '
            f'hand_x=[{min(hand_x)}..{max(hand_x)}] tooltip_inside_hand={inside} '
            f'cards_shifted={len(shifted)}'
        )
        score = (len(shifted), inside, len(hand))
        if best is None or score > best[0]:
            best = (score, frame, ds, p, e, hand, tips)
    time.sleep(INTERVAL)

print(f'\n{n} frames captured')
if not best:
    print('No frame had BOTH a hand and a tooltip. Was a card hovered?')
    sys.exit(0)

score, frame, ds, p, e, hand, tips = best
cv2.imwrite(f'{SP}/tooltip-frame.png', frame)
print(f'best frame saved -> {SP}/tooltip-frame.png')
print(
    '\nall detections   : '
    + ', '.join(
        f'{d.class_name}@{d.bbox[0]}' for d in sorted(ds, key=lambda d: d.bbox[0])
    )
)
print('old prompt index : ' + ', '.join(f'{i}:{d.class_name}' for i, d in enumerate(p)))
print('old exec index   : ' + ', '.join(f'{i}:{d.class_name}' for i, d in enumerate(e)))
print(
    'fixed hand       : '
    + ', '.join(f'{i}:{d.class_name}@{d.bbox[0]}' for i, d in enumerate(hand))
)
shifted = [c for c in hand if p.index(c) != e.index(c)]
print(
    f'\nVERDICT: {len(shifted)} of {len(hand)} real hand cards indexed differently '
    f'by the two old filters'
)
