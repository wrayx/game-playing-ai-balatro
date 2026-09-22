#!/usr/bin/env python3
"""Annotate a live frame with the index each filter assigns every detection.

  pixi run python docs/ai/findings/2026-09-22-hand-index-divergence/scripts/annotate_divergence.py \
      <input-frame.png> <output.png>
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[5] / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import cv2

from ai_balatro.core import entities
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector

LEGACY_PLAYABLE = {
    'poker_card_front',
    'joker_card',
    'tarot_card',
    'planet_card',
    'spectral_card',
}


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    detections = MultiYOLODetector().detect_entities(frame)

    prompt_side = sorted(
        [
            d
            for d in detections
            if 'card' in d.class_name.lower() and 'tooltip' not in d.class_name.lower()
        ],
        key=lambda d: d.bbox[0],
    )
    exec_side = sorted(
        [
            d
            for d in detections
            if d.class_name.lower() in LEGACY_PLAYABLE
            and 'description' not in d.class_name.lower()
            and 'back' not in d.class_name.lower()
        ],
        key=lambda d: d.bbox[0],
    )
    hand = entities.hand_cards(detections)

    for d in sorted(detections, key=lambda d: d.bbox[0]):
        x1, y1, x2, y2 = d.bbox
        p = prompt_side.index(d) if d in prompt_side else None
        e = exec_side.index(d) if d in exec_side else None

        if d in hand:
            colour = (0, 200, 0) if p == e else (0, 0, 235)
            label = f'P{p} E{e}' if p == e else f'P{p}->E{e}'
        else:
            colour = (0, 190, 235)
            label = f'P{p} {d.class_name}' if p is not None else d.class_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        ty = y1 - 6 if y1 > 20 else y2 + 16
        cv2.putText(
            frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3
        )
        cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    shifted = sum(1 for c in hand if prompt_side.index(c) != exec_side.index(c))
    banner = (
        f'{shifted}/{len(hand)} hand cards misindexed   '
        f'P = old prompt index, E = old executor index'
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        frame, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )

    cv2.imwrite(sys.argv[2], frame)
    print(f'{shifted}/{len(hand)} misindexed -> {sys.argv[2]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
