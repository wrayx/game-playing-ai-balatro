#!/usr/bin/env python3
"""Report which entity classes co-occur in real frames, and what the pre-fix
hand-card filters got wrong.

Measures three separate things, which are easy to conflate:

  1. Alignment  - did the old prompt filter and the old executor filter assign
                  a REAL hand card different indices? That is the only failure
                  that makes a selected index click the wrong card.
  2. Padding    - how many non-playable rows did the old prompt filter show the
                  LLM as hand cards, and by how much did it inflate hand size?
  3. Jokers     - how many joker detections were misfiled as hand cards, which
                  is what left the jokers list permanently empty.

Caveat: dataset frames are training captures, so no hover sweep is in progress
and card_description rarely co-occurs with a hand. This cannot settle whether a
live tooltip shifts indices; only a capture during a sweep can.

Needs the model and dataset submodules, but neither the game nor screen-recording
permissions, so it runs headless.

Examples:
  pixi run python cli/diagnose-entity-classes.py
  pixi run python cli/diagnose-entity-classes.py --limit 40 --verbose
  pixi run python cli/diagnose-entity-classes.py --targets out_00104.jpg out_00166.jpg
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence

_cli_dir = Path(__file__).parent
_src_dir = _cli_dir.parent / 'src'

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import cv2

from ai_balatro.core import entities
from ai_balatro.core.detection import Detection
from ai_balatro.core.multi_yolo_detector import MultiYOLODetector

DEFAULT_IMAGES_DIR = (
    'data/datasets/games-balatro-2024-entities-detection/data/train/yolo/images'
)


# The pre-fix executor whitelist, from CardPositionDetector._is_playable_card.
LEGACY_PLAYABLE = {
    'poker_card_front',
    'joker_card',
    'tarot_card',
    'planet_card',
    'spectral_card',
}


def legacy_prompt_cards(detections: Sequence[Detection]) -> List[Detection]:
    """Reproduce the pre-fix prompt filter from _build_base_state.

    Note the 'tooltip' guard never fired: the description classes are named
    card_description and poker_card_description, so all ten classes passed.
    """
    matched = [
        d
        for d in detections
        if 'card' in d.class_name.lower() and 'tooltip' not in d.class_name.lower()
    ]
    return sorted(matched, key=lambda d: d.bbox[0])


def legacy_executor_cards(detections: Sequence[Detection]) -> List[Detection]:
    """Reproduce the pre-fix executor filter, which decided what got clicked."""
    matched = [
        d
        for d in detections
        if d.class_name.lower() in LEGACY_PLAYABLE
        and 'description' not in d.class_name.lower()
        and 'back' not in d.class_name.lower()
    ]
    return sorted(matched, key=lambda d: d.bbox[0])


def describe(detections: Sequence[Detection]) -> str:
    """Render detections as class names with their left edge."""
    ordered = sorted(detections, key=lambda d: d.bbox[0])
    return ', '.join(f'{d.class_name}@{d.bbox[0]}' for d in ordered) or '(none)'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', default=DEFAULT_IMAGES_DIR)
    parser.add_argument('--targets', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f'Images directory not found: {images_dir}')
        print('Run: git lfs install && git submodule update --init')
        return 1

    if args.targets:
        paths = [images_dir / name for name in args.targets]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f'Missing target images: {", ".join(str(p) for p in missing)}')
            return 1
    else:
        paths = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )[: args.limit]

    if not paths:
        print(f'No images found in {images_dir}')
        return 1

    detector = MultiYOLODetector()
    if not detector.is_model_available('entities'):
        print('Entities model unavailable. Run: git submodule update --init')
        return 1

    class_counts: Counter = Counter()
    frames_with_class: Counter = Counter()
    misaligned = 0
    padding = 0
    jokers_misfiled = 0
    frames_with_hand = 0
    inflation: List = []
    analysed = 0
    unknown_seen: set = set()

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f'Could not read {path.name}, skipping')
            continue

        detections = detector.detect_entities(image, args.confidence)
        analysed += 1

        class_counts.update(d.class_name for d in detections)
        frames_with_class.update({d.class_name for d in detections})
        unknown_seen |= entities.unknown_classes(detections)

        prompt_side = legacy_prompt_cards(detections)
        exec_side = legacy_executor_cards(detections)
        current = entities.hand_cards(detections)

        # A real hand card is misaligned only if the two old filters gave it a
        # different position. Padding beyond the hand is harmless for indexing.
        shifted = [c for c in current if prompt_side.index(c) != exec_side.index(c)]
        if shifted:
            misaligned += 1
        padding += len(prompt_side) - len(current)
        jokers_misfiled += sum(1 for d in detections if entities.is_joker(d))
        if current:
            frames_with_hand += 1
            inflation.append((path.name, len(prompt_side), len(current)))

        if args.verbose or shifted:
            flag = 'MISALIGNED' if shifted else 'aligned'
            print(f'\n{path.name} [{flag}] {len(detections)} detections')
            print(f'  all        : {describe(detections)}')
            print(f'  old prompt : {len(prompt_side):>2} -> {describe(prompt_side)}')
            print(f'  old exec   : {len(exec_side):>2} -> {describe(exec_side)}')
            print(f'  fixed hand : {len(current):>2} -> {describe(current)}')
            if shifted:
                print(f'  {len(shifted)} real hand card(s) indexed differently')

    if not analysed:
        print('No images could be analysed')
        return 1

    print(f'\n{"=" * 70}')
    print(f'Frames analysed            : {analysed}')
    print(f'Frames holding a hand      : {frames_with_hand}')
    print(
        f'Frames MISALIGNED          : {misaligned}/{frames_with_hand or 1}'
        '   (old prompt vs old executor, real hand cards only)'
    )
    print(f'Phantom rows shown as hand : {padding}')
    print(
        f'Jokers misfiled as hand    : {jokers_misfiled}  (jokers list was always empty)'
    )

    if inflation:
        print('\nHand size the LLM was told vs actual:')
        for name, told, actual in inflation:
            mark = '' if told == actual else '  <-- inflated'
            print(f'  {name}: told {told:>2}, actually {actual:>2}{mark}')

    print(f'\n{"class":<28} {"detections":>10} {"frames":>8}')
    for name, count in class_counts.most_common():
        print(f'{name:<28} {count:>10} {frames_with_class[name]:>8}')

    if unknown_seen:
        print(f'\nClasses unknown to the taxonomy: {", ".join(sorted(unknown_seen))}')

    print('\nVerdict:')
    print(
        '  alignment: '
        + (
            f'{misaligned} frame(s) would click the wrong card'
            if misaligned
            else 'the two old filters agreed on every real hand card here'
        )
    )
    print(
        '  padding  : '
        + (
            f'the LLM was shown {padding} phantom cards it could not play'
            if padding
            else 'no phantom cards'
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
