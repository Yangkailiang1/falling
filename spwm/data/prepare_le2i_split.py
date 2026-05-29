"""
Generate video-level train/test split for Le2i balanced dataset.

Scene-stratified 80/20 split with round-robin leak level assignment (0-16).
Each video produces exactly one sample — one 16-frame context window per video.

Usage:
    python3 -m spwm.data.prepare_le2i_split --data_root Le2i_processed --out le2i_split.json --seed 42
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def parse_annotations(ann_path: str) -> List[Tuple[int, int, str]]:
    """Parse Le2i annotation file. Same logic as PreprocessedLe2iDataset._parse_annotations."""
    with open(ann_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    if not lines:
        return []

    try:
        line1 = lines[0].split(',')
        line2 = lines[1].split(',') if len(lines) > 1 else []
        if len(line1) == 1 and len(line2) == 1:
            fall_start = int(line1[0])
            fall_end = int(line2[0])
            if fall_start > 0 and fall_end > fall_start:
                return [(fall_start, fall_end, 'fall')]
            return []
    except (ValueError, IndexError):
        pass

    annotations = []
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                start = int(parts[0])
                end = int(parts[1])
                label = parts[2].strip().lower()
                if 'fall' in label:
                    annotations.append((start, end, 'fall'))
            except (ValueError, IndexError):
                continue
    return annotations


def extract_scene(video_dir_name: str) -> str:
    """Extract scene key from video directory name using known scene prefixes.

    The preprocessed directory names are flattened versions of the original paths:
      Coffee_room_01/Coffee_room_01/Videos/video (1).avi -> 'Coffee_room_01_Coffee_room_01_Videos_video (1)'
      Office/Office/video (1).avi -> 'Office_Office_video (1)'
    """
    for scene in ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02',
                  'Lecture_room', 'Office']:
        if video_dir_name.startswith(scene + '_'):
            return scene
    # Fallback: use whatever prefix we can extract
    return video_dir_name.split('_')[0] if '_' in video_dir_name else video_dir_name


def discover_videos(data_root: str) -> List[Dict]:
    """Scan data_root for all video directories with meta.pt."""
    videos = []
    for root, dirs, files in os.walk(data_root):
        if 'meta.pt' in files:
            vd = root
            meta = {}
            try:
                import torch
                meta = torch.load(os.path.join(vd, 'meta.pt'), map_location='cpu', weights_only=True)
            except Exception:
                pass

            n_frames = meta.get('n_frames', 0)
            fps = meta.get('fps', 25.0)

            ann_path = os.path.join(vd, 'annotation.txt')
            fall_annotations = []
            if os.path.exists(ann_path):
                fall_annotations = parse_annotations(ann_path)

            is_fall = len(fall_annotations) > 0
            fall_start = fall_annotations[0][0] if is_fall else 0
            fall_end = fall_annotations[0][1] if is_fall else 0
            scene = extract_scene(os.path.basename(vd))

            videos.append({
                'dir': vd,
                'name': os.path.basename(vd),
                'scene': scene,
                'is_fall': is_fall,
                'fall_start': fall_start,
                'fall_end': fall_end,
                'n_frames': n_frames,
                'fps': fps,
            })

    return videos


def stratified_split(items: List[Dict], train_frac: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Scene-stratified random split."""
    rng = random.Random(seed)
    by_scene = defaultdict(list)
    for item in items:
        by_scene[item['scene']].append(item)

    train, test = [], []
    for scene, scene_items in sorted(by_scene.items()):
        rng.shuffle(scene_items)
        n_train = max(1, int(len(scene_items) * train_frac)) if len(scene_items) > 1 else len(scene_items)
        n_train = min(n_train, len(scene_items) - (1 if len(scene_items) > 1 else 0))
        # Ensure at least 1 test sample if scene has >= 2 videos
        if len(scene_items) >= 2 and n_train == len(scene_items):
            n_train = len(scene_items) - 1
        train.extend(scene_items[:n_train])
        test.extend(scene_items[n_train:])

    return train, test


def assign_leak_levels(fall_videos: List[Dict], seed: int) -> None:
    """Assign leak levels 0-16 in round-robin fashion (balanced)."""
    rng = random.Random(seed)
    rng.shuffle(fall_videos)
    for i, v in enumerate(fall_videos):
        v['leak'] = i % 17


def compute_context_window(v: Dict, num_context_frames: int = 16) -> int:
    """Compute context start frame for a fall video at its assigned leak level.

    ctx_end = fall_start + leak
    ctx_start = ctx_end - num_context_frames

    For leak=k, context sees exactly k frames of the fall onset.
    """
    ctx_start = v['fall_start'] + v['leak'] - num_context_frames
    return max(0, ctx_start)


def main():
    parser = argparse.ArgumentParser(description='Generate balanced Le2i train/test split')
    parser.add_argument('--data_root', default='Le2i_processed', help='Path to preprocessed dataset')
    parser.add_argument('--out', default='le2i_split.json', help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_context_frames', type=int, default=16)
    parser.add_argument('--num_future_frames', type=int, default=8)
    args = parser.parse_args()

    # Discover all videos
    all_videos = discover_videos(args.data_root)
    print(f"Found {len(all_videos)} video directories")

    # Separate fall and non-fall
    fall_videos = [v for v in all_videos if v['is_fall']]
    nonfall_videos = [v for v in all_videos if not v['is_fall']]
    print(f"  Fall videos: {len(fall_videos)}")
    print(f"  Non-fall videos: {len(nonfall_videos)}")

    # Validate: all fall videos can support any leak level 0-16
    for v in fall_videos:
        if v['fall_start'] < args.num_context_frames:
            max_leak = v['fall_start']
            print(f"  WARNING: {v['name']}: fall_start={v['fall_start']}, "
                  f"cannot support leak levels > {max_leak}")

    # Scene-stratified split
    fall_train, fall_test = stratified_split(fall_videos, 0.8, args.seed)
    nonfall_train, nonfall_test = stratified_split(nonfall_videos, 0.8, args.seed)
    print(f"Fall: {len(fall_train)} train / {len(fall_test)} test")
    print(f"Non-fall: {len(nonfall_train)} train / {len(nonfall_test)} test")

    # Assign leak levels to fall videos (round-robin, balanced)
    assign_leak_levels(fall_train, seed=args.seed)
    assign_leak_levels(fall_test, seed=args.seed + 1)

    # Compute context windows
    for v in fall_train + fall_test:
        v['ctx_start'] = compute_context_window(v, args.num_context_frames)

    # For non-fall videos, pick a random 24-frame window (16 ctx + 8 future buffer)
    # Actually just pick any position that has at least 24 frames available
    rng = random.Random(args.seed + 2)
    for v in nonfall_train + nonfall_test:
        if v['n_frames'] >= args.num_context_frames + args.num_future_frames:
            max_start = v['n_frames'] - args.num_context_frames - args.num_future_frames
            v['ctx_start'] = rng.randint(0, max_start)
        elif v['n_frames'] >= args.num_context_frames:
            v['ctx_start'] = rng.randint(0, v['n_frames'] - args.num_context_frames)
        else:
            v['ctx_start'] = 0

    # Build output
    output = {
        'metadata': {
            'created': '2026-05-25',
            'seed': args.seed,
            'num_context_frames': args.num_context_frames,
            'num_future_frames': args.num_future_frames,
            'n_videos_total': len(all_videos),
            'n_fall_train': len(fall_train),
            'n_fall_test': len(fall_test),
            'n_nonfall_train': len(nonfall_train),
            'n_nonfall_test': len(nonfall_test),
        },
        'videos': {}
    }

    for v in all_videos:
        split = 'train' if v in fall_train or v in nonfall_train else 'test'
        entry = {
            'split': split,
            'is_fall': v['is_fall'],
            'scene': v['scene'],
            'n_frames': v['n_frames'],
            'ctx_start': v.get('ctx_start', 0),
        }
        if v['is_fall']:
            entry['fall_start'] = v['fall_start']
            entry['fall_end'] = v['fall_end']
            entry['leak'] = v['leak']
        output['videos'][v['name']] = entry

    # Write JSON
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Split written to {args.out}")

    # Print leak distribution for train/test
    for split_name, videos in [('train', fall_train), ('test', fall_test)]:
        leak_dist = defaultdict(list)
        for v in videos:
            leak_dist[v['leak']].append(v['name'])
        print(f"\n{split_name} leak distribution:")
        for leak in range(17):
            names = leak_dist.get(leak, [])
            print(f"  leak={leak:2d}: {len(names)} videos")


if __name__ == '__main__':
    main()
