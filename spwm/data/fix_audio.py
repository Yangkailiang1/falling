"""
Fix audio.pt files in Le2i_processed/ by re-extracting real audio from raw AVI files.

The original preprocess_le2i.py had a bug: decode(video=0) exhausted the container
before decode(audio=0), so all audio.pt ended up as torch.zeros(1) placeholders.

This script maps each processed directory back to its raw AVI, re-opens the container
separately for audio, and overwrites audio.pt with real audio data.

Usage:
    python3 -m spwm.data.fix_audio --data_root Le2i --processed_root Le2i_processed
"""

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict


def build_avi_index(data_root: str) -> dict:
    """Build mapping: video_dir_name -> raw_avi_path.

    The preprocessed dir names are the relative path with / replaced by _ and .avi stripped.
    Example: 'Coffee_room_01_Coffee_room_01_Videos_video (1)' ->
             'Coffee_room_01/Coffee_room_01/Videos/video (1).avi'
    """
    avi_index = {}
    patterns = ['*.avi', '*.AVI', '*.mp4', '*.MP4']
    for pat in patterns:
        for avi_path in glob.glob(os.path.join(data_root, '**', pat), recursive=True):
            rel = os.path.relpath(avi_path, data_root)
            vid_name = rel.replace('/', '_').replace('\\', '_')
            # Strip extension
            for ext in ['.avi', '.AVI', '.mp4', '.MP4']:
                if vid_name.endswith(ext):
                    vid_name = vid_name[:-len(ext)]
                    break
            avi_index[vid_name] = avi_path
    return avi_index


def extract_audio(avi_path: str) -> tuple:
    """Extract audio from AVI file using PyAV.

    Uses demux() instead of decode() so corrupt packets can be skipped
    individually (some Coffee_room mp3 streams have a few bad packets).
    Returns (audio_tensor, sample_rate).
    """
    from av import open as av_open

    audio_samples = []

    with av_open(avi_path) as container:
        try:
            audio_stream = container.streams.audio[0]
            sample_rate = audio_stream.rate if audio_stream.rate else 44100
            codec = audio_stream.codec_context
        except (IndexError, AttributeError):
            return torch.zeros(1), 44100

        # Use demux + per-packet try/except to skip corrupt packets
        for packet in container.demux(audio_stream):
            try:
                for frame in codec.decode(packet):
                    samples = frame.to_ndarray()
                    arr = np.array(samples, dtype=np.float32)
                    if frame.format.name.startswith('s16') or frame.format.name.startswith('pcm_s16'):
                        arr = arr / 32768.0
                    audio_samples.append(torch.from_numpy(arr))
            except Exception:
                pass  # skip corrupt packets

    if audio_samples:
        if audio_samples[0].dim() == 1:
            audio_tensor = torch.cat(audio_samples, dim=0)
        else:
            audio_tensor = torch.cat(audio_samples, dim=1)
            audio_tensor = audio_tensor.mean(dim=0)
    else:
        audio_tensor = torch.zeros(1)

    return audio_tensor, sample_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='Le2i')
    p.add_argument('--processed_root', default='Le2i_processed')
    args = p.parse_args()

    # Build AVI index
    print("Building AVI index...")
    avi_index = build_avi_index(args.data_root)
    print(f"  {len(avi_index)} AVI files indexed")

    # Find all processed video dirs
    processed_dirs = sorted(glob.glob(os.path.join(args.processed_root, '**', 'meta.pt'), recursive=True))
    processed_dirs = [os.path.dirname(mp) for mp in processed_dirs]
    print(f"  {len(processed_dirs)} processed directories")

    # Match and fix
    n_fixed = 0
    n_skipped = 0
    n_missing = 0

    for vd in tqdm(processed_dirs, desc="Fixing audio"):
        name = os.path.basename(vd)
        audio_path = os.path.join(vd, 'audio.pt')
        current = torch.load(audio_path, map_location='cpu', weights_only=True)

        # Skip if already has real audio (shape > 1)
        if current.numel() > 1:
            n_skipped += 1
            continue

        # Find matching AVI
        avi_path = avi_index.get(name)
        if avi_path is None:
            # Try fuzzy match: the dir name might have extra prefixes
            # e.g., 'Lecture_room_Lecture room_video (1)' vs 'Lecture_room/Lecture room/video (1).avi'
            # The AVI index key strips extension: 'Lecture_room_Lecture room_video (1)'
            n_missing += 1
            continue

        try:
            audio_tensor, sample_rate = extract_audio(avi_path)
            torch.save(audio_tensor, audio_path)

            # Update meta.pt with correct audio_sr
            meta_path = os.path.join(vd, 'meta.pt')
            meta = torch.load(meta_path, map_location='cpu', weights_only=True)
            meta['audio_sr'] = sample_rate
            torch.save(meta, meta_path)

            n_fixed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    print(f"\nDone: {n_fixed} fixed, {n_skipped} already ok, {n_missing} AVI not found")


if __name__ == '__main__':
    main()
