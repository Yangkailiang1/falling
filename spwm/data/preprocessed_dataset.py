"""
Pre-processed Le2i Dataset: reads from pre-decoded .pt files.

Each video directory under data_root contains:
  frames.pt     - (T, 3, 224, 224) float32, decoded and resized frames
  audio.pt      - (L,) float32, audio waveform
  meta.pt       - dict with fps, n_frames, audio_sr
  annotation.txt - (optional) ground-truth annotations
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple


class PreprocessedLe2iDataset(Dataset):
    """Fast Le2i dataset reading from pre-processed .pt files."""

    def __init__(
        self,
        data_root: str,
        num_context_frames: int = 16,
        num_future_frames: int = 8,
        stride: int = 8,
        classification_mode: bool = True,
        frame_size: int = 224,
        leak_frames: int = 16,  # max fall frames allowed in context (0=pure prediction, 16=detection)
    ):
        super().__init__()
        self.data_root = data_root
        self.num_context_frames = num_context_frames
        self.num_future_frames = num_future_frames
        self.stride = stride
        self.classification_mode = classification_mode
        self.frame_size = frame_size
        self.leak_frames = leak_frames

        # Find all video dirs (those containing meta.pt)
        self.video_dirs = []
        for vd in sorted(glob.glob(os.path.join(data_root, '**', 'meta.pt'), recursive=True)):
            self.video_dirs.append(os.path.dirname(vd))

        # Parse annotations and build clip index
        self.clips = self._build_clip_index()

        n_fall = sum(1 for c in self.clips if c['future_fall'])
        n_normal = len(self.clips) - n_fall
        print(f"[PreprocessedLe2i] {len(self.video_dirs)} videos, "
              f"{len(self.clips)} clips ({n_normal} normal, {n_fall} pre-fall)")

    def _parse_annotations(self, ann_path: str) -> List[Tuple[int, int, str]]:
        """Parse Le2i annotation file. Same logic as original Le2iDataset."""
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

    def _build_clip_index(self) -> List[Dict]:
        """Build clip index. For positive samples, records n_fall_frames seen by context."""
        clips = []
        for vd in self.video_dirs:
            meta_path = os.path.join(vd, 'meta.pt')
            meta = torch.load(meta_path, map_location='cpu', weights_only=True)
            n_frames = meta['n_frames']
            fps = meta.get('fps', 25.0)

            if n_frames < self.num_context_frames:
                continue

            # Get fall regions
            fall_regions = []
            ann_path = os.path.join(vd, 'annotation.txt')
            if os.path.exists(ann_path):
                fall_regions = self._parse_annotations(ann_path)

            max_start = n_frames - self.num_context_frames
            for start_frame in range(0, max_start, self.stride):
                ctx_end = start_frame + self.num_context_frames
                future_end = ctx_end + self.num_future_frames

                # Does future window overlap with any fall?
                future_fall = False
                n_fall_frames = 0
                for fs, fe, _ in fall_regions:
                    if not (future_end <= fs or ctx_end >= fe):
                        # Future overlaps with this fall
                        # Check if context leak is within allowed range
                        if ctx_end >= fs and ctx_end <= fs + self.leak_frames:
                            future_fall = True
                            n_fall_frames = max(0, min(ctx_end, fe) - max(start_frame, fs))
                            break

                if future_fall:
                    # Double-check: if context starts after fall ends, skip
                    if start_frame >= fe:
                        future_fall = False
                        n_fall_frames = 0

                clips.append({
                    'video_dir': vd,
                    'start_frame': start_frame,
                    'future_fall': future_fall,
                    'n_fall_frames': n_fall_frames,  # how many fall frames in context
                    'fps': fps,
                })

        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict:
        clip = self.clips[idx]
        vd = clip['video_dir']
        start = clip['start_frame']
        end = start + self.num_context_frames

        # Load frames (lazy, cached by OS page cache)
        frames = torch.load(os.path.join(vd, 'frames.pt'), map_location='cpu', weights_only=True)
        ctx_frames = frames[start:end].clone()

        # Load audio
        audio_path = os.path.join(vd, 'audio.pt')
        if os.path.exists(audio_path):
            audio = torch.load(audio_path, map_location='cpu', weights_only=True)
            meta = torch.load(os.path.join(vd, 'meta.pt'), map_location='cpu', weights_only=True)
            audio_sr = meta.get('audio_sr', 48000)
            n_audio_samples = len(audio)
            # Align audio with video frames
            audio_per_frame = n_audio_samples / max(frames.shape[0], 1)
            a_start = int(start * audio_per_frame)
            a_end = int(end * audio_per_frame)
            ctx_audio = audio[a_start:a_end]
            # Resample to 48000 if needed
            ctx_audio = self._resample_audio(ctx_audio, audio_sr)
        else:
            ctx_audio = torch.zeros(48000)

        return {
            'frames': ctx_frames,
            'audio': ctx_audio,
            'label': torch.tensor(float(clip['future_fall']), dtype=torch.float32),
            'n_fall_frames': clip.get('n_fall_frames', 0),
            'video_path': vd,
        }

    def _resample_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to 16kHz or pad/trim to match WavJEPA's expected length."""
        target_sr = 16000
        target_len = 48000  # WavJEPA expects 48000 samples
        if audio.numel() <= 1:
            return torch.zeros(target_len)
        # Simple resample via linear interpolation
        if orig_sr != target_sr:
            from torch.nn.functional import interpolate
            a = audio.unsqueeze(0).unsqueeze(0)
            new_len = int(len(audio) * target_sr / orig_sr)
            a = interpolate(a, size=new_len, mode='linear', align_corners=False)
            audio = a.squeeze()
        # Pad or trim
        if audio.shape[0] < target_len:
            audio = torch.cat([audio, torch.zeros(target_len - audio.shape[0])])
        elif audio.shape[0] > target_len:
            audio = audio[:target_len]
        return audio
