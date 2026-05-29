"""
Le2i Multi-Modal Dataset Loader for T-JEPA

Loads the Le2i Fall Detection Dataset with synchronized:
  - Video frames (320x240 → 224x224 at 25fps)
  - Audio (16kHz resampled from AVI PCM)
  - Skeleton keypoints (extracted via YOLOv8-Pose or MediaPipe)
  - Text annotations (from templates or heuristics)

The dataset has 4 scenes: Coffee room, Home, Lecture room, Office.
Each video has ground-truth fall frame annotations.

Training: only normal clips (no falls). Falls reserved for evaluation.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
import warnings


class Le2iDataset(Dataset):
    """
    Le2i Fall Detection Dataset for T-JEPA training.

    Each sample returns:
      - ctx_frames: (T_ctx, C, H, W) context video frames
      - tgt_frames: (T_tgt, C, H, W) target video frames
      - ctx_audio: (L_ctx,) context audio waveform
      - tgt_audio: (L_tgt,) target audio waveform
      - ctx_skeleton: (T_ctx, 17, 3) context skeleton
      - tgt_skeleton: (T_tgt, 17, 3) target skeleton
      - text_condition: str description of current state
      - target_text: str description of future state (for VL-JEPA Stage 2)
    """

    def __init__(
        self,
        data_root: str,
        num_context_frames: int = 8,
        num_target_frames: int = 8,
        frame_size: int = 224,
        audio_sample_rate: int = 16000,
        audio_duration: float = 2.0,
        train: bool = True,
        normal_only: bool = True,  # T-JEPA: train only on normal
        stride: int = 4,
        scene_filter: Optional[List[str]] = None,
        skeleton_extractor=None,
        text_annotator=None,
        cache_videos: bool = False,
        classification_mode: bool = False,  # supervised classification: single window + future_fall label
        num_future_frames: int = 8,  # frames ahead to check for fall (prediction horizon)
        leak_frames: int = 4,  # allow context to see first N frames of fall onset
    ):
        super().__init__()
        self.data_root = data_root
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.total_frames = num_context_frames + num_target_frames
        self.frame_size = frame_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.train = train
        self.normal_only = normal_only
        self.stride = stride
        self.cache_videos = cache_videos
        self.classification_mode = classification_mode
        self.num_future_frames = num_future_frames
        self.leak_frames = leak_frames

        self.skeleton_extractor = skeleton_extractor
        self.text_annotator = text_annotator

        # Find all AVI files
        self.video_paths = []
        self.annotations = {}  # video_path → [(start_frame, end_frame, 'fall'/'normal')]

        self._scan_dataset(scene_filter)
        self.clips = self._build_clip_index()

        # Video cache
        self._video_cache: Dict[str, Tuple] = {}

        print(f"[Le2iDataset] Found {len(self.video_paths)} videos, "
              f"{len(self.clips)} clips (normal_only={normal_only})")

    def _scan_dataset(self, scene_filter: Optional[List[str]] = None):
        """Scan dataset directory for AVI files and ground-truth annotations."""
        patterns = ['*.avi', '*.AVI', '*.mp4', '*.MP4']
        for pattern in patterns:
            for video_path in glob.glob(os.path.join(self.data_root, '**', pattern), recursive=True):
                if scene_filter:
                    if not any(s.lower() in video_path.lower() for s in scene_filter):
                        continue
                self.video_paths.append(video_path)

                # Look for annotation file
                ann_path = self._find_annotation(video_path)
                if ann_path:
                    self.annotations[video_path] = self._parse_annotations(ann_path)

        if not self.video_paths:
            warnings.warn(f"No video files found in {self.data_root}")

    def _find_annotation(self, video_path: str) -> Optional[str]:
        """Find annotation file for a video.

        Le2i structure: Video in Videos/video (N).avi, annotation in Annotation_files/video (N).txt
        """
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]  # e.g. "video (14)"

        # Search in sibling Annotation_files / Annotations_files directories
        video_dir = os.path.dirname(video_path)
        parent_dir = os.path.dirname(video_dir)

        for ann_dir_name in ['Annotation_files', 'Annotations_files', 'annotations', 'Annotations']:
            ann_dir = os.path.join(parent_dir, ann_dir_name)
            for ext in ['.txt', '_gt.txt', '_annotation.txt', '_label.txt']:
                ann_path = os.path.join(ann_dir, base_name + ext)
                if os.path.exists(ann_path):
                    return ann_path

            # Also check inside Videos/ itself
            for ext in ['.txt', '_gt.txt', '_annotation.txt', '_label.txt']:
                ann_path = os.path.join(video_dir, base_name + ext)
                if os.path.exists(ann_path):
                    return ann_path

        return None

    def _parse_annotations(self, ann_path: str) -> List[Tuple[int, int, str]]:
        """Parse Le2i ground-truth annotation file.

        Le2i format:
          Line 1: fall_start_frame (integer)
          Line 2: fall_end_frame (integer)
          Subsequent lines: frame,label,x,y,w,h (CSV bounding boxes)
        If no fall: first two lines are "0" or the file starts directly with CSV.
        """
        annotations = []
        with open(ann_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

        if not lines:
            return annotations

        # Check if first two lines are single integers (Le2i standard format)
        try:
            line1 = lines[0].split(',')
            line2 = lines[1].split(',') if len(lines) > 1 else []

            if len(line1) == 1 and len(line2) == 1:
                # Standard format: first two lines are fall_start, fall_end
                fall_start = int(line1[0])
                fall_end = int(line2[0])
                if fall_start > 0 and fall_end > fall_start:
                    annotations.append((fall_start, fall_end, 'fall'))
                return annotations
        except (ValueError, IndexError):
            pass

        # Fallback: CSV format (start_frame,end_frame,label)
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    start_frame = int(parts[0])
                    end_frame = int(parts[1])
                    label = parts[2].strip().lower()
                    if 'fall' in label:
                        annotations.append((start_frame, end_frame, 'fall'))
                    else:
                        annotations.append((start_frame, end_frame, 'normal'))
                except (ValueError, IndexError):
                    continue

        return annotations

    def _build_clip_index(self) -> List[Dict]:
        """Build clip index from all videos, excluding falls for training."""
        clips = []
        for video_path in self.video_paths:
            # Get video info without loading
            try:
                from av import open as av_open
                with av_open(video_path) as container:
                    video_stream = container.streams.video[0]
                    total_frames = video_stream.frames or 0
                    fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
            except Exception:
                total_frames = 1000
                fps = 25.0

            if total_frames < self.total_frames:
                continue

            # Get fall regions
            fall_regions = []
            if video_path in self.annotations:
                for start, end, label in self.annotations[video_path]:
                    if label == 'fall':
                        fall_regions.append((start, end))

            if self.classification_mode:
                # Classification: single context window + future_fall label
                # Each clip = context frames [start, start+num_context_frames)
                # Label = 1 if fall occurs in [start+num_context_frames, start+num_context_frames+num_future_frames)
                # CRITICAL: context must NOT contain any fall frames (data leakage prevention).
                # We predict the fall BEFORE it happens, not detect it while it's happening.
                max_start = total_frames - self.num_context_frames
                n_skipped_leak = 0
                n_skipped_partial = 0
                for start_frame in range(0, max_start, self.stride):
                    ctx_end = start_frame + self.num_context_frames
                    future_end = ctx_end + self.num_future_frames

                    # Does future window overlap with any fall?
                    future_fall = any(
                        not (future_end < fall_start or ctx_end >= fall_end)
                        for fall_start, fall_end in fall_regions
                    )

                    if future_fall:
                        # Allow context to show at most leak_frames of fall onset.
                        # Context must end within [fall_start, fall_start+leak_frames].
                        valid = False
                        for fall_start, fall_end in fall_regions:
                            if not (future_end < fall_start or ctx_end >= fall_end):
                                if ctx_end >= fall_start and ctx_end <= fall_start + self.leak_frames:
                                    valid = True
                                    break
                        if not valid:
                            n_skipped_leak += 1
                            continue

                    clips.append({
                        'video_path': video_path,
                        'start_frame': start_frame,
                        'future_fall': future_fall,
                        'fps': fps,
                    })

                if n_skipped_leak > 0:
                    vid_name = os.path.basename(video_path)
                    print(f"  [{vid_name}] Skipped {n_skipped_leak} leaky pre-fall samples (context contained fall frames)")
            else:
                # JEPA mode: context + target window pairs
                for start_frame in range(0, total_frames - self.total_frames, self.stride):
                    clip_end = start_frame + self.total_frames
                    has_fall = any(
                        not (clip_end < fall_start or start_frame > fall_end)
                        for fall_start, fall_end in fall_regions
                    )

                    if self.normal_only and has_fall:
                        continue

                    clips.append({
                        'video_path': video_path,
                        'start_frame': start_frame,
                        'has_fall': has_fall,
                        'fps': fps,
                    })

        if self.classification_mode:
            n_fall = sum(1 for c in clips if c['future_fall'])
            n_normal = len(clips) - n_fall
            print(f"[Le2iDataset] Classification mode: {n_normal} normal, {n_fall} pre-fall clips")

        return clips

    def _load_video(self, video_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load video frames and audio from a video file."""
        if self.cache_videos and video_path in self._video_cache:
            return self._video_cache[video_path]

        try:
            from av import open as av_open

            frames = []
            audio_samples = []

            with av_open(video_path) as container:
                # Video
                video_stream = container.streams.video[0]
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                    frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

                # Audio
                try:
                    audio_stream = container.streams.audio[0]
                    for frame in container.decode(audio=0):
                        samples = frame.to_ndarray()  # (channels, samples)
                        audio_samples.append(torch.from_numpy(samples.astype(np.float32)))
                except Exception:
                    pass

            video_tensor = torch.stack(frames) if frames else torch.zeros(self.total_frames, 3, self.frame_size, self.frame_size)

            if audio_samples:
                audio_tensor = torch.cat(audio_samples, dim=-1).mean(dim=0)  # mono
            else:
                audio_tensor = None

            if self.cache_videos:
                self._video_cache[video_path] = (video_tensor, audio_tensor)

            return video_tensor, audio_tensor

        except Exception as e:
            warnings.warn(f"Failed to load {video_path}: {e}")
            return torch.zeros(self.total_frames, 3, self.frame_size, self.frame_size), None

    def _resample_frames(self, frames: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Resample frames to target count using uniform sampling."""
        T = frames.shape[0]
        if T == target_frames:
            return frames
        indices = torch.linspace(0, T - 1, target_frames).long()
        return frames[indices]

    def _resample_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if audio is None:
            target_samples = int(self.audio_duration * self.audio_sample_rate)
            return torch.zeros(target_samples)

        if orig_sr == self.audio_sample_rate:
            return audio

        from torch.nn.functional import interpolate
        audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        target_len = int(len(audio.squeeze()) * self.audio_sample_rate / orig_sr)
        audio = interpolate(audio, size=target_len, mode='linear', align_corners=False)
        return audio.squeeze()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict:
        clip = self.clips[idx]

        # Load video
        all_frames, all_audio = self._load_video(clip['video_path'])

        if self.classification_mode:
            return self._get_classification_item(clip, all_frames, all_audio)

        start = clip['start_frame']
        end = start + self.total_frames

        if end > all_frames.shape[0]:
            # Pad with last frame
            pad = all_frames[-1:].repeat(end - all_frames.shape[0], 1, 1, 1)
            all_frames = torch.cat([all_frames, pad], dim=0)

        clip_frames = all_frames[start:end]

        # Resize to target size
        if clip_frames.shape[-1] != self.frame_size:
            from torch.nn.functional import interpolate
            T, C, H, W = clip_frames.shape
            clip_frames = clip_frames.reshape(T * C, H, W).unsqueeze(0)  # (1, T*C, H, W)
            clip_frames = interpolate(
                clip_frames,
                size=(self.frame_size, self.frame_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)  # (T*C, H', W')
            clip_frames = clip_frames.reshape(T, C, self.frame_size, self.frame_size)

        # Split into context and target
        ctx_frames = clip_frames[:self.num_context_frames]
        tgt_frames = clip_frames[self.num_context_frames:]

        # Audio
        if all_audio is not None:
            # Extract audio segment roughly aligned with video
            audio_per_frame = len(all_audio) / max(all_frames.shape[0], 1)
            audio_start = int(start * audio_per_frame)
            audio_end = int(end * audio_per_frame)
            audio_segment = all_audio[audio_start:audio_end]

            audio_mid = len(audio_segment) // 2
            ctx_audio = self._resample_audio(audio_segment[:audio_mid], 48000)
            tgt_audio = self._resample_audio(audio_segment[audio_mid:], 48000)
        else:
            target_samples = int(self.audio_duration * self.audio_sample_rate)
            ctx_audio = torch.zeros(target_samples)
            tgt_audio = torch.zeros(target_samples)

        # Skeleton
        if self.skeleton_extractor is not None:
            ctx_skeleton = self.skeleton_extractor.extract_batch(ctx_frames)
            tgt_skeleton = self.skeleton_extractor.extract_batch(tgt_frames)
        else:
            ctx_skeleton = torch.zeros(self.num_context_frames, 17, 3)
            tgt_skeleton = torch.zeros(self.num_target_frames, 17, 3)

        # Text annotations
        text_condition = "老人在活动"
        target_text = "老人继续活动"
        if self.text_annotator is not None:
            text_condition = self.text_annotator.get_annotation(
                skeleton=ctx_skeleton
            )
            target_text = self.text_annotator.generate_future_description(
                current_activity="standing",  # Could be improved with activity classifier
                is_fall=clip['has_fall'],
            )

        return {
            'ctx_frames': ctx_frames,
            'tgt_frames': tgt_frames,
            'ctx_audio': ctx_audio,
            'tgt_audio': tgt_audio,
            'ctx_skeleton': ctx_skeleton,
            'tgt_skeleton': tgt_skeleton,
            'text_condition': text_condition,
            'target_text': target_text,
            'has_fall': clip['has_fall'],
            'video_path': clip['video_path'],
        }

    def _get_classification_item(self, clip: Dict, all_frames: torch.Tensor, all_audio: Optional[torch.Tensor]) -> Dict:
        """Get a single-window item for supervised classification."""
        start = clip['start_frame']
        end = start + self.num_context_frames

        if end > all_frames.shape[0]:
            pad = all_frames[-1:].repeat(end - all_frames.shape[0], 1, 1, 1)
            all_frames = torch.cat([all_frames, pad], dim=0)

        ctx_frames = all_frames[start:end]

        # Resize frames
        if ctx_frames.shape[-1] != self.frame_size:
            from torch.nn.functional import interpolate
            T, C, H, W = ctx_frames.shape
            ctx_frames = ctx_frames.reshape(T * C, H, W).unsqueeze(0)
            ctx_frames = interpolate(
                ctx_frames,
                size=(self.frame_size, self.frame_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            ctx_frames = ctx_frames.reshape(T, C, self.frame_size, self.frame_size)

        # Audio: extract segment aligned with context frames
        if all_audio is not None:
            audio_per_frame = len(all_audio) / max(all_frames.shape[0], 1)
            audio_start = int(start * audio_per_frame)
            audio_end = int(end * audio_per_frame)
            ctx_audio = all_audio[audio_start:audio_end]
            # Resample to target sample rate and pad/trim to fixed length
            ctx_audio = self._resample_audio(ctx_audio, 48000)
        else:
            target_samples = int(self.audio_duration * self.audio_sample_rate)
            ctx_audio = torch.zeros(target_samples)

        return {
            'frames': ctx_frames,
            'audio': ctx_audio,
            'label': torch.tensor(float(clip['future_fall']), dtype=torch.float32),
            'video_path': clip['video_path'],
        }


def create_le2i_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Factory function to create a Le2i DataLoader."""
    dataset = Le2iDataset(data_root=data_root, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset_kwargs.get('train', True),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
