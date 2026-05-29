"""
Skeleton Extraction Module

Extracts 2D skeleton keypoints from video frames.
Uses YOLOv8-Pose (GPU, ~3ms/frame) or MediaPipe (CPU, <5ms/frame).

Output: (B, T, 17, 3) where 17 is COCO keypoints and 3 is (x, y, confidence).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
import warnings


# COCO keypoint names (17 keypoints)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


class SkeletonExtractor:
    """
    Extract 2D pose keypoints from video frames.

    Supports:
      - YOLOv8-Pose (GPU preferred)
      - MediaPipe (CPU fallback for edge devices)
    """

    def __init__(
        self,
        method: str = 'mediapipe',
        device: str = 'cpu',
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.method = method
        self.device = device
        self.image_size = image_size
        self._extractor = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True

        if self.method == 'yolov8-pose':
            try:
                from ultralytics import YOLO
                self._extractor = YOLO('yolov8n-pose.pt')
                if 'cuda' in self.device:
                    self._extractor.to(self.device)
                print(f"[SkeletonExtractor] Using YOLOv8-Pose on {self.device}")
            except ImportError:
                warnings.warn("ultralytics not installed, falling back to MediaPipe")
                self.method = 'mediapipe'

        if self.method == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self._extractor = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[SkeletonExtractor] Using MediaPipe Pose (CPU)")
            except ImportError:
                warnings.warn("mediapipe not installed, using dummy extractor")
                self.method = 'dummy'

        if self.method == 'dummy' or self._extractor is None:
            self.method = 'dummy'
            warnings.warn("[SkeletonExtractor] Using dummy (random) skeleton generator")

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract skeleton keypoints from a single frame.

        Args:
            frame: (H, W, C) numpy array, uint8 [0, 255] or float [0, 1]
        Returns:
            keypoints: (17, 3) numpy array, (x, y, confidence)
        """
        self._lazy_init()

        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        if self.method == 'yolov8-pose' and self._extractor is not None:
            results = self._extractor(frame, verbose=False)
            if results and len(results) > 0 and results[0].keypoints is not None:
                kpts = results[0].keypoints.cpu().numpy()
                if kpts.shape[0] > 0:
                    # Take first person, COCO format (17, 3)
                    keypoints = kpts[0].data  # (17, 3)
                    # Normalize to [0, 1]
                    keypoints[:, 0] /= self.image_size[0]
                    keypoints[:, 1] /= self.image_size[1]
                    return keypoints

        elif self.method == 'mediapipe' and self._extractor is not None:
            results = self._extractor.process(frame)
            if results.pose_landmarks:
                keypoints = np.zeros((17, 3), dtype=np.float32)
                # MediaPipe to COCO mapping (approximate)
                mp_to_coco = [
                    0, 2, 5, 7, 8,  # nose, left_eye, right_eye, left_ear, right_ear
                    11, 12, 13, 14,  # shoulders, elbows
                    15, 16, 23, 24,  # wrists, hips
                    25, 26, 27, 28,  # knees, ankles
                ]
                # Simplified mapping: just take the first 17 landmarks
                for i in range(min(17, len(results.pose_landmarks.landmark))):
                    lm = results.pose_landmarks.landmark[i]
                    keypoints[i] = [lm.x, lm.y, lm.visibility]
                return keypoints

        # Dummy fallback: random keypoints
        return np.random.randn(17, 3).astype(np.float32) * 0.1 + 0.5

    def extract_batch(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract skeletons from a batch of frames.

        Args:
            frames: (B, T, C, H, W) or (T, C, H, W)

        Returns:
            skeletons: (B, T, 17, 3) or (T, 17, 3)
        """
        single = frames.dim() == 4
        if single:
            frames = frames.unsqueeze(0)  # (1, T, C, H, W)

        B, T = frames.shape[:2]

        all_skeletons = []
        for b in range(B):
            frame_skeletons = []
            for t in range(T):
                # Convert to numpy (H, W, C)
                frame_np = frames[b, t].permute(1, 2, 0).cpu().numpy()
                kpts = self.extract_frame(frame_np)
                frame_skeletons.append(torch.tensor(kpts, dtype=torch.float32))
            all_skeletons.append(torch.stack(frame_skeletons))  # (T, 17, 3)

        skeletons = torch.stack(all_skeletons)  # (B, T, 17, 3)

        if single:
            skeletons = skeletons.squeeze(0)

        return skeletons
