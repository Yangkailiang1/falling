"""
Text Annotation Module

Generates text descriptions for video clips, either from:
  1. Pre-existing annotations (dataset ground truth)
  2. Heuristic rules based on skeleton state (TC-JEPA fallback)
  3. Simple templates for known activities

Used for TC-JEPA text conditioning and VL-JEPA text alignment training.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


# Template annotations for common elderly activities
ACTIVITY_TEMPLATES = {
    'walking': [
        "老人正在缓慢行走",
        "老人在房间里走动",
        "老人慢慢走过房间",
        "老人拄着拐杖行走",
    ],
    'standing': [
        "老人站立在房间中央",
        "老人靠着墙站立",
        "老人扶着桌子站立",
        "老人静静站立",
    ],
    'sitting': [
        "老人坐在椅子上",
        "老人坐在沙发上",
        "老人坐在床边",
        "老人坐着休息",
    ],
    'lying': [
        "老人躺在床上",
        "老人躺在沙发上休息",
        "老人侧卧在床上",
    ],
    'falling': [
        "老人向前摔倒",
        "老人向后倒下",
        "老人失去平衡正在跌倒",
        "老人从椅子上滑落",
        "老人摔倒在地上",
    ],
    'bending': [
        "老人弯腰捡东西",
        "老人蹲下",
        "老人俯身",
    ],
    'transition': [
        "老人从站立转为行走",
        "老人准备坐下",
        "老人从椅子上站起来",
        "老人转身",
    ],
}


class TextAnnotator:
    """
    Generate text annotations for video clips.

    Supports:
      1. Dataset annotations (from file)
      2. Template-based (from activity label)
      3. Skeleton-based heuristic (fallback)
    """

    def __init__(self, use_chinese: bool = True, default_activity: str = 'standing'):
        self.use_chinese = use_chinese
        self.default_activity = default_activity
        self._annotations_cache: Dict[str, str] = {}

    def load_annotations(self, annotation_file: str):
        """
        Load per-clip annotations from file.

        Expected format (JSON):
        {
            "clip_001": "老人缓慢行走",
            "clip_002": "老人从椅子上站起来然后摔倒",
            ...
        }
        """
        import json
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self._annotations_cache = json.load(f)
        print(f"[TextAnnotator] Loaded {len(self._annotations_cache)} annotations")

    def from_activity(self, activity: str) -> str:
        """
        Generate text from activity label using templates.

        Args:
            activity: one of 'walking', 'standing', 'sitting', 'lying',
                     'falling', 'bending', 'transition'
        Returns:
            text description string
        """
        import random
        templates = ACTIVITY_TEMPLATES.get(activity, ACTIVITY_TEMPLATES['standing'])
        return random.choice(templates)

    def from_skeleton(self, skeleton: torch.Tensor) -> str:
        """
        Generate text description from skeleton keypoints (heuristic).

        Simple rules based on keypoint positions and velocities.

        Args:
            skeleton: (T, 17, 3) skeleton sequence
        Returns:
            text description string
        """
        if skeleton.dim() == 4:
            skeleton = skeleton.squeeze(0)

        T = skeleton.shape[0]

        if T < 2:
            return "老人在监控画面中"

        # Extract relevant keypoints
        # hips: indices 11, 12; shoulders: 5, 6; head: 0
        hip_y = skeleton[:, 11:13, 1].mean(dim=-1)  # (T,) average hip y
        shoulder_y = skeleton[:, 5:7, 1].mean(dim=-1)
        head_y = skeleton[:, 0, 1]

        # Velocities
        hip_vel = (hip_y[-1] - hip_y[-2]).item() if T >= 2 else 0
        shoulder_vel = (shoulder_y[-1] - shoulder_y[-2]).item() if T >= 2 else 0

        # Body angle (shoulder to hip)
        body_center = skeleton[:, 5:13, :2].mean(dim=1)  # (T, 2)
        body_std = body_center.std(dim=0).mean().item()  # overall movement

        # Decision rules
        hip_height = hip_y.mean().item()

        # Falling detection: hip rapidly moves down
        if hip_vel > 0.02:  # y increases downward in image coords
            if head_y[-1].item() > hip_y[-1].item() + 0.05:
                return "老人向前摔倒" if shoulder_vel > 0 else "老人向后倒下"
            return "老人正在跌倒"

        # Lying down: hip height very low
        if hip_height > 0.7:  # bottom of frame
            return "老人躺在地上" if body_std < 0.02 else "老人在地上挣扎"

        # Walking: moderate horizontal movement
        if 0.005 < body_std < 0.03:
            return "老人正在缓慢行走"

        # Bending: upper body lower than hips
        if shoulder_y.mean().item() > hip_height + 0.03:
            return "老人弯腰捡东西"

        # Standing: minimal movement
        if body_std < 0.005:
            return "老人保持静止站立"

        # Slight movement
        if body_std < 0.01:
            return "老人在轻微活动"

        return "老人在活动"

    def get_annotation(
        self,
        clip_id: Optional[str] = None,
        activity: Optional[str] = None,
        skeleton: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Get text annotation for a clip, preferring available sources in order:
        1. Pre-loaded annotation (from file)
        2. Activity template
        3. Skeleton-based heuristic

        Args:
            clip_id: optional clip identifier for lookup
            activity: optional activity label
            skeleton: optional skeleton tensor for heuristic

        Returns:
            text description string
        """
        if clip_id and clip_id in self._annotations_cache:
            return self._annotations_cache[clip_id]

        if activity:
            return self.from_activity(activity)

        if skeleton is not None:
            return self.from_skeleton(skeleton)

        return "老人在监控画面中"

    def generate_future_description(
        self,
        current_activity: str,
        is_fall: bool = False,
    ) -> str:
        """
        Generate a description of the future state for VL-JEPA training.

        Used as the target text embedding for Stage 2 (VL-JEPA text alignment).

        Args:
            current_activity: current observed activity
            is_fall: whether the future contains a fall

        Returns:
            future description string
        """
        import random

        if is_fall:
            fall_descs = [
                "老人摔倒在地上",
                "老人失去平衡后倒地",
                "老人跌倒后躺在地上",
                "老人因滑倒而仰面倒地",
                "老人向前扑倒在地",
            ]
            return random.choice(fall_descs)
        else:
            # Predict plausible continuation
            transitions = {
                'walking': ["老人继续缓慢行走", "老人走到椅子旁", "老人停下脚步"],
                'standing': ["老人继续站立", "老人开始缓慢走动", "老人坐在椅子上"],
                'sitting': ["老人继续坐着休息", "老人站起身", "老人坐着看书"],
                'lying': ["老人继续躺着休息", "老人翻身侧躺"],
                'bending': ["老人直起身", "老人捡起物品后站直"],
                'transition': ["老人完成转身", "老人坐稳在椅子上"],
            }
            options = transitions.get(current_activity, ["老人正常活动"])
            return random.choice(options)
