"""
Frame Masking Augmentation (Fall-Mamba Optimization 3)

Randomly masks a fraction of frames during training to improve robustness
to missing frames, occlusion, and low-light conditions.

From Fall-Mamba paper (Zhang et al., 2025):
 - Random mask 10-30% of frames during training
 - Improved accuracy from 98.88% → 99.63% on Le2i
 - Maintains 98.14% in dark conditions
"""

import torch


def frame_masking_augment(
    frames: torch.Tensor,
    mask_ratio: float = 0.2,
    mask_value: float = 0.0,
    temporal_only: bool = False,
) -> torch.Tensor:
    """
    Randomly mask frames with a given ratio.

    Args:
        frames: (B, T, C, H, W) or (T, C, H, W) video frames
        mask_ratio: fraction of frames to mask (0.1 - 0.3)
        mask_value: value to fill masked frames (0.0 = black, or learnable)
        temporal_only: if True, masks entire frames (same frame index across batch)
                       if False, each sample independently

    Returns:
        masked_frames: same shape as input, with masked frames zeroed
    """
    if frames.dim() == 4:
        # Single sample: (T, C, H, W)
        T = frames.shape[0]
        if temporal_only:
            mask = torch.rand(T, 1, 1, 1, device=frames.device) > mask_ratio
        else:
            mask = torch.rand(T, 1, 1, 1, device=frames.device) > mask_ratio
        return frames * mask.float() + mask_value * (1 - mask.float())
    else:
        # Batch: (B, T, C, H, W)
        B, T = frames.shape[:2]

        if temporal_only:
            # Same frame index masked across batch
            mask = torch.rand(T, device=frames.device) > mask_ratio
            mask = mask.view(1, T, 1, 1, 1)
        else:
            # Each sample independently
            mask = torch.rand(B, T, 1, 1, 1, device=frames.device) > mask_ratio

        return frames * mask.float()


def temporal_subsample(
    frames: torch.Tensor,
    target_frames: int,
    mode: str = 'uniform',
) -> torch.Tensor:
    """
    Sub-sample video frames to target count.

    Args:
        frames: (T, C, H, W) or (B, T, C, H, W)
        target_frames: desired number of frames
        mode: 'uniform' (evenly spaced), 'random' (random selection)

    Returns:
        Subsampled frames
    """
    is_batch = frames.dim() == 5
    if is_batch:
        B, T = frames.shape[:2]
        frames_flat = frames.view(B, T, -1)
    else:
        T = frames.shape[0]
        frames_flat = frames.view(T, -1).unsqueeze(0)
        B = 1

    if T <= target_frames:
        return frames  # Already small enough

    if mode == 'uniform':
        indices = torch.linspace(0, T - 1, target_frames).long()
    else:
        indices = torch.randperm(T)[:target_frames].sort()[0]

    indices = indices.to(frames.device)
    result = frames_flat[:, indices, :]

    if is_batch:
        result = result.view(B, target_frames, *frames.shape[2:])
    else:
        result = result.view(target_frames, *frames.shape[1:]).squeeze(0)

    return result
