"""
DropPathway: Single-Modality Fault Tolerance (Fall-Mamba Optimization 4)

Randomly drops a modality during training to train the model to handle
missing sensor data (e.g., no audio at night, no skeleton in darkness).

From Fall-Mamba paper: model continues to work when a modality is missing,
without catastrophic failure.
"""

import torch
from typing import Dict, Tuple


def drop_modality(
    video: torch.Tensor,
    audio: torch.Tensor,
    skeleton: torch.Tensor,
    text_embed: torch.Tensor,
    drop_prob: float = 0.15,
    max_drop: int = 1,  # max modalities to drop simultaneously
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly drop modalities during training for robustness.

    Args:
        video: (B, ...) video tensor
        audio: (B, ...) audio tensor
        skeleton: (B, ...) skeleton tensor
        text_embed: (B, ...) text embedding tensor
        drop_prob: probability of dropping each modality
        max_drop: maximum number of modalities to drop at once

    Returns:
        video, audio, skeleton, text_embed, modality_mask
        where modality_mask is (4,) boolean: True = kept, False = dropped
    """
    B = video.shape[0]

    # Independent coin flip for each modality
    keep_v = torch.rand(1).item() > drop_prob
    keep_a = torch.rand(1).item() > drop_prob
    keep_s = torch.rand(1).item() > drop_prob
    keep_t = torch.rand(1).item() > drop_prob

    # Enforce max_drop constraint
    keeps = [keep_v, keep_a, keep_s, keep_t]
    n_dropped = sum(not k for k in keeps)

    if n_dropped > max_drop:
        # Keep the ones that were already kept, and re-enable some dropped ones
        dropped_indices = [i for i, k in enumerate(keeps) if not k]
        import random
        to_keep = random.sample(dropped_indices, n_dropped - max_drop)
        for idx in to_keep:
            keeps[idx] = True
        keep_v, keep_a, keep_s, keep_t = keeps

    # Apply drops
    video_out = video if keep_v else torch.zeros_like(video)
    audio_out = audio if keep_a else torch.zeros_like(audio)
    skeleton_out = skeleton if keep_s else torch.zeros_like(skeleton)
    text_embed_out = text_embed if keep_t else torch.zeros_like(text_embed)

    modality_mask = torch.tensor([keep_v, keep_a, keep_s, keep_t])

    return video_out, audio_out, skeleton_out, text_embed_out, modality_mask


def drop_pathway_structured(
    embeddings: Dict[str, torch.Tensor],
    drop_prob: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Structured DropPathway for dictionary of modality embeddings.

    Args:
        embeddings: dict with keys 'v_tokens', 'a_tokens', 's_tokens', 't_embed'
        drop_prob: per-modality drop probability

    Returns:
        Updated embeddings dict (modified in place)
    """
    result = {}
    for key in ['v_tokens', 'a_tokens', 's_tokens', 't_embed']:
        if key in embeddings:
            if torch.rand(1).item() > drop_prob:
                result[key] = embeddings[key]
            else:
                result[key] = torch.zeros_like(embeddings[key])
    return result
