"""
T-JEPA Phrase Retriever

Manages phrase library for Gate 2 semantic verification.

Builds a library of pre-encoded text descriptions covering:
  1. Fall phrases (various types of elderly falls)
  2. Non-fall anomaly phrases (cat, running, lights off, etc.)
  3. Normal activity phrases

During inference, z_text is compared against these phrases via cosine similarity
to determine whether an anomaly corresponds to a fall event.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default Phrase Libraries (Chinese + English)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Fall phrases (should trigger alarm)
FALL_PHRASES_CN = [
    "老人向前摔倒",
    "老人向后倒下",
    "老人侧向倒地",
    "老人从椅子上滑落",
    "老人从床上跌落",
    "老人失去平衡向前倾倒",
    "老人腿部无力摔倒",
    "老人被绊倒",
    "老人晕厥倒地",
    "老人突然跌倒",
    "老人头部着地摔倒",
    "老人滑倒",
    "老人走路时摔倒",
    "老人转身时失去平衡倒地",
    "老人扶着墙慢慢倒下",
    "老人脊椎着地摔倒",
]

FALL_PHRASES_EN = [
    "elderly person falling forward",
    "elderly person falling backward",
    "elderly person collapsing to the ground",
    "elderly person slipping and falling",
    "elderly person losing balance and falling",
    "elderly person tripping and falling",
    "elderly person fainting and falling",
    "elderly person sliding off chair",
    "elderly person falling off bed",
    "elderly person stumbling forward",
    "elderly person hitting head during fall",
    "elderly person sudden fall",
]

# Non-fall anomaly phrases (should NOT trigger alarm)
NON_FALL_PHRASES_CN = [
    "猫快速跑过",
    "狗在奔跑",
    "关灯",
    "开灯",
    "窗帘飘动",
    "椅子被移动",
    "快速奔跑",
    "跳跃",
    "弯腰捡东西",
    "蹲下",
    "坐在地板上",
    "躺下休息",
    "做伸展运动",
    "打扫卫生",
    "拖地",
    "窗外飞鸟掠过的影子",
    "光线变化",
]

NON_FALL_PHRASES_EN = [
    "cat running across",
    "dog running",
    "lights turning off",
    "lights turning on",
    "curtain moving",
    "chair being moved",
    "person running quickly",
    "person jumping",
    "person bending to pick up something",
    "person squatting down",
    "person sitting on floor",
    "person lying down to rest",
    "person stretching",
    "person cleaning",
    "shadow of bird passing window",
    "lighting change",
]

# Normal activity phrases
NORMAL_PHRASES_CN = [
    "老人缓慢行走",
    "老人站着",
    "老人坐着",
    "老人看书",
    "老人看电视",
    "老人喝水",
    "老人在窗前站立",
    "老人慢慢走过房间",
    "老人坐着聊天",
    "老人正常站立",
]

NORMAL_PHRASES_EN = [
    "elderly person walking slowly",
    "elderly person standing",
    "elderly person sitting",
    "elderly person reading",
    "elderly person watching TV",
    "elderly person drinking water",
    "elderly person standing by window",
    "elderly person walking across room",
    "elderly person sitting and talking",
    "elderly person standing normally",
]


class PhraseLibrary:
    """
    Encoded phrase library for cosine similarity search.

    Stores pre-computed text embeddings for fall, non-fall, and normal phrases.
    """

    def __init__(self, embed_dim: int = 3584):
        """
        Args:
            embed_dim: embedding dimension (matches text encoder)
        """
        self.embed_dim = embed_dim

        # Phrase storage
        self.phrases: List[str] = []
        self.embeddings: List[torch.Tensor] = []
        self.is_fall: List[bool] = []
        self.categories: List[str] = []  # 'fall', 'non_fall_anomaly', 'normal'

        self._built = False

    def build(
        self,
        text_encoder,
        custom_phrases: Optional[Dict[str, List[str]]] = None,
        use_chinese: bool = True,
    ):
        """
        Build phrase library by encoding all phrases with the text encoder.

        Args:
            text_encoder: callable that maps list[str] → (N, embed_dim) tensor
            custom_phrases: optional dict with keys 'fall', 'non_fall', 'normal'
            use_chinese: if True, use Chinese phrases; else English
        """
        all_phrases = []
        all_categories = []

        if use_chinese:
            # Fall phrases
            phrases_fall = custom_phrases.get('fall', FALL_PHRASES_CN) if custom_phrases else FALL_PHRASES_CN
            all_phrases.extend(phrases_fall)
            all_categories.extend(['fall'] * len(phrases_fall))

            # Non-fall anomaly phrases
            phrases_nonfall = custom_phrases.get('non_fall', NON_FALL_PHRASES_CN) if custom_phrases else NON_FALL_PHRASES_CN
            all_phrases.extend(phrases_nonfall)
            all_categories.extend(['non_fall_anomaly'] * len(phrases_nonfall))

            # Normal phrases
            phrases_normal = custom_phrases.get('normal', NORMAL_PHRASES_CN) if custom_phrases else NORMAL_PHRASES_CN
            all_phrases.extend(phrases_normal)
            all_categories.extend(['normal'] * len(phrases_normal))
        else:
            phrases_fall = custom_phrases.get('fall', FALL_PHRASES_EN) if custom_phrases else FALL_PHRASES_EN
            all_phrases.extend(phrases_fall)
            all_categories.extend(['fall'] * len(phrases_fall))

            phrases_nonfall = custom_phrases.get('non_fall', NON_FALL_PHRASES_EN) if custom_phrases else NON_FALL_PHRASES_EN
            all_phrases.extend(phrases_nonfall)
            all_categories.extend(['non_fall_anomaly'] * len(phrases_nonfall))

            phrases_normal = custom_phrases.get('normal', NORMAL_PHRASES_EN) if custom_phrases else NORMAL_PHRASES_EN
            all_phrases.extend(phrases_normal)
            all_categories.extend(['normal'] * len(phrases_normal))

        # Encode all phrases
        print(f"[PhraseLibrary] Encoding {len(all_phrases)} phrases...")
        with torch.no_grad():
            embeddings = text_encoder(all_phrases)  # (N, D)

        self.phrases = all_phrases
        self.embeddings = F.normalize(embeddings, dim=-1).cpu()  # pre-normalize
        self.is_fall = [c == 'fall' for c in all_categories]
        self.categories = all_categories
        self._built = True

        # Statistics
        n_fall = sum(self.is_fall)
        n_nonfall = len(self.is_fall) - n_fall
        print(f"[PhraseLibrary] Built: {len(all_phrases)} phrases "
              f"({n_fall} fall, {n_nonfall} non-fall)")

    def search(
        self,
        query_embed: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, float, bool]]:
        """
        Search phrase library by cosine similarity.

        Args:
            query_embed: (D,) query embedding (z_text from projector)
            top_k: number of top matches to return

        Returns:
            List of (phrase_text, similarity, is_fall) tuples
        """
        if not self._built:
            raise RuntimeError("Phrase library not built. Call build() first.")

        # Normalize query
        query_norm = F.normalize(query_embed.unsqueeze(0), dim=-1).cpu()

        # Cosine similarity - self.embeddings is (N, D) tensor
        similarities = (query_norm @ self.embeddings.T).squeeze(0)  # (N,)

        # Top-k
        top_sims, top_indices = similarities.topk(min(top_k, len(self.phrases)))

        results = []
        for idx, sim in zip(top_indices, top_sims):
            idx = int(idx)
            results.append((self.phrases[idx], float(sim), self.is_fall[idx]))

        return results

    def is_fall_phrase(self, phrase: str) -> bool:
        """Check if a phrase is in the fall category."""
        if phrase in self.phrases:
            idx = self.phrases.index(phrase)
            return self.is_fall[idx]
        return False

    def get_labels(self) -> List[Tuple[str, bool]]:
        """Return list of (phrase, is_fall) tuples for external use."""
        return list(zip(self.phrases, self.is_fall))

    def get_embedding_tensor(self) -> torch.Tensor:
        """Return all embeddings as a single tensor (N, D)."""
        return self.embeddings

    def save(self, path: str):
        """Save phrase library to disk."""
        data = {
            'phrases': self.phrases,
            'embeddings': self.embeddings.clone(),
            'is_fall': self.is_fall,
            'categories': self.categories,
            'embed_dim': self.embed_dim,
        }
        torch.save(data, path)

    def load(self, path: str):
        """Load phrase library from disk."""
        data = torch.load(path, map_location='cpu')
        self.phrases = data['phrases']
        self.embeddings = data['embeddings']
        self.is_fall = data['is_fall']
        self.categories = data['categories']
        self.embed_dim = data['embed_dim']
        self._built = True
        n_fall = sum(self.is_fall)
        print(f"[PhraseLibrary] Loaded: {len(self.phrases)} phrases "
              f"({n_fall} fall, {len(self.is_fall) - n_fall} non-fall)")
