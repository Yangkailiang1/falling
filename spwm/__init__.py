# -*- coding: utf-8 -*-
"""
T-JEPA: Temporal Joint Embedding Predictive Architecture
Multimodal Fall Detection with Predictive Text Output.

Video + Audio + Skeleton + Text -> M3-JEPA MoE Fusion -> Anomaly Gate -> Text Output
"""

from .config import TJEPSConfig
from .tjepa_model import TJEPS

__all__ = ["TJEPSConfig", "TJEPS"]
