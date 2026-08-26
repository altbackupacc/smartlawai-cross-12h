"""Deterministic seeding across python/numpy/torch. The five pre-registered seeds
for this project are 42, 1337, 2024, 31337, 8191 (RESEARCH.md #6)."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
