"""Seeding for reproducible runs."""
from __future__ import annotations

import os
import random
from typing import Optional

DEFAULT_SEED = 7919


def set_seed(seed: int = DEFAULT_SEED) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass
    return seed


def seeded_rng(seed: Optional[int] = None) -> random.Random:
    """An isolated RNG that doesn't touch the global state."""
    return random.Random(DEFAULT_SEED if seed is None else seed)
