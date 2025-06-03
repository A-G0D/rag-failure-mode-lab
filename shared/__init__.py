"""Shared utilities: structured logging + determinism.

Small, dependency-free helpers reused across the project so the logging format
and the reproducibility guarantee stay consistent everywhere.
"""
from .obs import LogEvent, Observer, read_events
from .determinism import set_seed, seeded_rng

__all__ = ["LogEvent", "Observer", "read_events", "set_seed", "seeded_rng"]
