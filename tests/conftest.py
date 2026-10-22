"""Shared pytest setup: keep every test offline and deterministic.

Deliberately defines no fixtures, so it cannot collide with fixtures in individual test files.
"""
from __future__ import annotations

import os

import torch

# Optional integrations must never reach the network from the test suite.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_runtest_setup(item) -> None:
    """Seed torch before every test so results do not depend on test order."""
    torch.manual_seed(0)
