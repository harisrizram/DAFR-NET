"""
tests/test_classifier.py
Unit tests for DamageClassifier.

Run:  pytest tests/test_classifier.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.classifier.model import DamageClassifier, DAMAGE_CLASSES


# Minimal config — uses ResNet-18 (tiny) with pretrained=False for speed
CFG = {
    "model": {
        "backbone": "resnet18",
        "num_classes": 3,
        "pretrained": False,
        "freeze_backbone_epochs": 0,
    },
    "training": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 5,
    },
}


@pytest.fixture
def model():
    return DamageClassifier(CFG)


# ---------------------------------------------------------------------------

def test_forward_shape(model):
    x      = torch.randn(2, 3, 256, 256)
    logits = model(x)
    assert logits.shape == (2, 3), f"Got {logits.shape}"


def test_num_output_classes(model):
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape[-1] == 3


def test_predict_damage_type_returns_valid_class(model):
    x      = torch.randn(3, 256, 256)
    result = model.predict_damage_type(x)
    assert result in DAMAGE_CLASSES, f"Unknown class: {result}"


def test_training_step_returns_scalar_loss(model):
    x      = torch.randn(4, 3, 256, 256)
    y      = torch.randint(0, 3, (4,))
    loss   = model.training_step((x, y), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_validation_step_returns_scalar_loss(model):
    x    = torch.randn(2, 3, 64, 64)
    y    = torch.randint(0, 3, (2,))
    loss = model.validation_step((x, y), batch_idx=0)
    assert loss.ndim == 0


def test_configure_optimizers_returns_valid_pair(model):
    result = model.configure_optimizers()
    assert len(result) == 2
    optimizers, schedulers = result
    assert len(optimizers) == 1
    assert len(schedulers) == 1
