"""The shared source template must not launch an unbalanced experiment."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from train.ippo_rnn_coordination_grid import main


def test_unprepared_root_config_is_rejected_before_training():
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.load(root / "train/config/ippo_coordination_grid.yaml")
    with pytest.raises(ValueError, match="--prepare"):
        main.__wrapped__(config)
