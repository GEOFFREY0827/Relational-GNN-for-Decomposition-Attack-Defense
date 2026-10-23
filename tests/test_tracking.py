import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from intent_recovery.tracking import add_tracking_args, init_tracking, metric_log


def test_tracking_default_does_not_initialize(monkeypatch):
    sdk = Mock()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    parser = argparse.ArgumentParser()
    add_tracking_args(parser)
    assert init_tracking(parser.parse_args([]), {}, "train") is None
    sdk.init.assert_not_called()


def test_tracking_passes_config_and_flattens_quadrants(monkeypatch):
    sdk = Mock()
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    args = SimpleNamespace(wandb=True, wandb_project="test", wandb_entity=None,
                           wandb_name="smoke", wandb_mode="offline")
    assert init_tracking(args, {"seed": 0}, "train") is sdk.init.return_value
    sdk.init.assert_called_once_with(project="test", entity=None, name="smoke",
                                     mode="offline", config={"seed": 0}, job_type="train")
    assert metric_log({"mean_cos": 0.7, "quadrant_fraction": [0.1, 0.2, 0.3, 0.4]}, "val") == {
        "val/mean_cos": 0.7, "val/quadrant_fraction/r1": 0.1,
        "val/quadrant_fraction/r2": 0.2, "val/quadrant_fraction/r3": 0.3,
        "val/quadrant_fraction/r4": 0.4,
    }
