"""Server-side checks. No pretrained downloads or local GPU assumptions."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from intent_recovery.distributed import TrainingStep
from intent_recovery.generation import DecoderAdapter
from intent_recovery.model import IntentRecoveryModel, ModelConfig


def test_training_step_keeps_reconstruction_separate():
    class Decoder:
        lora_config = {"rank": 1}
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
        def lora_parameters(self):
            return [self.weight]
        def reconstruction_loss(self, z, references):
            return (z[:, 0] * self.weight).square().mean()

    graph = IntentRecoveryModel(ModelConfig(in_dim=3, target_dim=3, hidden_dim=4, num_layers=1))
    adapter = DecoderAdapter(3, 4)
    decoder = Decoder()
    system = TrainingStep(graph, adapter, decoder, 2.0, 0.0, 1.0)
    assert any(p is decoder.weight for p in system.parameters())
    H, mask, target = torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 3)
    loss, alignment, generation = system(H, mask, target, ["a", "b"], 1.0)
    assert loss.requires_grad and not alignment.requires_grad and not generation.requires_grad
    loss.backward()
    assert graph.readout[0].weight.grad.abs().sum() > 0
    assert adapter.mlp[-1].weight.grad.abs().sum() > 0
    assert decoder.weight.grad.abs().sum() > 0


@pytest.mark.skipif(os.environ.get("RUN_DISTRIBUTED_TESTS") != "1",
                    reason="Set RUN_DISTRIBUTED_TESTS=1 on the server to launch two CPU workers")
def test_accelerate_two_process_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "distributed"
    result = subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        "train.py", "--synthetic", "--device", "cpu", "--epochs", "1", "--batch-size", "128",
        "--hidden-dim", "4", "--layers", "1", "--out", str(out),
    ], cwd=root, capture_output=True, text=True, timeout=180,
       env={**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"})
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((out / "config.json").read_text())
    assert config["world_size"] == 2
    assert config["global_batch_size"] == 256
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    assert all(not key.startswith("module.") for key in checkpoint["model"])
    assert result.stdout.count("selected checkpoint val cos:") == 1
