"""Server-side LoRA checks with a tiny random T5; no model downloads needed."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
from transformers import T5Config, T5ForConditionalGeneration

from intent_recovery.generation import GTRInverter


def make_inverter(state=None):
    config = T5Config(d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
                      num_heads=2, d_kv=8, vocab_size=24, decoder_start_token_id=0,
                      pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
    base = T5ForConditionalGeneration(config)
    if state is not None:
        base.load_state_dict(state)
    inverter = GTRInverter.__new__(GTRInverter)
    inverter.corrector = None
    inverter.lora_config = None
    inverter.model = torch.nn.Module()
    inverter.model.encoder_decoder = base.eval().requires_grad_(False)
    return inverter


def test_lora_only_updates_decoder_and_can_reload():
    inverter = make_inverter()
    initial = {k: v.clone() for k, v in inverter.model.encoder_decoder.state_dict().items()}
    inverter.enable_decoder_lora(rank=2, alpha=4)
    names = [n for n, p in inverter.model.named_parameters() if p.requires_grad]
    assert names and all("decoder." in n and "lora_" in n for n in names)
    embeddings = torch.randn(1, 3, 16, requires_grad=True)
    inputs = {"inputs_embeds": embeddings, "labels": torch.tensor([[2, 3, 1]])}
    model = inverter.model.encoder_decoder
    optimizer = torch.optim.AdamW(inverter.lora_parameters(), lr=0.01)
    model(**inputs).loss.backward()
    assert embeddings.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in inverter.lora_parameters())
    assert all(p.grad is None for n, p in inverter.model.named_parameters() if not p.requires_grad)
    optimizer.step()
    saved = inverter.lora_checkpoint()
    restored = make_inverter(initial)
    restored.load_lora_checkpoint(saved)
    with torch.no_grad():
        expected = model(**inputs).logits
        actual = restored.model.encoder_decoder(**inputs).logits
    assert torch.allclose(actual, expected)
    # Oracle must use original pretrained weights, not the post-trained LoRA.
    base = make_inverter(initial).model.encoder_decoder
    with torch.no_grad(), model.disable_adapter():
        assert torch.allclose(model(**inputs).logits, base(**inputs).logits)


def test_lora_rejects_iterative_corrector():
    inverter = make_inverter()
    inverter.corrector = SimpleNamespace()
    with pytest.raises(ValueError, match="zero-step"):
        inverter.enable_decoder_lora()
