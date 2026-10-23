import json
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from intent_recovery.generation import GTR_ENCODER, DecoderAdapter, load_decoder_adapter, prepare_prediction, text_scores
from intent_recovery.model import IntentRecoveryModel, ModelConfig


def test_prediction_norm_uses_fixed_training_scalar():
    z = torch.tensor([[3., 4.], [0., 2.]])
    result = prepare_prediction(z, 2.5)
    assert torch.allclose(result.norm(dim=-1), torch.full((2,), 2.5))
    assert torch.allclose(torch.nn.functional.cosine_similarity(z, result), torch.ones(2))
    for scale in (0, float("nan")):
        with pytest.raises(ValueError):
            prepare_prediction(z, scale)
    with pytest.raises(ValueError):
        prepare_prediction(torch.zeros(1, 2), 1)


def test_lexical_metrics_handle_repeats_and_empty():
    assert text_scores("Hello, WORLD!", "hello world")["normalized_exact_match"] == 1
    assert text_scores("a a b", "a b b")["word_f1"] == pytest.approx(2/3)
    assert text_scores("", "")["word_f1"] == 1


def test_gtr_encoder_uses_raw_masked_mean(monkeypatch):
    from intent_recovery.data import SentenceEncoder

    class Tokens(dict):
        def to(self, device):
            return self
        @property
        def attention_mask(self):
            return self["attention_mask"]

    class Encoder(torch.nn.Module):
        config = SimpleNamespace(d_model=2)
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, **kw):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[2., 4.], [4., 8.], [90., 90.]]]))

    class Tokenizer:
        def __call__(self, texts, **kw):
            assert kw["max_length"] == 128
            return Tokens(attention_mask=torch.tensor([[1, 1, 0]]))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        T5EncoderModel=SimpleNamespace(from_pretrained=lambda _: Encoder()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: Tokenizer())))
    enc = SentenceEncoder(GTR_ENCODER)
    assert torch.equal(enc.encode(["invented text"]), torch.tensor([[3., 6.]]))


@pytest.mark.parametrize("use_adapter", [False, True])
def test_decode_cli_separates_oracle_and_prediction(tmp_path, monkeypatch, use_adapter):
    import intent_recovery.data as data
    import intent_recovery.generation as generation

    class Encoder:
        def __init__(self, *args, **kw):
            self.tokenizer = lambda texts, **kw: {"input_ids": [[1, 2] for _ in texts]}
        def encode(self, texts):
            # Distinct embeddings prove reference text changes cannot alter predicted text.
            return torch.tensor([[len(t), 1., 2.] for t in texts], dtype=torch.float32)

    class Inverter:
        def __init__(self, *args):
            pass
        def decode(self, embeddings):
            return [str([round(x, 5) for x in z.tolist()]) for z in embeddings]

    monkeypatch.setattr(data, "SentenceEncoder", Encoder)
    monkeypatch.setattr(generation, "GTRInverter", Inverter)
    cfg = ModelConfig(in_dim=3, target_dim=3, hidden_dim=4, num_layers=1)
    model = IntentRecoveryModel(cfg)
    ckpt = tmp_path / "model.pt"
    from dataclasses import asdict
    adapter = DecoderAdapter(3, 4)
    with torch.no_grad():
        adapter.mlp[-1].bias.fill_(0.7)
    torch.save({"config": asdict(cfg), "model": model.state_dict(),
                "encoders": {"subtasks": GTR_ENCODER, "target": GTR_ENCODER},
                "decoder_adapter": {"config": adapter.config, "state_dict": adapter.state_dict()} if use_adapter else None,
                "target_mean_norm": 3.0}, ckpt)
    test = tmp_path / "test.jsonl"
    outputs = []
    for reference in ("Plan a picnic", "A completely different reference"):
        test.write_text(json.dumps({"Q": reference, "S": ["Choose a park", "Check weather"],
                                   "canary": "invented-canary"}) + "\n")
        out = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(sys, "argv", ["decode_intent.py", "--test", str(test),
                                        "--ckpt", str(ckpt), "--out", str(out)])
        runpy.run_module("decode_intent", run_name="__main__")
        outputs.append(json.loads(out.read_text()))
    assert outputs[0]["predicted_text"] == outputs[1]["predicted_text"]
    assert outputs[0]["oracle_text"] != outputs[1]["oracle_text"]
    assert outputs[0]["canary"] == "invented-canary"
    from intent_recovery.data import collate
    flat = Encoder().encode(["Choose a park", "Check weather"])
    batch = collate([{"H": flat, "z_q": torch.zeros(3)}])
    model.eval()
    with torch.no_grad():
        expected = prepare_prediction(model(batch["H"], batch["mask"])["z_G"], 3.0)
        if use_adapter:
            expected = adapter(expected)
    assert outputs[0]["predicted_text"] == Inverter().decode(expected)[0]


def test_decode_rejects_old_checkpoint_before_model_download(tmp_path, monkeypatch):
    path = tmp_path / "old.pt"
    torch.save({"config": {}}, path)
    test = tmp_path / "test.jsonl"
    test.write_text('{"Q":"picnic", "S":["park"]}\n')
    monkeypatch.setattr(sys, "argv", ["decode_intent.py", "--test", str(test), "--ckpt", str(path)])
    with pytest.raises(SystemExit):
        runpy.run_module("decode_intent", run_name="__main__")


def test_reconstruction_loss_reaches_predictions_through_frozen_decoder():
    from intent_recovery.generation import GTRInverter

    class Tokens(SimpleNamespace):
        def to(self, device):
            return self

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
            self.tokenizer = lambda texts, **kw: Tokens(input_ids=torch.tensor([[1, 0]]),
                                                        attention_mask=torch.tensor([[1, 0]]))
        def forward(self, **kw):
            assert kw["labels"].tolist() == [[1, -100]]
            assert kw["embedder_input_ids"] is None
            return SimpleNamespace(loss=(kw["frozen_embeddings"] * self.weight).square().mean())

    decoder = GTRInverter.__new__(GTRInverter)
    decoder.model, decoder.device = Decoder(), "cpu"
    z = torch.tensor([[1., 2.]], requires_grad=True)
    decoder.reconstruction_loss(z, ["invented reference"]).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert decoder.model.weight.grad is None


def test_generation_supervised_training_saves_encoder_metadata(tmp_path, monkeypatch):
    import train
    import intent_recovery.generation as generation

    calls = []
    class Decoder:
        def __init__(self, **kw):
            pass
        def reconstruction_loss(self, embeddings, references):
            calls.append(references)
            assert embeddings.requires_grad == torch.is_grad_enabled()
            return (embeddings[:, 0] - 1).square().mean()

    monkeypatch.setattr(generation, "GTRInverter", Decoder)
    monkeypatch.setattr(train, "get_embedded", lambda *args: [
        {"H": torch.ones(2, 3), "z_q": torch.tensor([3., 4., 0.])},
        {"H": torch.eye(3), "z_q": torch.tensor([0., 0., 5.])}])
    monkeypatch.setattr(train, "load_jsonl", lambda path: [{"Q": "First intent"}, {"Q": "Second intent"}])
    monkeypatch.setattr(sys, "argv", ["train.py", "--train", "invented-train", "--val", "invented-val",
        "--encoder", GTR_ENCODER, "--lambda-generation", "0.1", "--epochs", "1",
        "--hidden-dim", "4", "--layers", "1", "--out", str(tmp_path)])
    train.main()
    saved = torch.load(tmp_path / "best.pt", weights_only=True)
    assert saved["encoders"] == {"subtasks": GTR_ENCODER, "target": GTR_ENCODER}
    assert saved["target_mean_norm"] == 5.0
    assert saved["lambda_generation"] == 0.1
    assert saved["decoder_adapter"]["config"]["dim"] == 3
    # Starts with zero output bias; an optimizer step must update it.
    assert saved["decoder_adapter"]["state_dict"]["mlp.2.bias"].abs().sum() > 0
    assert saved["selection_metric"] == "val_cos - lambda_generation * val_generation_loss"
    assert sorted(calls[0]) == ["First intent", "Second intent"]


def test_adapter_identity_gradient_and_checkpoint_roundtrip():
    adapter = DecoderAdapter(3, 5)
    z = torch.tensor([[1., 2., 3.]], requires_grad=True)
    assert torch.equal(adapter(z), z)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    loss = (adapter(z) - torch.tensor([[3., 1., 0.]])).square().mean()
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert adapter.mlp[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(adapter(z), z)
    saved = {"decoder_adapter": {"config": adapter.config, "state_dict": adapter.state_dict()}}
    loaded = load_decoder_adapter(saved, "cpu")
    assert not loaded.training
    assert torch.equal(loaded(z), adapter(z))
    assert load_decoder_adapter({}, "cpu") is None


def test_reconstruction_gradient_is_isolated_from_gnn():
    graph_vector = torch.tensor([[1., 2., 3.]], requires_grad=True)
    adapter = DecoderAdapter(3, 4)
    decoder_input = adapter(prepare_prediction(graph_vector.detach(), 2.0))
    (decoder_input - torch.tensor([[0., 1., 0.]])).square().mean().backward()
    assert graph_vector.grad is None
    assert adapter.mlp[-1].weight.grad.abs().sum() > 0
