import torch

from intent_recovery.data import collate, make_synthetic
from intent_recovery.losses import balance_loss, intent_loss
from intent_recovery.model import IntentRecoveryModel, ModelConfig


def _cfg(**kw):
    base = dict(in_dim=16, target_dim=16, hidden_dim=32, num_layers=2, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(sizes, dim=16):
    items = [{"H": torch.randn(n, dim), "z_q": torch.randn(dim)} for n in sizes]
    return collate(items)


def test_shapes_and_pair_mask():
    model = IntentRecoveryModel(_cfg(causal=True))
    b = _batch([3, 5, 4])
    out = model(b["H"], b["mask"])
    assert out["z_G"].shape == (3, 16)
    assert out["pi"].shape == (3, 5, 5, 4)
    pi = out["pi"]
    # causal: only j < i may carry messages
    upper = torch.triu(torch.ones(5, 5, dtype=torch.bool))
    assert pi[:, upper].abs().sum() == 0
    # valid pairs have assignments summing to one
    assert torch.allclose(pi.sum(-1)[out["valid"]], torch.ones(int(out["valid"].sum())), atol=1e-5)


def test_padding_invariance_eval():
    torch.manual_seed(0)
    model = IntentRecoveryModel(_cfg()).eval()
    short = {"H": torch.randn(3, 16), "z_q": torch.randn(16)}
    long = {"H": torch.randn(6, 16), "z_q": torch.randn(16)}
    alone = collate([short])
    padded = collate([short, long])
    z_alone = model(alone["H"], alone["mask"])["z_G"][0]
    z_padded = model(padded["H"], padded["mask"])["z_G"][0]
    assert torch.allclose(z_alone, z_padded, atol=1e-5)


def test_backward_reaches_all_modules():
    model = IntentRecoveryModel(_cfg())
    b = _batch([4, 5, 6])
    out = model(b["H"], b["mask"], tau=0.5)
    loss = intent_loss(out["z_G"], b["z_q"]) + 0.1 * balance_loss(out["pi"], out["valid"])
    loss.backward()
    for name in ["scorer.w_q.weight", "scorer.w_k.weight", "layers.0.w1.weight", "layers.0.w3.weight",
                 "layers.0.u_i.weight", "layers.0.w2.weight", "readout.0.weight"]:
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # the edge scorer is learned through pi (Gumbel-Softmax path)
    assert model.scorer.w_q.weight.grad.abs().sum() > 0


def test_losses_ranges():
    z = torch.randn(4, 16)
    assert intent_loss(z, z).item() < 1e-5
    assert 0.0 <= intent_loss(z, -z).item() <= 2.0 + 1e-5


def test_second_layer_recomputes_relations_from_updated_nodes():
    model = IntentRecoveryModel(_cfg(dynamic_relations=True)).eval()
    inputs, layer_outputs = [], []
    scorer_hook = model.scorer.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    layer_hook = model.layers[0].register_forward_hook(lambda module, args, result: layer_outputs.append(result.detach().clone()))
    batch = _batch([3, 5])
    out = model(batch["H"], batch["mask"])
    scorer_hook.remove()
    layer_hook.remove()
    assert len(inputs) == 2
    assert torch.equal(inputs[1], layer_outputs[0])
    assert not torch.equal(inputs[0], inputs[1])
    for index, states in enumerate(inputs):
        assert torch.allclose(out["scores_by_layer"][index], model.scorer(states))
        pi = out["pi_by_layer"][index]
        assert pi[~out["valid"]].abs().sum() == 0
        assert torch.allclose(pi.sum(-1)[out["valid"]], torch.ones(int(out["valid"].sum())))
    assert out["pi"] is out["pi_by_layer"][-1]
    assert out["scores"] is out["scores_by_layer"][-1]


def test_legacy_checkpoint_preserves_static_relation_behavior():
    from dataclasses import asdict
    original = IntentRecoveryModel(_cfg(dynamic_relations=False)).eval()
    saved_config = asdict(original.cfg)
    saved_config.pop("dynamic_relations")
    restored = IntentRecoveryModel(ModelConfig.from_checkpoint(saved_config)).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    seen = []
    handle = restored.scorer.register_forward_pre_hook(lambda module, args: seen.append(1))
    batch = _batch([4])
    actual = restored(batch["H"], batch["mask"])
    handle.remove()
    expected = original(batch["H"], batch["mask"])
    assert len(seen) == 1
    assert torch.equal(actual["z_G"], expected["z_G"])
    assert actual["pi_by_layer"][0] is actual["pi_by_layer"][1]
    assert ModelConfig.from_checkpoint({"dynamic_relations": True}).dynamic_relations


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    items = make_synthetic(128, dim=16, seed=0)
    cfg = _cfg()
    model = IntentRecoveryModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    b = collate(items)
    first = None
    for step in range(60):
        out = model(b["H"], b["mask"], tau=1.0)
        loss = intent_loss(out["z_G"], b["z_q"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
