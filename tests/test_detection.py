import pytest
import torch

from intent_recovery.detection import IntentDetector, attach_labels, detection_metrics, load_detector
from intent_recovery.distributed import TrainingStep
from intent_recovery.model import IntentRecoveryModel, ModelConfig


def test_confusion_metrics_and_auc():
    result = detection_metrics(torch.tensor([3., 2., -2., -3.]), torch.tensor([1., 0., 1., 0.]))
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["f1"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["auroc"] == 0.75
    tied = detection_metrics(torch.zeros(4), torch.tensor([0, 1, 0, 1]))
    assert tied["auroc"] == 0.5
    assert detection_metrics(torch.ones(2), torch.ones(2))["auroc"] is None


def test_label_attachment_and_roundtrip():
    assert attach_labels([{}], [{"label": "harmful"}])[0]["detection_label"] == 1
    with pytest.raises(ValueError):
        attach_labels([{}], [{"label": "unknown"}])
    with pytest.raises(ValueError):
        attach_labels([{}], [])
    head = IntentDetector(3, 4).eval()
    restored = load_detector({"detector": {"config": head.config, "state_dict": head.state_dict()}}, "cpu")
    z = torch.randn(2, 3)
    assert torch.equal(head(z), restored(z))
    assert load_detector({}, "cpu") is None


def test_detection_does_not_change_graph_alignment_gradients():
    graph = IntentRecoveryModel(ModelConfig(in_dim=3, target_dim=3, hidden_dim=4, num_layers=1, dropout=0))
    detector = IntentDetector(3, 4)
    step = TrainingStep(graph, None, None, 1.0, 0, 0, detector=detector)
    batch = (torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 3))
    torch.manual_seed(0)
    loss, *_ = step(*batch, None, 1.0, detection_labels=torch.tensor([0., 1.]))
    loss.backward()
    actual = {n: p.grad.clone() for n, p in graph.named_parameters() if p.grad is not None}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in detector.parameters())
    graph.zero_grad()
    from intent_recovery.losses import intent_loss
    torch.manual_seed(0)
    intent_loss(graph(batch[0], batch[1])["z_G"], batch[2]).backward()
    for name, p in graph.named_parameters():
        if name in actual:
            assert torch.allclose(actual[name], p.grad)
