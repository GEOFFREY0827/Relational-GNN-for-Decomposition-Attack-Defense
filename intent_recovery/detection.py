"""Sequence-level harmful intent probe on graph embeddings (harmful=1)."""
import torch


class IntentDetector(torch.nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        if min(dim, hidden_dim) <= 0:
            raise ValueError("Detector dimensions must be positive")
        self.config = {"dim": dim, "hidden_dim": hidden_dim}
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_dim, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def attach_labels(items, records):
    if len(items) != len(records):
        raise ValueError("Embedding count does not match dataset; regenerate the cache")
    result = []
    for item, record in zip(items, records):
        label = record.get("label")
        if label not in ("harmful", "benign"):
            raise ValueError(f"Expected harmful/benign label for {record.get('id')}")
        result.append(dict(item, detection_label=int(label == "harmful")))
    return result


def detection_metrics(logits, labels, threshold=0.5):
    logits, labels = logits.detach().cpu().reshape(-1), labels.detach().cpu().reshape(-1)
    if not 0 <= threshold <= 1 or len(logits) == 0 or len(logits) != len(labels):
        raise ValueError("Invalid detection threshold or sample counts")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Expected finite logits and binary labels")
    positive = labels == 1
    predicted = logits.sigmoid() >= threshold
    tp = int((positive & predicted).sum())
    fp = int((~positive & predicted).sum())
    tn = int((~positive & ~predicted).sum())
    fn = int((positive & ~predicted).sum())
    def ratio(a, b):
        return a / b if b else 0.0
    positives, negatives = logits[positive], logits[~positive]
    auc = None
    if len(positives) and len(negatives):
        # Pairwise AUC gives equal scores half credit; no sklearn dependency.
        diff = positives[:, None] - negatives[None, :]
        auc = ((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item()
    return {"threshold": threshold, "num_samples": len(labels),
            "num_harmful": tp + fn, "num_benign": tn + fp,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": ratio(tp + tn, len(labels)), "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "false_positive_rate": ratio(fp, fp + tn), "false_negative_rate": ratio(fn, fn + tp),
            "auroc": auc}


def load_detector(checkpoint, device):
    saved = checkpoint.get("detector")
    if saved is None:
        return None
    head = IntentDetector(**saved["config"]).to(device)
    head.load_state_dict(saved["state_dict"], strict=True)
    return head.eval()
