import torch
import torch.nn.functional as F

from .model import NUM_QUADRANTS


def retrieval_metrics(zg, zq, tie_atol=1e-6):
    """Exact-instance retrieval with conservative and optimistic tie bounds.

    Scores within tie_atol of the true target are treated as tied. Default
    top-k requires the target to succeed even in the worst ordering of ties.
    Duplicate targets are diagnosed as matching normalized embedding pairs.
    """
    if zg.ndim != 2 or zg.shape != zq.shape or not len(zg):
        raise ValueError("Expected matching non-empty [samples, dimension] tensors")
    if tie_atol < 0 or not torch.isfinite(zg).all() or not torch.isfinite(zq).all():
        raise ValueError("Expected finite embeddings and nonnegative tie tolerance")
    q = F.normalize(zq, dim=-1)
    sim = F.normalize(zg, dim=-1) @ q.T
    delta = sim - sim.diag().unsqueeze(1)
    other = ~torch.eye(len(zg), dtype=torch.bool, device=sim.device)
    better = ((delta > tie_atol) & other).sum(1)
    tied = ((delta.abs() <= tie_atol) & other).sum(1)
    # Coordinate-level agreement is distinct from a coincidentally tied query score.
    duplicates = torch.zeros(len(zq), len(zq), dtype=torch.bool, device=zq.device)
    for i in range(len(zq)):
        duplicates[i] = ((q - q[i]).abs() <= tie_atol).all(dim=1)
    duplicates.fill_diagonal_(False)
    result = {
        "retrieval_tie_atol": tie_atol,
        "retrieval_tied_target_fraction": (tied > 0).float().mean().item(),
        "retrieval_mean_tied_candidates": tied.float().mean().item(),
        "duplicate_target_pairs": int(duplicates.triu(1).sum().item()),
        "duplicate_target_sample_fraction": duplicates.any(1).float().mean().item(),
    }
    for k in (1, 5):
        result[f"retrieval_top{k}"] = ((better + tied) < k).float().mean().item()
        result[f"retrieval_top{k}_optimistic"] = (better < k).float().mean().item()
    return result


@torch.no_grad()
def evaluate(model, loader, device, detector=None, detection_threshold=0.5) -> dict:
    """Intent-recovery evaluation: cos(z_G, z_Q) with the ground-truth intent.

    Also reports retrieval accuracy (is the true z_Q the nearest one to z_G among the whole eval set?)
    and the fraction of node pairs assigned to each quadrant.
    """
    model.eval()
    if detector is not None:
        detector.eval()
    logits_all, labels_all = [], []
    zg_all, zq_all = [], []
    counts = torch.zeros(NUM_QUADRANTS)
    layer_counts = None
    for batch in loader:
        H = batch["H"].to(device)
        mask = batch["mask"].to(device)
        out = model(H, mask)
        if detector is not None:
            logits_all.append(detector(out["z_G"]).cpu())
            labels_all.append(batch["detection_labels"].cpu())
        zg_all.append(out["z_G"].cpu())
        zq_all.append(batch["z_q"])
        idx = out["scores"].argmax(dim=-1)
        valid = out["valid"]
        for k in range(NUM_QUADRANTS):
            counts[k] += ((idx == k) & valid).sum().item()
        scores_by_layer = out.get("scores_by_layer", (out["scores"],))
        if layer_counts is None:
            layer_counts = torch.zeros(len(scores_by_layer), NUM_QUADRANTS)
        for index, scores in enumerate(scores_by_layer):
            assignment = scores.argmax(dim=-1)
            for k in range(NUM_QUADRANTS):
                layer_counts[index, k] += ((assignment == k) & valid).sum().item()

    zg = torch.cat(zg_all)
    zq = torch.cat(zq_all)
    cos = F.cosine_similarity(zg, zq, dim=-1)
    result = {
        "num_samples": int(zg.size(0)),
        "mean_cos": cos.mean().item(),
        **retrieval_metrics(zg, zq),
        "quadrant_fraction": (counts / counts.sum().clamp(min=1)).tolist(),
        **{f"quadrant_fraction_layer_{i+1}": (c / c.sum().clamp(min=1)).tolist()
           for i, c in enumerate(layer_counts)},
    }
    if detector is not None:
        from .detection import detection_metrics
        result.update({f"detection/{k}": v for k, v in detection_metrics(
            torch.cat(logits_all), torch.cat(labels_all), detection_threshold).items()})
    return result
