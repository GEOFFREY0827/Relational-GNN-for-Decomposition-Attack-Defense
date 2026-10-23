import pytest
import torch

from intent_recovery.metrics import retrieval_metrics


def test_unique_correct_targets():
    z = torch.eye(8)
    m = retrieval_metrics(z, z)
    assert m["retrieval_top1"] == 1
    assert m["retrieval_top5"] == 1
    assert m["retrieval_tied_target_fraction"] == 0
    assert m["duplicate_target_pairs"] == 0


def test_all_equal_cannot_get_perfect_conservative_retrieval():
    z = torch.ones(8, 3)
    m = retrieval_metrics(z, z)
    assert m["retrieval_top1"] == 0
    assert m["retrieval_top5"] == 0
    assert m["retrieval_top1_optimistic"] == 1
    assert m["retrieval_top5_optimistic"] == 1
    assert m["retrieval_tied_target_fraction"] == 1
    assert m["retrieval_mean_tied_candidates"] == 7
    assert m["duplicate_target_pairs"] == 28
    assert m["duplicate_target_sample_fraction"] == 1


def test_duplicate_targets_only_affect_ambiguous_rows():
    z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    m = retrieval_metrics(z, z)
    assert m["retrieval_top1"] == pytest.approx(1 / 3)
    assert m["retrieval_top1_optimistic"] == 1
    assert m["retrieval_top5"] == 1
    assert m["duplicate_target_pairs"] == 1


def test_top5_boundary_and_wrong_targets():
    q = torch.eye(7)
    g = q.clone()
    # Target zero ranks sixth: five strictly better candidates.
    g[0] = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1])
    m = retrieval_metrics(g, q)
    assert m["retrieval_top1"] == pytest.approx(6 / 7)
    assert m["retrieval_top5"] == pytest.approx(6 / 7)


def test_near_ties_respect_tolerance():
    q = torch.eye(2)
    g = torch.tensor([[1., 1. + 1e-7], [0., 1.]])
    assert retrieval_metrics(g, q)["retrieval_top1_optimistic"] == 1
    assert retrieval_metrics(g, q, tie_atol=0)["retrieval_top1_optimistic"] == 0.5


def test_invalid_embeddings():
    with pytest.raises(ValueError):
        retrieval_metrics(torch.empty(0, 3), torch.empty(0, 3))
    with pytest.raises(ValueError):
        retrieval_metrics(torch.tensor([[float("nan")]]), torch.ones(1, 1))
