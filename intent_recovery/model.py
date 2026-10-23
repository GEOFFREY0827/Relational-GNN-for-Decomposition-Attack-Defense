"""Four-quadrant relational graph network for intent recovery.

Pipeline (per sample):
    subtask embeddings H -> quadrant edge scores (4 heads) -> Gumbel-Softmax assignment pi
    -> message passing, with relation reassignment before every layer -> pooled embedding z_G.

z_G is aligned with the embedding z_Q of the original intent Q (see losses.py).

Quadrant index (0-based) -> relation -> aggregation:
    0 (r1)  intent similar   / semantic similar    : sum_j W1 h_j
    1 (r2)  intent similar   / semantic unsimilar  : sum_j g_ij * W2 h_j,  g_ij = sigmoid(U[h_i; h_j])
    2 (r3)  intent unsimilar / semantic unsimilar  : sum_j W3 (h_i - h_j)
    3 (r4)  intent unsimilar / semantic similar    : no message
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_QUADRANTS = 4
QUADRANT_NAMES = (
    "r1: intent-sim / semantic-sim",
    "r2: intent-sim / semantic-unsim",
    "r3: intent-unsim / semantic-unsim",
    "r4: intent-unsim / semantic-sim (no message)",
)


@dataclass
class ModelConfig:
    in_dim: int = 384          # dimension of the (frozen) subtask encoder E
    target_dim: int = 384      # dimension of z_Q (the intent encoder E_Q); z_G is projected to this
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    causal: bool = True        # node i only attends to earlier nodes j < i (online defense)
    tau: float = 1.0           # default Gumbel-Softmax temperature
    hard: bool = False         # straight-through hard samples during training
    aggregate: str = "mean"    # "mean" (divide by #valid neighbours) or "sum"
    dynamic_relations: bool = True  # recompute scores using the current node states each layer

    @classmethod
    def from_checkpoint(cls, config):
        # Historical checkpoints computed relations once. Do not change their predictions.
        return cls(**{"dynamic_relations": False, **config})


class QuadrantEdgeScorer(nn.Module):
    """One attention head per quadrant: e_ij^(k) = <W_Q^(k) h_i, W_K^(k) h_j> / sqrt(d)."""

    def __init__(self, dim: int, num_quadrants: int = NUM_QUADRANTS):
        super().__init__()
        self.k = num_quadrants
        self.d = dim
        self.w_q = nn.Linear(dim, num_quadrants * dim, bias=False)
        self.w_k = nn.Linear(dim, num_quadrants * dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        q = self.w_q(h).view(b, n, self.k, d)
        k = self.w_k(h).view(b, n, self.k, d)
        return torch.einsum("bikd,bjkd->bijk", q, k) / math.sqrt(d)  # [B, n, n, 4]


def build_pair_mask(node_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    """valid[b, i, j] is True if the pair (i, j) may carry a message (i receives from j)."""
    b, n = node_mask.shape
    valid = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=node_mask.device)
    valid = valid & ~eye
    if causal:
        lower = torch.tril(torch.ones(n, n, dtype=torch.bool, device=node_mask.device), diagonal=-1)
        valid = valid & lower  # j < i
    return valid


def assign_quadrants(scores: torch.Tensor, tau: float, hard: bool, training: bool) -> torch.Tensor:
    """Soft (Gumbel-Softmax) assignment in training, hard argmax one-hot in eval."""
    if training:
        return F.gumbel_softmax(scores, tau=tau, hard=hard, dim=-1)
    idx = scores.argmax(dim=-1)
    return F.one_hot(idx, scores.size(-1)).to(scores.dtype)


class QuadrantMessageLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)
        self.w3 = nn.Linear(dim, dim, bias=False)
        # sigmoid(U [h_i; h_j]) = sigmoid(U_i h_i + U_j h_j + b)
        self.u_i = nn.Linear(dim, dim, bias=True)
        self.u_j = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pi: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        # h: [B, n, d]; pi: [B, n, n, 4] (already zero on invalid pairs); deg: [B, n, 1]
        p1, p2, p3 = pi[..., 0], pi[..., 1], pi[..., 2]  # r4 (index 3) sends no message

        # r1: linear
        m1 = torch.einsum("bij,bjd->bid", p1, self.w1(h))

        # r2: gated, non-additive
        a = self.w2(h)                                                       # [B, n, d]
        gate = torch.sigmoid(self.u_i(h).unsqueeze(2) + self.u_j(h).unsqueeze(1))  # [B, n, n, d]
        m2 = torch.einsum("bij,bijd->bid", p2, gate * a.unsqueeze(1))

        # r3: difference, W3 (h_i - h_j) = W3 h_i - W3 h_j
        w3h = self.w3(h)
        m3 = p3.sum(dim=-1, keepdim=True) * w3h - torch.einsum("bij,bjd->bid", p3, w3h)

        m = (m1 + m2 + m3) / deg
        return self.norm(h + self.drop(F.relu(m)))


class IntentRecoveryModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.in_proj = nn.Sequential(nn.Linear(cfg.in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim))
        self.scorer = QuadrantEdgeScorer(cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [QuadrantMessageLayer(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.target_dim),
        )

    def forward(
        self,
        H: torch.Tensor,
        node_mask: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """H: [B, n, in_dim] subtask embeddings (padded); node_mask: [B, n] bool (True = real node)."""
        cfg = self.cfg
        tau = cfg.tau if tau is None else tau

        h = self.in_proj(H)
        valid = build_pair_mask(node_mask, cfg.causal)            # [B, n, n]

        if cfg.aggregate == "mean":
            deg = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(h.dtype)
        else:
            deg = torch.ones_like(h[..., :1])

        scores_by_layer, pi_by_layer = [], []
        for index, layer in enumerate(self.layers):
            if index == 0 or cfg.dynamic_relations:
                scores = self.scorer(h)
                pi = assign_quadrants(scores, tau, cfg.hard, self.training)
                pi = pi * valid.unsqueeze(-1).to(pi.dtype)
            scores_by_layer.append(scores)
            pi_by_layer.append(pi)
            h = layer(h, pi, deg)

        m = node_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        z_g = self.readout(pooled)
        # Existing pi/scores fields refer to the LAST message-passing layer.
        return {"z_G": z_g, "pi": pi, "scores": scores, "valid": valid,
                "pi_by_layer": tuple(pi_by_layer), "scores_by_layer": tuple(scores_by_layer)}
