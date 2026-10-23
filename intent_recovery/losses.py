import math
import torch
import torch.nn.functional as F


def intent_loss(z_g: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """L_intent = 1 - cos(z_G, z_Q). z_Q is a fixed target (no gradient)."""
    return (1.0 - F.cosine_similarity(z_g, z_q.detach(), dim=-1)).mean()


def balance_loss(pi: torch.Tensor, valid: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(mean quadrant usage || uniform) over valid pairs, averaged at batch level.

    Discourages the four quadrants from collapsing onto a single one. Set its weight to 0 to disable.
    """
    count = valid.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1).to(pi.dtype)  # [B, 1]
    usage = (pi.sum(dim=(1, 2)) / count).mean(dim=0)                       # [4]
    usage = usage / usage.sum().clamp(min=eps)
    k = usage.numel()
    return (usage * (torch.log(usage + eps) - math.log(1.0 / k))).sum()
