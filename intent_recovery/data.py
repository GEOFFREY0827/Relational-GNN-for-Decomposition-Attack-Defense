"""Data utilities.

Input format (JSONL, one sample per line):
    {"Q": "original intent text", "S": ["subtask 1", "subtask 2", ...]}

Subtasks and intents are embedded once with frozen sentence encoders (E for subtasks, E_Q for Q),
cached to disk, then batched with padding.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "Q" not in r or "S" not in r or len(r["S"]) == 0:
                raise ValueError(f"Each record needs 'Q' and a non-empty 'S': {r}")
            records.append(r)
    return records


class SentenceEncoder:
    """Thin wrapper around sentence-transformers (frozen, no gradients)."""

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64):
        self.gtr = model_name == "vec2text/gtr-base"
        self.batch_size = batch_size
        if self.gtr:
            from transformers import AutoTokenizer, T5EncoderModel

            name = "sentence-transformers/gtr-t5-base"
            self.model = T5EncoderModel.from_pretrained(name).to(device or "cpu").eval()
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.dim = self.model.config.d_model
            return
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.gtr:
            chunks = []
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                        truncation=True, max_length=128, return_tensors="pt")
                tokens = tokens.to(next(self.model.parameters()).device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
                # Vec2Text gtr_base uses raw masked mean pooling, NOT ST normalization.
                chunks.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu())
            return torch.cat(chunks)
        emb = self.model.encode(
            texts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=len(texts) > 2000
        )
        return emb.float().cpu()


def embed_records(records: List[Dict], enc_s: SentenceEncoder, enc_q: Optional[SentenceEncoder] = None):
    """Return a list of {"H": [n, d_s], "z_q": [d_q]} tensors."""
    enc_q = enc_q or enc_s
    flat = [s for r in records for s in r["S"]]
    flat_emb = enc_s.encode(flat)
    q_emb = enc_q.encode([r["Q"] for r in records])
    out, ptr = [], 0
    for r, zq in zip(records, q_emb):
        n = len(r["S"])
        out.append({"H": flat_emb[ptr : ptr + n].clone(), "z_q": zq.clone()})
        ptr += n
    return out


def get_embedded(path: str, enc_s_name: str, enc_q_name: Optional[str] = None, device: Optional[str] = None):
    """Embed a JSONL file, using a cache file next to it if it is up to date."""
    enc_q_name = enc_q_name or enc_s_name
    tag = f"{enc_s_name}__{enc_q_name}".replace("/", "_")
    cache = f"{path}.{tag}.pt"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return torch.load(cache)
    records = load_jsonl(path)
    enc_s = SentenceEncoder(enc_s_name, device=device)
    enc_q = enc_s if enc_q_name == enc_s_name else SentenceEncoder(enc_q_name, device=device)
    data = embed_records(records, enc_s, enc_q)
    torch.save(data, cache)
    return data


def make_synthetic(num: int, dim: int = 64, n_min: int = 4, n_max: int = 8, seed: int = 0):
    """Random embeddings with a learnable structure, for smoke tests only (no real semantics).

    z_Q mixes a non-additive pair interaction (nodes 0 and 1) with the mean of the remaining nodes.
    """
    assert n_min >= 3
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(dim, dim, generator=g) / dim**0.5
    out = []
    for _ in range(num):
        n = int(torch.randint(n_min, n_max + 1, (1,), generator=g))
        H = torch.randn(n, dim, generator=g)
        z = torch.tanh(H[0] * H[1]) + H[2:].mean(dim=0)
        out.append({"H": H, "z_q": z @ mix})
    return out


class EmbeddedDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    n_max = max(b["H"].size(0) for b in batch)
    d = batch[0]["H"].size(1)
    H = torch.zeros(len(batch), n_max, d)
    mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["H"].size(0)
        H[i, :n] = b["H"]
        mask[i, :n] = True
    z_q = torch.stack([b["z_q"] for b in batch])
    result = {"H": H, "mask": mask, "z_q": z_q}
    if "Q" in batch[0]:
        result["Q"] = [b["Q"] for b in batch]
    if "detection_label" in batch[0]:
        result["detection_labels"] = torch.tensor([b["detection_label"] for b in batch], dtype=torch.float32)
    return result
