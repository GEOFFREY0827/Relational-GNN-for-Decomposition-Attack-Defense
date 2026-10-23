"""Frozen GTR embedding inversion and lexical reconstruction diagnostics."""
from collections import Counter
from contextlib import nullcontext
import math
import re

import torch
import torch.nn.functional as F

GTR_ENCODER = "vec2text/gtr-base"
INVERTER = "jxm/gtr__nq__32"


class DecoderAdapter(torch.nn.Module):
    """Residual MLP in GTR vector space before the frozen inverter projection.

    Starts as identity on norm-calibrated inputs. Alignment continues to act
    on z_G; reconstruction trains this adapter on detached graph vectors.
    """
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.config = {"dim": dim, "hidden_dim": hidden_dim}
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden_dim), torch.nn.GELU(),
                                       torch.nn.Linear(hidden_dim, dim))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z):
        return z + self.mlp(z)


def load_decoder_adapter(checkpoint, device):
    """Legacy checkpoints have no adapter; new ones must contain config + weights."""
    saved = checkpoint.get("decoder_adapter")
    if saved is None:
        return None
    adapter = DecoderAdapter(**saved["config"]).to(device)
    adapter.load_state_dict(saved["state_dict"], strict=True)
    return adapter.eval()


def prepare_prediction(z, mean_norm):
    """Cosine training leaves norms unconstrained; use only a training-set scalar."""
    if not math.isfinite(mean_norm) or mean_norm <= 0:
        raise ValueError("A positive finite training target_mean_norm is required")
    if not torch.isfinite(z).all() or (z.norm(dim=-1) < 1e-12).any():
        raise ValueError("Predictions must be finite and nonzero")
    return F.normalize(z, dim=-1) * mean_norm


def text_scores(prediction, reference):
    """Lexical diagnostics, not an intent-equivalence judge."""
    def tokens(s):
        return re.findall(r"\w+", s.casefold())
    p, r = tokens(prediction), tokens(reference)
    overlap = sum((Counter(p) & Counter(r)).values())
    f1 = 2 * overlap / (len(p) + len(r)) if p or r else 1.0
    return {"normalized_exact_match": float(p == r), "word_f1": f1}


class GTRInverter:
    def __init__(self, device="cpu", steps=0):
        import vec2text

        self.api = vec2text
        self.steps = steps
        self.device = device
        if steps:
            self.corrector = vec2text.load_pretrained_corrector("gtr-base")
            self.corrector.model.to(device).eval()
            self.model = self.corrector.inversion_trainer.model.to(device).eval()
        else:
            # Avoid trainer construction and its external metric downloads for zero-step decoding.
            self.corrector = None
            self.model = vec2text.models.InversionModel.from_pretrained(INVERTER).to(device).eval()
        self.model.requires_grad_(False)
        self.lora_config = None

    def enable_decoder_lora(self, rank=8, alpha=16):
        """Adapt only T5 decoder attention q/v; encoder and base weights stay frozen."""
        if self.corrector is not None:
            raise ValueError("Decoder LoRA currently supports zero-step inversion only")
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.lora_config is not None:
            raise ValueError("LoRA is already enabled")
        from peft import LoraConfig, get_peft_model

        seq2seq = self.model.encoder_decoder
        targets = [name for name, module in seq2seq.named_modules()
                   if name.startswith("decoder.") and name.rsplit(".", 1)[-1] in ("q", "v")
                   and isinstance(module, torch.nn.Linear)]
        if not targets:
            raise ValueError("No T5 decoder q/v projections found")
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                            bias="none", task_type="SEQ_2_SEQ_LM", target_modules=targets)
        self.model.encoder_decoder = get_peft_model(seq2seq, config)
        self.model.eval()  # Base dropout stays disabled; LoRA weights still receive gradients.
        self.lora_config = {"rank": rank, "alpha": alpha}

    def lora_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def lora_checkpoint(self):
        if self.lora_config is None:
            return None
        from peft import get_peft_model_state_dict

        return {"config": self.lora_config, "state_dict": {
            k: v.detach().cpu().clone() for k, v in
            get_peft_model_state_dict(self.model.encoder_decoder).items()}}

    def load_lora_checkpoint(self, saved):
        if saved is None:
            return
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        self.enable_decoder_lora(**saved["config"])
        expected = get_peft_model_state_dict(self.model.encoder_decoder)
        if expected.keys() != saved["state_dict"].keys():
            raise ValueError("LoRA checkpoint keys do not match the decoder")
        set_peft_model_state_dict(self.model.encoder_decoder, saved["state_dict"])

    def decode_oracle(self, embeddings):
        """Keep the original pretrained decoder as an unchanged oracle control."""
        context = self.model.encoder_decoder.disable_adapter() if self.lora_config else nullcontext()
        with context:
            return self.decode(embeddings)

    def reconstruction_loss(self, embeddings, references):
        """Frozen decoder, but preserve gradients to the predicted embeddings."""
        tokens = self.model.tokenizer(references, padding=True, truncation=True,
                                      max_length=128, return_tensors="pt").to(self.device)
        labels = tokens.input_ids.masked_fill(tokens.attention_mask == 0, -100)
        return self.model(embedder_input_ids=None, embedder_attention_mask=None,
                          frozen_embeddings=embeddings.to(self.device), labels=labels).loss

    @torch.no_grad()
    def decode(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.model.embedder_dim:
            raise ValueError("Expected vectors in the pretrained GTR embedding space")
        embeddings = embeddings.to(self.device)
        if self.corrector is not None:
            return self.api.invert_embeddings(embeddings, self.corrector, num_steps=self.steps)
        ids = self.model.generate(
            inputs={"frozen_embeddings": embeddings},
            generation_kwargs={"max_length": 128, "min_length": 1, "do_sample": False, "num_beams": 1},
        )
        return self.model.tokenizer.batch_decode(ids, skip_special_tokens=True)
