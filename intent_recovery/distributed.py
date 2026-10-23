"""Hugging Face Accelerate setup; no implicit CPU fallback in DDP."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, requested="auto"):
        from accelerate import Accelerator, DataLoaderConfiguration
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

        world = int(os.environ.get("WORLD_SIZE", "1"))
        if requested == "cuda" or (world > 1 and requested != "cpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Check PyTorch CUDA build and driver; GPU training will not fall back to CPU.")
        self.accelerator = Accelerator(
            cpu=requested == "cpu", gradient_accumulation_steps=1,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True),
                             InitProcessGroupKwargs(timeout=timedelta(hours=1))])
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index
        self.distributed = self.world_size > 1
        self.device = self.accelerator.device
        if requested == "cuda" and self.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA but Accelerate selected {self.device}")

    @property
    def primary(self):
        return self.accelerator.is_main_process

    def barrier(self):
        if self.distributed and self.device.type == "cuda" and dist.is_initialized():
            # Cache barriers occur before DDP's first CUDA collective. Bind the
            # barrier explicitly so NCCL need not guess this rank's device.
            dist.barrier(device_ids=[self.device.index])
        else:
            self.accelerator.wait_for_everyone()

    def sum(self, values):
        totals = torch.tensor(values, dtype=torch.float64, device=self.device)
        return self.accelerator.reduce(totals, reduction="sum").tolist()

    def close(self):
        self.accelerator.end_training()
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


class TrainingStep(torch.nn.Module):
    """One DDP forward covers all trainable branches, including optional LoRA."""
    def __init__(self, graph, adapter, decoder, mean_norm, balance_weight, generation_weight,
                 detector=None, detection_weight=1.0):
        super().__init__()
        self.graph = graph
        self.adapter = adapter
        self.decoder = decoder
        self.detector = detector
        self.detection_weight = detection_weight
        # Register only LoRA leaves, avoiding DDP broadcasts of the frozen pretrained model.
        # The inverter still owns and uses these same Parameter objects in its forward.
        if decoder is not None and getattr(decoder, "lora_config", None):
            self.decoder_lora = torch.nn.ParameterList(decoder.lora_parameters())
        self.mean_norm = mean_norm
        self.balance_weight = balance_weight
        self.generation_weight = generation_weight

    def forward(self, H, mask, target, references, tau, detection_labels=None):
        from .losses import balance_loss, intent_loss
        from .generation import prepare_prediction

        output = self.graph(H, mask, tau=tau)
        alignment = intent_loss(output["z_G"], target)
        loss = alignment
        if self.balance_weight:
            assignments = output.get("pi_by_layer", (output["pi"],))
            loss = loss + self.balance_weight * torch.stack([
                balance_loss(pi, output["valid"]) for pi in assignments]).mean()
        generation = alignment.new_zeros(())
        if self.adapter is not None:
            adapted = self.adapter(prepare_prediction(output["z_G"].detach(), self.mean_norm))
            generation = self.decoder.reconstruction_loss(adapted, references)
            loss = loss + self.generation_weight * generation
        if self.detector is not None:
            if detection_labels is None:
                raise ValueError("Detection training requires labels")
            logits = self.detector(output["z_G"].detach())
            detection_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, detection_labels)
            loss = loss + self.detection_weight * detection_loss
            return loss, alignment.detach(), generation.detach(), detection_loss.detach()
        return loss, alignment.detach(), generation.detach()
