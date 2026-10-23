"""Train the four-quadrant intent-recovery network.

Real data:
    python train.py --train data/train.jsonl --val data/val.jsonl --out runs/exp1
Smoke test (random embeddings, no downloads):
    python train.py --synthetic --epochs 20 --out runs/synth
"""
import argparse
import json
import os
import random
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from intent_recovery.distributed import Runtime, TrainingStep
from intent_recovery.detection import IntentDetector, attach_labels

from intent_recovery.data import EmbeddedDataset, collate, get_embedded, make_synthetic, load_jsonl
from intent_recovery.losses import balance_loss, intent_loss
from intent_recovery.metrics import evaluate
from intent_recovery.model import IntentRecoveryModel, ModelConfig
from intent_recovery.tracking import add_tracking_args, init_tracking, metric_log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str)
    p.add_argument("--val", type=str)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--q-encoder", type=str, default=None, help="defaults to --encoder")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--static-relations", action="store_true",
                   help="ablation: compute relations once instead of before each layer")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-causal", action="store_true")
    p.add_argument("--hard", action="store_true", help="straight-through hard assignment in training")
    p.add_argument("--aggregate", choices=["mean", "sum"], default="mean")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32, help="batch size PER GPU/process")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.3)
    p.add_argument("--lambda-balance", type=float, default=0.0)
    p.add_argument("--lambda-generation", type=float, default=0.0,
                   help="weight of frozen Vec2Text token reconstruction loss; requires GTR targets")
    p.add_argument("--adapter-hidden-dim", type=int, default=256,
                   help="hidden width of trainable GNN-to-decoder MLP; enabled with generation loss")
    p.add_argument("--decoder-lora", action="store_true", help="train decoder q/v LoRA alongside MLP")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--decoder-lr", type=float, default=1e-4, help="learning rate for decoder LoRA")
    p.add_argument("--detection-head", action="store_true", help="train a harmful/benign probe on detached z_G")
    p.add_argument("--detector-hidden-dim", type=int, default=128)
    p.add_argument("--lambda-detection", type=float, default=1.0)
    p.add_argument("--detection-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/exp")
    add_tracking_args(p)
    return p.parse_args()


def train(args, runtime):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = runtime.device
    print(f"rank={runtime.rank}/{runtime.world_size} device={device} "
          f"per_device_batch={args.batch_size} global_batch={args.batch_size * runtime.world_size}", flush=True)
    if args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("batch-size and epochs must be positive")
    if args.detection_head and (args.synthetic or args.lambda_detection <= 0 or
                                args.detector_hidden_dim <= 0 or not 0 <= args.detection_threshold <= 1):
        raise SystemExit("Detection requires labeled real data, positive dimensions/weight, and threshold in [0,1]")
    if args.lambda_generation < 0:
        raise SystemExit("--lambda-generation must be nonnegative")
    if args.decoder_lora and (args.lambda_generation <= 0 or args.lora_rank <= 0 or
                              args.lora_alpha <= 0 or args.decoder_lr <= 0):
        raise SystemExit("--decoder-lora requires positive generation weight, rank, alpha and decoder learning rate")
    if args.lambda_generation and (args.synthetic or (args.q_encoder or args.encoder) != "vec2text/gtr-base"):
        raise SystemExit("Generation supervision requires real data and --q-encoder vec2text/gtr-base")

    if args.synthetic:
        train_items = make_synthetic(512, seed=args.seed)
        val_items = make_synthetic(128, seed=args.seed + 1)
    else:
        if not args.train or not args.val:
            raise SystemExit("Provide --train and --val (JSONL), or use --synthetic.")
        # Only rank 0 creates embedding caches; all other ranks load them afterwards.
        if runtime.distributed and not runtime.primary:
            runtime.barrier()
        train_items = get_embedded(args.train, args.encoder, args.q_encoder, device)
        val_items = get_embedded(args.val, args.encoder, args.q_encoder, device)
        if runtime.distributed and runtime.primary:
            runtime.barrier()

    if not train_items or not val_items:
        raise SystemExit("Training and validation datasets must be non-empty")
    target_mean_norm = torch.stack([x["z_q"].norm() for x in train_items]).mean().item()
    decoder = None
    if args.lambda_generation:
        from intent_recovery.generation import DecoderAdapter, GTRInverter, prepare_prediction

        train_items = [dict(item, Q=row["Q"]) for item, row in zip(train_items, load_jsonl(args.train))]
        val_items = [dict(item, Q=row["Q"]) for item, row in zip(val_items, load_jsonl(args.val))]
        decoder = GTRInverter(device=device)
        if args.decoder_lora:
            decoder.enable_decoder_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    if args.detection_head:
        train_items = attach_labels(train_items, load_jsonl(args.train))
        val_items = attach_labels(val_items, load_jsonl(args.val))
        if len({x["detection_label"] for x in train_items}) != 2:
            raise SystemExit("Detection training requires both harmful and benign examples")

    in_dim = train_items[0]["H"].size(1)
    target_dim = train_items[0]["z_q"].size(0)
    cfg = ModelConfig(
        in_dim=in_dim,
        target_dim=target_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        causal=not args.no_causal,
        hard=args.hard,
        aggregate=args.aggregate,
        dynamic_relations=not args.static_relations,
    )
    model = IntentRecoveryModel(cfg).to(device)
    adapter = DecoderAdapter(target_dim, args.adapter_hidden_dim).to(device) if decoder is not None else None
    detector = IntentDetector(target_dim, args.detector_hidden_dim).to(device) if args.detection_head else None
    parameters = list(model.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
    if detector is not None:
        parameters += list(detector.parameters())
    groups = [{"params": parameters, "lr": args.lr}]
    lora_parameters = decoder.lora_parameters() if args.decoder_lora else []
    if lora_parameters:
        groups.append({"params": lora_parameters, "lr": args.decoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if args.decoder_lora and runtime.primary:
        print(f"Trainable decoder LoRA parameters: {sum(p.numel() for p in lora_parameters)}")

    system = TrainingStep(model, adapter, decoder, target_mean_norm, args.lambda_balance, args.lambda_generation,
                          detector=detector, detection_weight=args.lambda_detection)
    train_loader = DataLoader(EmbeddedDataset(train_items), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    step, opt, train_loader = runtime.accelerator.prepare(system, opt, train_loader)
    # Model initialization was identical; use independent dropout/Gumbel streams per rank.
    torch.manual_seed(args.seed + runtime.rank)
    val_loader = DataLoader(EmbeddedDataset(val_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    config = {"model": asdict(cfg), "args": vars(args), "world_size": runtime.world_size,
              "global_batch_size": args.batch_size * runtime.world_size}
    if runtime.primary:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    run = init_tracking(args, config, "train") if runtime.primary else None
    try:
        best = -1.0
        best_score = -float("inf")
        best_detection_f1 = -1.0
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            frac = epoch / max(1, args.epochs - 1)
            tau = args.tau_start + (args.tau_end - args.tau_start) * frac  # linear anneal

            step.train()
            train_loader.set_epoch(epoch)
            if decoder is not None:
                if hasattr(decoder, "model"):
                    decoder.model.eval()
            if adapter is not None:
                adapter.train()
            total, generation_total, total_loss, nb, detection_total = 0.0, 0.0, 0.0, 0, 0.0
            for batch in train_loader:
                H = batch["H"].to(device)
                mask = batch["mask"].to(device)
                z_q = batch["z_q"].to(device)

                opt.zero_grad()
                losses = step(H, mask, z_q, batch.get("Q"), tau,
                              detection_labels=batch.get("detection_labels"))
                loss, loss_intent, generation_loss = losses[:3]
                if detector is not None:
                    detection_total += losses[3].item() * H.size(0)
                runtime.accelerator.backward(loss)
                runtime.accelerator.unscale_gradients(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if adapter is not None:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                if detector is not None:
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
                if lora_parameters:
                    torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
                opt.step()
                n = H.size(0)
                total += loss_intent.item() * n
                generation_total += generation_loss.item() * n
                total_loss += loss.item() * n
                nb += n

            total, generation_total, total_loss, nb, detection_total = runtime.sum(
                [total, generation_total, total_loss, nb, detection_total])
            # Rank 0 evaluates the complete validation set, so retrieval has all candidates
            # and no DistributedSampler padding/duplicate examples enter the metrics.
            if runtime.primary:
                unwrapped = runtime.accelerator.unwrap_model(step)
                m = evaluate(unwrapped.graph, val_loader, device, detector=detector,
                             detection_threshold=args.detection_threshold)
                if adapter is not None:
                    adapter.eval()
                    val_gen, val_count = 0.0, 0
                    with torch.no_grad():
                        for batch in val_loader:
                            zg = unwrapped.graph(batch["H"].to(device), batch["mask"].to(device))["z_G"]
                            n = len(batch["Q"])
                            val_gen += decoder.reconstruction_loss(
                                adapter(prepare_prediction(zg, target_mean_norm)), batch["Q"]).item() * n
                            val_count += n
                    m["generation_loss"] = val_gen / val_count
                # Include decoder performance when selecting a generation-supervised checkpoint.
                score = m["mean_cos"] - args.lambda_generation * m.get("generation_loss", 0.0)
                epoch_seconds = time.perf_counter() - epoch_start
                print(
                    f"epoch {epoch + 1:3d} tau={tau:.2f} train_L_intent={total / nb:.4f} "
                    f"train_L_gen={generation_total / nb:.4f} "
                    f"val_L_gen={m.get('generation_loss', 0.0):.4f} "
                    f"val_cos={m['mean_cos']:.4f} top1={m['retrieval_top1']:.3f} "
                    f"top1_opt={m['retrieval_top1_optimistic']:.3f} "
                    f"ties={m['retrieval_tied_target_fraction']:.3f} "
                    f"duplicate_pairs={m['duplicate_target_pairs']} "
                    f"quadrants={[round(x, 2) for x in m['quadrant_fraction']]} "
                    f"epoch_seconds={epoch_seconds:.1f}"
                )
                if detector is not None:
                    print(f"detection val_f1={m['detection/f1']:.4f} "
                          f"recall={m['detection/recall']:.4f} "
                          f"FPR={m['detection/false_positive_rate']:.4f} "
                          f"train_L_det={detection_total / nb:.4f}")
                if run is not None:
                    run.log({"epoch": epoch + 1, "train/intent_loss": total / nb,
                             "train/detection_loss": detection_total / nb,
                             "train/generation_loss": generation_total / nb, "train/total_loss": total_loss / nb,
                             "time/epoch_seconds": epoch_seconds,
                             "train/tau": tau, **metric_log(m, "val")}, step=epoch + 1)
                if score > best_score:
                    best_score = score
                    best = m["mean_cos"]
                    if run is not None:
                        run.summary["best/val_mean_cos"] = best
                        run.summary["best/epoch"] = epoch + 1
                        run.summary["best/selection_score"] = score
                        if adapter is not None:
                            run.summary["best/val_generation_loss"] = m["generation_loss"]
                    save_best = True
                else:
                    save_best = False
                checkpoint = {"model": model.state_dict(), "config": asdict(cfg),
                                "encoders": {"subtasks": args.encoder, "target": args.q_encoder or args.encoder},
                                "target_mean_norm": target_mean_norm,
                                "decoder_adapter": {"config": adapter.config, "state_dict": adapter.state_dict()} if adapter is not None else None,
                                "decoder_lora": decoder.lora_checkpoint() if args.decoder_lora else None,
                                "selection_score": score,
                                "selection_metric": "val_cos - lambda_generation * val_generation_loss",
                                "lambda_generation": args.lambda_generation,
                                "detector": {"config": detector.config, "state_dict": detector.state_dict(),
                                             "threshold": args.detection_threshold} if detector is not None else None}
                if save_best:
                    torch.save(checkpoint, os.path.join(args.out, "best.pt"))
                if detector is not None and m["detection/f1"] > best_detection_f1:
                    best_detection_f1 = m["detection/f1"]
                    checkpoint["detection_selection_metric"] = "val_f1"
                    checkpoint["detection_selection_score"] = best_detection_f1
                    torch.save(checkpoint, os.path.join(args.out, "best_detection.pt"))
                    if run is not None:
                        run.summary["best_detection/val_f1"] = best_detection_f1
                        run.summary["best_detection/epoch"] = epoch + 1


            runtime.barrier()

        if runtime.primary:
            print(f"selected checkpoint val cos: {best:.4f}; selection score: {best_score:.4f}")
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()


def main():
    args = parse_args()
    runtime = Runtime(args.device)
    try:
        train(args, runtime)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
