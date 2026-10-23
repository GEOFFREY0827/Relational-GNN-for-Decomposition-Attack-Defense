"""Evaluate a checkpoint: cos(z_G, z_Q) between the graph embedding and the true intent embedding.

    python eval.py --ckpt runs/exp1/best.pt --test data/test.jsonl
    python eval.py --ckpt runs/synth/best.pt --synthetic
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from intent_recovery.data import EmbeddedDataset, collate, get_embedded, make_synthetic, load_jsonl
from intent_recovery.detection import attach_labels, load_detector
from intent_recovery.metrics import evaluate
from intent_recovery.model import IntentRecoveryModel, ModelConfig
from intent_recovery.tracking import add_tracking_args, init_tracking, metric_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test", type=str)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--encoder", type=str, default=None, help="defaults to checkpoint encoder or MiniLM")
    p.add_argument("--q-encoder", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--detection-threshold", type=float, default=None,
                   help="override saved threshold; choose on validation, never on test")
    p.add_argument("--out", help="save numeric evaluation metrics as JSON")
    add_tracking_args(p)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_encoders = ckpt.get("encoders", {})
    args.encoder = args.encoder or saved_encoders.get("subtasks", "sentence-transformers/all-MiniLM-L6-v2")
    args.q_encoder = args.q_encoder or saved_encoders.get("target")
    if saved_encoders and (args.encoder != saved_encoders["subtasks"] or
                           (args.q_encoder or args.encoder) != saved_encoders["target"]):
        p.error("Evaluation encoders must match the checkpoint's training encoders")
    model = IntentRecoveryModel(ModelConfig.from_checkpoint(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    detector = load_detector(ckpt, device)
    if args.detection_threshold is not None and (detector is None or not 0 <= args.detection_threshold <= 1):
        p.error("Threshold override requires a detection checkpoint and a value in [0,1]")
    threshold = ckpt["detector"].get("threshold", 0.5) if detector is not None else 0.5
    if args.detection_threshold is not None:
        threshold = args.detection_threshold

    if args.synthetic:
        items = make_synthetic(256, seed=123)
    else:
        if not args.test:
            raise SystemExit("Provide --test (JSONL) or use --synthetic.")
        items = get_embedded(args.test, args.encoder, args.q_encoder, device)
    if detector is not None:
        if args.synthetic:
            p.error("Detection evaluation requires labeled real data")
        items = attach_labels(items, load_jsonl(args.test))
    if args.out and Path(args.out).resolve() in {Path(p).resolve() for p in (args.ckpt, args.test) if p}:
        p.error("Output must not overwrite checkpoint or input data")

    loader = DataLoader(EmbeddedDataset(items), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    run = init_tracking(args, {"model": ckpt["config"], "args": vars(args)}, "eval")
    try:
        metrics = evaluate(model, loader, device, detector=detector, detection_threshold=threshold)
        print(json.dumps(metrics, indent=2))
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if run is not None:
            run.log(metric_log(metrics, "test"))
            run.summary.update(metric_log(metrics, "test"))
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
