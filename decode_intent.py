"""Compare oracle GTR-vector reconstruction with GNN-vector reconstruction."""
import argparse
import json
from pathlib import Path

import torch

from intent_recovery.data import SentenceEncoder, collate, load_jsonl
from intent_recovery.generation import GTR_ENCODER, INVERTER, GTRInverter, prepare_prediction, text_scores, load_decoder_adapter
from intent_recovery.model import IntentRecoveryModel, ModelConfig
from intent_recovery.tracking import add_tracking_args, init_tracking, metric_log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", required=True)
    p.add_argument("--ckpt", help="Omit to run only the oracle encode/decode control")
    p.add_argument("--out", default="runs/decode/predictions.jsonl")
    p.add_argument("--limit", type=int, default=10, help="First N examples; 0 uses the entire file")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=0, help="0: inverter only; >0: iterative corrector")
    add_tracking_args(p)
    args = p.parse_args()
    if args.limit < 0 or args.batch_size <= 0 or args.steps < 0:
        p.error("limit/steps must be nonnegative and batch-size positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_jsonl(args.test)
    if args.limit:
        records = records[:args.limit]
    if not records:
        p.error("No examples to decode")
    model = None
    checkpoint = None
    adapter = None
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if checkpoint.get("encoders", {}).get("target") != GTR_ENCODER:
            p.error("Checkpoint must be retrained with --q-encoder vec2text/gtr-base; MiniLM vectors are incompatible")
        prepare_prediction(torch.ones(1, 1), checkpoint.get("target_mean_norm", float("nan")))
        model = IntentRecoveryModel(ModelConfig.from_checkpoint(checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        adapter = load_decoder_adapter(checkpoint, device)
        if checkpoint.get("decoder_lora") and args.steps:
            p.error("LoRA checkpoints currently require --steps 0; the pretrained corrector is not adapted")
    target_encoder = SentenceEncoder(GTR_ENCODER, device=device, batch_size=args.batch_size)
    input_encoder = None
    if model is not None:
        name = checkpoint["encoders"]["subtasks"]
        input_encoder = target_encoder if name == GTR_ENCODER else SentenceEncoder(name, device=device)
    inverter = GTRInverter(device, args.steps)
    if checkpoint and checkpoint.get("decoder_lora"):
        inverter.load_lora_checkpoint(checkpoint["decoder_lora"])
    out = Path(args.out)
    if out.resolve() == Path(args.test).resolve() or (args.ckpt and out.resolve() == Path(args.ckpt).resolve()):
        p.error("Output must not overwrite the dataset or checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    run = init_tracking(args, vars(args), "decode")
    totals = {}
    truncated = 0
    try:
        with out.open("w", encoding="utf-8") as f, torch.no_grad():
            for start in range(0, len(records), args.batch_size):
                rows = records[start:start + args.batch_size]
                references = [r["Q"] for r in rows]
                # Oracle is a separate diagnostic branch; never supplied to GNN decoding.
                zq = target_encoder.encode(references)
                # Adapter-free original decoder control, even after decoder post-training.
                oracle = inverter.decode_oracle(zq) if checkpoint and checkpoint.get("decoder_lora") else inverter.decode(zq)
                predicted = None
                norms = None
                if model is not None:
                    flat = input_encoder.encode([s for r in rows for s in r["S"]])
                    items, ptr = [], 0
                    for r in rows:
                        n = len(r["S"])
                        items.append({"H": flat[ptr:ptr+n], "z_q": torch.zeros(model.cfg.target_dim)})
                        ptr += n
                    batch = collate(items)
                    zg = model(batch["H"].to(device), batch["mask"].to(device))["z_G"]
                    norms = zg.norm(dim=-1).tolist()
                    decoder_input = prepare_prediction(zg, checkpoint["target_mean_norm"])
                    if adapter is not None:
                        decoder_input = adapter(decoder_input)
                    predicted = inverter.decode(decoder_input)
                lengths = [len(t) for t in target_encoder.tokenizer(references, truncation=False)["input_ids"]]
                for i, r in enumerate(rows):
                    entry = {"index": start+i, "Q": r["Q"], "oracle_text": oracle[i],
                             "oracle_scores": text_scores(oracle[i], r["Q"]),
                             "target_tokens": lengths[i], "target_truncated": lengths[i] > 128}
                    truncated += entry["target_truncated"]
                    for key in ("id", "label", "source", "canary"):
                        if key in r:
                            entry[key] = r[key]
                    if predicted is not None:
                        entry.update(predicted_text=predicted[i], raw_prediction_norm=norms[i],
                                     predicted_scores=text_scores(predicted[i], r["Q"]))
                    for branch in ("oracle", "predicted"):
                        for key, value in entry.get(branch + "_scores", {}).items():
                            name = branch + "/" + key
                            totals[name] = totals.get(name, 0) + value
                    f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"Decoded {min(start+args.batch_size, len(records))}/{len(records)}")
        metrics = {k: v / len(records) for k, v in totals.items()}
        metrics.update(num_samples=len(records), target_truncated_fraction=truncated / len(records))
        summary = {"metrics": metrics, "args": vars(args), "target_encoder": GTR_ENCODER,
                   "inverter": INVERTER,
                   "corrector": "jxm/gtr__nq__32__correct" if args.steps else None,
                   "training_target_mean_norm": checkpoint["target_mean_norm"] if checkpoint else None,
                   "decoder_adapter": adapter.config if adapter is not None else None,
                   "decoder_lora": checkpoint["decoder_lora"]["config"] if checkpoint and checkpoint.get("decoder_lora") else None,
                   "prediction_norm": "training mean target norm", "input_max_tokens": 128,
                   "warning": "Lexical scores are not intent-equivalence judgments; inspect generated text."}
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        if run is not None:
            run.log(metric_log(metrics, "decode"))
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
