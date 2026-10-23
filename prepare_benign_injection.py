"""Add N benign-source subtasks to each group, without changing Q or label.

Each split supplies its own donor pool. Original subtask order is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def normalize(text):
    return " ".join(text.casefold().split())


def inject_records(records, count=25, seed=42, split="train", placement="interleave"):
    if count < 0 or placement not in ("interleave", "prepend", "append"):
        raise ValueError("Invalid injection count or placement")
    pool = []
    for index, row in enumerate(records):
        if row.get("label") not in ("harmful", "benign"):
            raise ValueError(f"Missing or invalid label in row {index}")
        if not isinstance(row.get("Q"), str) or not isinstance(row.get("S"), list) or not row["S"]:
            raise ValueError(f"Invalid Q/S in row {index}")
        for text in row["S"]:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid subtask in row {index}")
            if row["label"] == "benign":
                pool.append((text, index, row.get("id", str(index))))
    result = []
    for index, row in enumerate(records):
        excluded = {normalize(s) for s in row["S"]}
        candidates = {}
        for text, donor_index, donor_id in pool:
            key = normalize(text)
            if donor_index != index and key not in excluded:
                candidates.setdefault(key, (text, donor_id))
        if len(candidates) < count:
            raise ValueError(f"Row {index}: need {count} distinct external benign subtasks, found {len(candidates)}")
        digest = hashlib.sha256(f"{seed}:{split}:{index}:{row.get('id', '')}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        chosen = rng.sample(list(candidates.values()), count)
        n, total = len(row["S"]), len(row["S"]) + count
        if placement == "prepend":
            injected_positions = set(range(count))
        elif placement == "append":
            injected_positions = set(range(n, total))
        else:
            injected_positions = set(rng.sample(range(total), count))
        originals = iter(row["S"])
        extras = iter(chosen)
        mixed, original_positions, donors = [], [], []
        for position in range(total):
            if position in injected_positions:
                text, donor_id = next(extras)
                mixed.append(text)
                donors.append({"position": position, "donor_id": donor_id})
            else:
                mixed.append(next(originals))
                original_positions.append(position)
        output = dict(row, S=mixed)
        # Source index conventions are not assumed; stale positions must not be used.
        source_harmful_index = output.pop("harmful_index", None)
        output["harmful_index"] = None
        output["benign_injection"] = {
            "count": count, "seed": seed, "split": split, "placement": placement,
            "position_convention": "zero_based", "original_positions": original_positions,
            "original_harmful_index": source_harmful_index, "donors": donors}
        result.append(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="data/decomposedharm_qa")
    p.add_argument("--out", default="data/benign_injection_25")
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--placement", choices=["interleave", "prepend", "append"], default="interleave")
    args = p.parse_args()
    source, out = Path(args.input_dir), Path(args.out)
    if source.resolve() == out.resolve():
        p.error("Output must be separate from clean inputs")
    prepared, manifest = {}, {"args": vars(args), "splits": {}}
    for split in ("train", "validation", "test"):
        path = source / f"{split}.jsonl"
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            p.error(f"Empty split: {split}")
        prepared[split] = inject_records(rows, args.count, args.seed, split, args.placement)
        manifest["splits"][split] = {
            "rows": len(rows), "source_sha256": hashlib.sha256(content).hexdigest(),
            "benign_donor_groups": sum(r["label"] == "benign" for r in rows),
            "mean_original_subtasks": sum(len(r["S"]) for r in rows) / len(rows),
            "mean_augmented_subtasks": sum(len(r["S"]) for r in prepared[split]) / len(rows)}
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{split}: {len(rows)} groups; +{args.count} benign subtasks/group")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
