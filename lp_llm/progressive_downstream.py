"""Independent, provenance-checked downstream evaluation of completed LP-LLM models."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from .progressive_lp import BRANCHES, _files
from .progressive_scaled import layers, materialize, scaled_identity
from .progressive_storage_feasibility import _checkpoint, _verify_source
from .storage import atomic_json, sha256
from .progressive_scaled_aggregate import aggregate as perplexity_matrix

HARNESS_SHA = "d021bf846218f1bb3bdc0603864789329476d464"
TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande")
TASK_FILES = {"arc_easy": "arc/arc_easy.yaml", "arc_challenge": "arc/arc_challenge.yaml",
              "hellaswag": "hellaswag/hellaswag.yaml", "piqa": "piqa/piqa.yaml",
              "winogrande": "winogrande/default.yaml"}
DATASETS = {"arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
            "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
            "hellaswag": ("Rowan/hellaswag", None, "validation"),
            "piqa": ("baber/piqa", None, "validation"),
            "winogrande": ("winogrande", "winogrande_xl", "validation")}
MODELS = ("original_fp16",) + tuple(
    f"{branch}_{suffix}" for branch in ("s01", "r02")
    for suffix in ("w2", "lp4", "lp6", "lp8", "w4", "w6", "w8"))
PILOTS = ("original_fp16", "s01_lp6", "r02_w6")


def _revision(checkout: Path, expected: str) -> None:
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != expected:
        raise ValueError(f"Pinned checkout differs: {checkout}: {actual}")


def _versions() -> dict:
    names = ("torch", "transformers", "accelerate", "datasets", "safetensors",
             "tokenizers", "numpy", "lm_eval")
    return {name: importlib.metadata.version(name) for name in names}


def _cgroup_snapshot() -> dict:
    base = Path("/sys/fs/cgroup")
    stat = base / "memory.stat"
    if not stat.is_file():
        return {"available": False}
    values = {}
    for line in stat.read_text().splitlines():
        name, count = line.split()
        if name in ("anon", "file", "shmem", "inactive_file", "active_file"):
            values[name + "_bytes"] = int(count)
    for name in ("memory.peak", "memory.max"):
        path = base / name
        values[name.replace(".", "_") + "_bytes"] = path.read_text().strip() if path.is_file() else None
    return {"available": True, **values}


def _task_configs(checkout: Path) -> dict:
    result = {}
    for task, relative in TASK_FILES.items():
        path = checkout / "lm_eval/tasks" / relative
        if not path.is_file():
            raise ValueError(f"Pinned harness task missing: {task}")
        content = path.read_text()
        if f"task: {task}" not in content:
            raise ValueError(f"Pinned harness task ID differs: {task}")
        result[task] = {"yaml": relative, "sha256": sha256(path),
                        "dataset": DATASETS[task], "metric": "acc",
                        "normalized_metric": None if task == "winogrande" else "acc_norm"}
    return result


def _manifest(path: Path, code_sha: str, checkout: Path, *, create: bool) -> dict:
    _revision(checkout, HARNESS_SHA)
    expected = {"code_sha": code_sha, "harness_sha": HARNESS_SHA,
                "tasks": _task_configs(checkout), "versions": _versions(),
                "protocol": {"model": "hf", "dtype": "float16", "num_fewshot": 0,
                             "batch_size": 1, "apply_chat_template": False,
                             "limit": None, "seeds": [0, 1234, 1234, 1234]}}
    if path.is_file():
        saved = json.loads(path.read_text())
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"Downstream manifest changed at {key}")
        if (json.loads((path.parent / "environment.json").read_text()) != saved["versions"] or
                json.loads((path.parent / "task_manifest.json").read_text()) !=
                {"tasks": saved["tasks"], "datasets": saved["datasets"]}):
            raise ValueError("Downstream environment or task manifest changed")
        return saved
    if not create:
        raise FileNotFoundError(f"CPU downstream preflight required: {path}")
    from datasets import load_dataset
    datasets = {}
    for task, (dataset, config, split) in DATASETS.items():
        data = load_dataset(dataset, config, split=split)
        datasets[task] = {"path": dataset, "config": config, "split": split,
                          "fingerprint": data._fingerprint, "samples": len(data),
                          "dataset_info_version": str(data.info.version),
                          "download_checksums": data.info.download_checksums}
    expected["datasets"] = datasets
    atomic_json(path, expected)
    atomic_json(path.parent / "environment.json", expected["versions"])
    atomic_json(path.parent / "task_manifest.json",
                {"tasks": expected["tasks"], "datasets": datasets})
    return expected


def _verify_datasets(manifest: dict) -> None:
    from datasets import load_dataset
    import fcntl
    lock_path = Path(os.environ["HF_HOME"]) / ".downstream-cache-read.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task, recorded in manifest["datasets"].items():
            data = load_dataset(recorded["path"], recorded["config"], split=recorded["split"])
            if (data._fingerprint != recorded["fingerprint"] or len(data) != recorded["samples"] or
                    str(data.info.version) != recorded["dataset_info_version"] or
                    data.info.download_checksums != recorded["download_checksums"]):
                raise ValueError(f"Cached benchmark changed: {task}")


def _historical(root: Path, model: str, source: Path) -> tuple[Path | None, dict]:
    if model == "original_fp16":
        return source, {"source_identity": _files(source)}
    branch, suffix = model.split("_", 1)
    if suffix.startswith("lp"):
        depth = int(suffix[2:]) // 2 - 1
        base = root / f"quantizer_followups/{BRANCHES[branch]}/export"
        lp = root / f"progressive_llm/scaled_signed_v1/{branch}"
        ident = scaled_identity(branch, source, base)
        prior = layers(branch, depth, source, base, lp, ident)
        record = root / f"progressive_llm/scaled_signed_v1/{branch}/evaluation/depth{depth}/full.json"
        full = json.loads(record.read_text())
        if (full.get("status") != "completed_validated" or full.get("identity") != ident or
                full.get("depth") != depth or len(prior) != depth or
                sha256(record.parent / "evaluation.log") != full.get("evaluation_log_sha256")):
            raise ValueError(f"Historical LP evaluation differs: {model}")
        return None, {"historical_report": str(record), "report_sha256": sha256(record),
                      "historical_materialization": full["materialization"],
                      "lp_identity": ident, "depth": depth, "branch": branch,
                      "base": str(base), "lp_root": str(lp)}
    bits = int(suffix[1:])
    item = _checkpoint(root, branch, bits)
    _verify_source(source, item)
    if bits == 4 and branch == "s01":
        full = json.loads(Path(item["full_json"]).read_text())
        audit = json.loads(Path(item["completion_json"]).read_text())
        seal = json.loads(Path(item["completion_json"]).with_name("audit.json.sha256.json").read_text())
        if seal["audit_sha256"] != sha256(Path(item["completion_json"])) or \
                audit["source_identity"] != _files(source):
            raise ValueError("Recovered W4 audit/source changed")
        view = root / "progressive_llm/scaled_signed_v1/control_recovery/s01_w4/w4_view"
        from .progressive_control_recovery import _view
        if _view(view, source, audit) != full["view"]:
            raise ValueError("Recovered W4 evaluation view differs")
        return view, item
    return Path(item["checkpoint"]), item


def registry(source: Path, root: Path) -> dict:
    return {model: _historical(root, model, source) for model in MODELS}


def _materialize_lp(source: Path, model: str, provenance: dict, target: Path) -> Path:
    if shutil.disk_usage(target.parent).free < 18 * 1024**3:
        raise ValueError("Insufficient node-local scratch for LP checkpoint")
    report = materialize(provenance["branch"], provenance["depth"], source,
                         Path(provenance["base"]), Path(provenance["lp_root"]), target)
    historical = provenance["historical_materialization"]
    if (report["shard_sha256"] != historical["shard_sha256"] or
            report["tensor_count"] != historical["tensor_count"] or
            report["projection_count"] != historical["projection_count"]):
        raise ValueError(f"LP materialization differs from evaluated weights: {model}")
    return target


def _metrics(raw: dict, manifest: dict, tasks: tuple[str, ...]) -> dict:
    results, configs, samples = raw.get("results", {}), raw.get("configs", {}), raw.get("n-samples", {})
    output = {}
    for task in tasks:
        row = results.get(task, {})
        acc = row.get("acc,none")
        norm = row.get("acc_norm,none")
        if not isinstance(acc, (int, float)) or not math.isfinite(acc):
            raise ValueError(f"Missing harness accuracy: {task}")
        expected = manifest["datasets"][task]
        if samples.get(task, {}).get("effective") != expected["samples"]:
            raise ValueError(f"Harness evaluated wrong sample count: {task}")
        output[task] = {"acc": acc, "acc_stderr": row.get("acc_stderr,none"),
                        "acc_norm": norm, "acc_norm_stderr": row.get("acc_norm_stderr,none"),
                        "samples": samples[task], "task_config": configs.get(task),
                        "raw_metrics": row}
    return output


def evaluate(source: Path, root: Path, output_root: Path, model: str, manifest: dict,
             *, pilot: bool, scratch: Path) -> dict:
    if model not in MODELS or pilot and model not in PILOTS:
        raise ValueError("Unknown or unsupported pilot model")
    target = output_root / ("pilot" if pilot else "models") / model
    full_path = target / "full.json"
    if full_path.exists():
        old = json.loads(full_path.read_text())
        if old.get("manifest_sha256") != sha256(output_root / "manifest.json") or \
                old.get("status") != "completed_validated":
            raise ValueError("Completed downstream evaluation identity differs")
        return old
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Incomplete downstream result requires inspection")
    checkpoint, provenance = _historical(root, model, source)
    owned = scratch / f"downstream-{model}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    if owned.exists():
        raise FileExistsError("Job scratch already exists")
    owned.mkdir(parents=True)
    target.mkdir(parents=True)
    try:
        if checkpoint is None:
            checkpoint = _materialize_lp(source, model, provenance, owned / "checkpoint")
        from lm_eval import simple_evaluate
        from lm_eval.utils import handle_non_serializable
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA GPU unavailable")
        tasks = TASKS[:2] if pilot else TASKS
        args = f"pretrained={checkpoint},dtype=float16,device=cuda:0,parallelize=False"
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        raw = simple_evaluate(model="hf", model_args=args, tasks=list(tasks),
                              num_fewshot=0, batch_size=1, limit=8 if pilot else None,
                              random_seed=0, numpy_random_seed=1234,
                              torch_random_seed=1234, fewshot_random_seed=1234,
                              log_samples=False)
        raw = json.loads(json.dumps(raw, default=handle_non_serializable))
        if pilot:
            # Pilots intentionally use eight examples per task, never full-split counts.
            metrics = {task: raw["results"][task] for task in tasks}
        else:
            metrics = _metrics(raw, manifest, tasks)
        raw_path = target / "harness.json"
        atomic_json(raw_path, raw)
        record = {"status": "completed_validated", "scientific_status": "completed_validated",
                  "resource_status": "measured_separately", "model": model, "pilot": pilot,
                  "code_sha": manifest["code_sha"], "manifest_sha256": sha256(output_root / "manifest.json"),
                  "checkpoint": str(checkpoint), "provenance": provenance,
                  "tasks": metrics, "harness_sha": HARNESS_SHA,
                  "settings": {"model": "hf", "model_args": args, "num_fewshot": 0,
                               "batch_size": 1, "limit": 8 if pilot else None},
                  "harness_sha256": sha256(raw_path), "seconds": time.monotonic() - started,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "cgroup_snapshot_after_evaluation": _cgroup_snapshot()}
        atomic_json(full_path, record)
        return record
    finally:
        if owned.exists() and (owned / "checkpoint").exists() and full_path.exists():
            shutil.rmtree(owned)


def aggregate(root: Path, output_root: Path) -> dict:
    ppl = {row["id"]: row.get("perplexity") for row in perplexity_matrix(root)["rows"]}
    rows = []
    for model in MODELS:
        path = output_root / "models" / model / "full.json"
        row = {"model": model, "bits": 16 if model == "original_fp16" else
               int(model.rsplit("w", 1)[-1]) if "_w" in model else int(model.rsplit("lp", 1)[-1]),
               "wikitext_ppl": ppl.get(model), "status": "missing", "mean_acc": None}
        if path.is_file():
            full = json.loads(path.read_text())
            raw = path.parent / "harness.json"
            if (full.get("status") != "completed_validated" or full.get("pilot") is not False or
                    full.get("manifest_sha256") != sha256(output_root / "manifest.json") or
                    not raw.is_file() or sha256(raw) != full.get("harness_sha256")):
                row["status"] = "invalid"
            elif set(full.get("tasks", {})) != set(TASKS):
                row["status"] = "invalid"
            else:
                row["status"] = "completed_validated"
                for task in TASKS:
                    row[task] = full["tasks"][task]["acc"]
                    row[f"{task}_acc_norm"] = full["tasks"][task]["acc_norm"]
                row["mean_acc"] = sum(row[task] for task in TASKS) / len(TASKS)
        rows.append(row)
    by_id = {row["model"]: row for row in rows}
    differences = {}
    reference = by_id["original_fp16"].get("mean_acc")
    for branch in ("s01", "r02"):
        for bits in (4, 6, 8):
            lp, mono = by_id[f"{branch}_lp{bits}"], by_id[f"{branch}_w{bits}"]
            differences[f"{branch}_{bits}"] = {
                "lp_minus_monolithic_mean_acc": lp["mean_acc"] - mono["mean_acc"]
                if lp["mean_acc"] is not None and mono["mean_acc"] is not None else None,
                "lp_reference_gap_recovered": (lp["mean_acc"] - by_id[f"{branch}_w2"]["mean_acc"]) /
                    (reference - by_id[f"{branch}_w2"]["mean_acc"])
                if all(v is not None for v in (lp["mean_acc"], reference,
                    by_id[f"{branch}_w2"]["mean_acc"])) and
                    reference != by_id[f"{branch}_w2"]["mean_acc"] else None}
    return {"status": "reported_results_only", "metric": "unweighted_mean_acc",
            "rows": rows, "matched_precision_comparisons": differences,
            "storage_note": "Analytical packed storage is not measured physical storage"}


def write_aggregate(root: Path, output_root: Path) -> dict:
    report = aggregate(root, output_root)
    target = output_root / "aggregate.json"
    if any((output_root / f"aggregate.{ext}").exists() for ext in ("json", "csv", "md")) or \
            (output_root / "aggregate_normalized.csv").exists():
        raise FileExistsError("Aggregate output already exists")
    atomic_json(target, report)
    fields = ("model", "bits", "wikitext_ppl", *TASKS, "mean_acc", "status")
    with (output_root / "aggregate.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(report["rows"])
    with (output_root / "aggregate_normalized.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", *TASKS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({"model": row["model"], **{task: row.get(f"{task}_acc_norm")
                          for task in TASKS}} for row in report["rows"])
    lines = ["# Zero-shot downstream accuracy", "", "Primary metric: unweighted mean of five raw accuracies.",
             "", "| Model | Bits | W2 PPL | ARC-E | ARC-C | HellaSwag | PIQA | WinoGrande | Mean | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["rows"]:
        values = [row["model"], row["bits"], row["wikitext_ppl"],
                  *[row.get(task) for task in TASKS], row["mean_acc"], row["status"]]
        lines.append("| " + " | ".join("" if v is None else str(v) for v in values) + " |")
    (output_root / "aggregate.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("preflight", "evaluate", "aggregate"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--harness", type=Path, required=True)
    p.add_argument("--code-sha", required=True)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--scratch", type=Path)
    a = p.parse_args()
    manifest_path = a.output_root / "manifest.json"
    manifest = _manifest(manifest_path, a.code_sha, a.harness, create=a.action == "preflight")
    if a.action == "preflight":
        registry(a.source, a.result_root)
        print(json.dumps({"models": len(MODELS), "tasks": list(TASKS)}))
    elif a.action == "aggregate":
        print(json.dumps({"rows": len(write_aggregate(a.result_root, a.output_root)["rows"])}))
    else:
        if a.model is None or a.scratch is None:
            p.error("evaluate requires --model and --scratch")
        _verify_datasets(manifest)
        result = evaluate(a.source, a.result_root, a.output_root, a.model, manifest,
                          pilot=a.pilot, scratch=a.scratch)
        print(json.dumps({"model": a.model, "status": result["status"]}))


if __name__ == "__main__":
    main()
