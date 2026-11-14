"""Read-only exact-grid feasibility audit of evaluated fake-QDQ checkpoints.

This is an analytical audit, not a packed checkpoint writer. A failed bounded
search is deliberately not reported as a proof of non-representability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

from .progressive_lp import _files
from .rrq_pr2308 import source_identity
from .rrq_pr2308_compare import _eligible, _weight_map
from .storage import atomic_json, sha256

TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"
TOKEN_COUNT = 298862
TOKEN_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
GROUP = 128
BRANCHES = ("s01", "r02")
BITS = (2, 4, 6, 8)
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I8": 1, "U8": 1,
               "I16": 2, "I32": 4, "I64": 8}


def _result(root: Path, branch: str, bits: int) -> tuple[Path, Path, Path]:
    if branch == "s01" and bits == 4:
        record = root / "progressive_llm/scaled_signed_v1/control_recovery/s01_w4/evaluation/full.json"
        audit = record.parents[1] / "audit.json"
        return record, audit, root / "progressive_llm/controls/s01_w4"
    if bits == 2:
        base = root / f"quantizer_followups/{branch}_fp16_load_repair"
        return base / "full.json", base / "encoding_complete.json", base
    base = root / f"progressive_llm/controls/{branch}_w{bits}"
    return base / "full.json", base / "encoding_complete.json", base


def _checkpoint(root: Path, branch: str, bits: int) -> dict:
    full_path, marker_path, job = _result(root, branch, bits)
    full = json.loads(full_path.read_text())
    marker = json.loads(marker_path.read_text())
    if full.get("status") != "completed_validated" or marker.get("status") != "completed_validated":
        raise ValueError(f"{branch} W{bits} lacks validated evaluation or completion")
    if (full.get("corpus", {}).get("token_sha256") != TOKEN_SHA or
            full.get("corpus", {}).get("scored_tokens") != TOKEN_COUNT or
            sha256(full_path.parent / "evaluation.log") != full.get("evaluation_log_sha256")):
        raise ValueError(f"{branch} W{bits} evaluation provenance differs")
    checked = full["checkpoint_validation"]
    if branch == "s01" and bits == 4:
        seal = json.loads(marker_path.with_name("audit.json.sha256.json").read_text())
        if (seal != {"code_sha": marker.get("code_sha"), "audit_sha256": sha256(marker_path)} or
                full.get("audit_sha256") != sha256(marker_path) or
                full.get("auditing_code_sha") != marker.get("code_sha") or
                checked != marker.get("checkpoint_validation")):
            raise ValueError("S01 W4 sealed CPU audit differs from evaluated recovery")
        path = Path(full["view"]["checkpoint"])
        if path != Path(checked["checkpoint"]["resolved_checkpoint_path"]):
            raise ValueError("S01 W4 evaluated checkpoint path differs")
    else:
        if (checked != marker.get("checkpoint_validation") or
                full.get("identity") != marker.get("identity")):
            raise ValueError(f"{branch} W{bits} completion differs from evaluation")
        recorded_path = checked.get("checkpoint", checked)
        path = Path(recorded_path.get("resolved_checkpoint_path", job / "export"))
    if not path.is_dir():
        raise ValueError(f"Checkpoint missing: {path}")
    resolved_path = job / "resolved_config.json"
    if bits == 2:
        resolved_path = root / f"quantizer_followups/{branch}/resolved_config.json"
    resolved = json.loads(resolved_path.read_text()) if resolved_path.is_file() else None
    recorded = checked.get("checkpoint", checked)
    for name, digest in recorded.get("shard_sha256", {}).items():
        if sha256(path / name) != digest:
            raise ValueError(f"Checkpoint shard changed: {name}")
    for key, name in (("config_sha256", "config.json"), ("index_sha256", "model.safetensors.index.json")):
        if key in recorded and sha256(path / name) != recorded[key]:
            raise ValueError(f"Checkpoint {name} changed")
    return {"id": f"{branch}_w{bits}", "branch": branch, "bits": bits,
            "family": "signround_v2_symmetric" if branch == "s01" else "optimized_rtn_neuqi_asymmetric",
            "checkpoint": str(path.resolve()), "full_json": str(full_path.resolve()),
            "full_sha256": sha256(full_path), "completion_json": str(marker_path.resolve()),
            "completion_sha256": sha256(marker_path), "producing_sha":
                (full.get("identity") or {}).get("code_sha") or full.get("original_producer_sha") or
                (marker.get("identity") or {}).get("original_run_identity", {}).get("code_sha"),
            "evaluation_sha": full.get("code_sha") or (full.get("identity") or {}).get("code_sha"),
            "audit_sha": full.get("auditing_code_sha"), "validated_evaluation": True,
            "recorded_configuration": (full.get("identity") or marker.get("identity") or {}),
            "resolved_quantizer_configuration": resolved,
            "resolved_configuration_sha256": sha256(resolved_path) if resolved is not None else None,
            "checkpoint_validation": checked}


def _verify_source(source: Path, item: dict) -> None:
    """Use the producing experiment's source identity schema, never cross-compare schemas."""
    full = json.loads(Path(item["full_json"]).read_text())
    branch, bits = item["branch"], item["bits"]
    if branch == "s01" and bits == 4:
        marker = json.loads(Path(item["completion_json"]).read_text())
        recorded_source = marker.get("source_identity")
        current = _files(source)
    elif bits == 2:
        recorded_source = (full.get("identity") or {}).get("original_run_identity", {}).get("source")
        current = source_identity(source)
    else:
        recorded_source = (full.get("identity") or {}).get("source")
        current = _files(source)
    if not isinstance(recorded_source, dict) or recorded_source != current:
        raise ValueError(f"{item['id']} source identity differs")


def inventory(source: Path, root: Path, branch: str, bits: int) -> dict:
    item = _checkpoint(root, branch, bits)
    _verify_source(source, item)
    path = Path(item["checkpoint"])
    mapping, source_map = _weight_map(path), _weight_map(source)
    if set(mapping) != set(source_map):
        raise ValueError(f"{item['id']} tensor inventory differs from source")
    source_shapes = {}
    for shard in sorted(set(source_map.values())):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source_shapes[name] = list(handle.get_slice(name).get_shape())
    tensors = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            for name in sorted(n for n, file in mapping.items() if file == shard):
                sl = handle.get_slice(name)
                shape, dtype = list(sl.get_shape()), sl.get_dtype()
                if shape != source_shapes[name]:
                    raise ValueError(f"Tensor shape differs from original: {name}")
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"Unsupported tensor dtype: {name} {dtype}")
                if _eligible(name) and (len(shape) != 2 or dtype not in ("F16", "BF16")):
                    raise ValueError(f"Eligible tensor must be FP16/BF16 matrix: {name}")
                tensors[name] = {"shape": shape, "dtype": dtype, "shard": shard,
                                 "eligible": _eligible(name), "elements": math.prod(shape),
                                 "payload_bytes": math.prod(shape) * DTYPE_BYTES[dtype]}
    eligible = [t for t in tensors.values() if t["eligible"]]
    excluded = [t for t in tensors.values() if not t["eligible"]]
    if len(eligible) != 252 or len(excluded) != 147:
        raise ValueError(f"Unexpected projection/exclusion coverage for {item['id']}")
    required_sidecars = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p),
                                   "origin": "checkpoint"} for p in path.iterdir()
                         if p.is_file() and p.suffix != ".safetensors"}
    for name in TOKEN_FILES:
        if name not in required_sidecars:
            token = source / name
            if not token.is_file():
                raise ValueError(f"Required original tokenizer missing: {name}")
            required_sidecars[name] = {"bytes": token.stat().st_size,
                                       "sha256": sha256(token), "origin": "original_source"}
    item.update({"group_size": GROUP, "axis": "input_channels_per_output_row",
                 "tensors": tensors, "tensor_count": len(tensors),
                 "eligible_tensor_count": len(eligible), "excluded_tensor_count": len(excluded),
                 "eligible_elements": sum(t["elements"] for t in eligible),
                 "excluded_elements": sum(t["elements"] for t in excluded),
                 "shard_bytes": {s: (path / s).stat().st_size for s in set(mapping.values())},
                 "checkpoint_file_bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                 "required_sidecars": required_sidecars,
                 "explicit_codes_scales_zeros": {k: [n for n in tensors if k in n]
                     for k in ("qweight", "scales", "qzeros")}})
    return item


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int16)


def _ulp_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    def ordered(value: torch.Tensor) -> torch.Tensor:
        raw = _bits(value).to(torch.int32) & 0xffff
        return torch.where((raw & 0x8000) != 0, 0xffff - raw, raw + 0x8000)
    return int((ordered(left) - ordered(right)).abs().max())


def _candidate_scales(values: torch.Tensor, codes: range, dtype: torch.dtype,
                      limit: int = 256) -> list[float]:
    unique = torch.unique(values.float())
    nonzero = unique[unique != 0]
    if not len(nonzero):
        return [-1.0, 0.0, 1.0]
    candidates = set()
    selected = nonzero[torch.linspace(0, len(nonzero) - 1, min(len(nonzero), 8)).long()]
    anchors = set()
    for value in selected.tolist():
        for code in codes:
            if code == 0:
                continue
            ratio = torch.tensor(value / code, dtype=dtype)
            for neighbor in (ratio, torch.nextafter(ratio, torch.tensor(float("-inf"), dtype=dtype)),
                             torch.nextafter(ratio, torch.tensor(float("inf"), dtype=dtype))):
                if torch.isfinite(neighbor) and neighbor != 0:
                    candidates.add(float(neighbor))
                    if abs(code) == 1:
                        anchors.add(float(neighbor))
    ordered = sorted(candidates, key=lambda number: (abs(number), number))
    if limit == 0:
        return []
    if len(ordered) > limit:
        anchors = sorted(anchors, key=lambda number: (abs(number), number))[:limit]
        remaining = limit - len(anchors)
        if remaining == 0:
            return anchors
        stride = len(ordered) / remaining
        return anchors + [ordered[int(i * stride)] for i in range(remaining)]
    return ordered


def recover_group(values: torch.Tensor, bits: int, branch: str, scale_dtype: str,
                  *, search_limit: int = 256, pre_export_fp16: bool = False,
                  candidate_scales: list[float] | None = None) -> dict:
    if values.dtype not in (torch.float16, torch.bfloat16) or values.ndim != 1 or not 0 < len(values) <= GROUP:
        raise ValueError("Expected one nonempty FP16/BF16 input-channel group")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite evaluated weight")
    if branch not in BRANCHES or bits not in BITS or scale_dtype not in ("fp16", "fp32"):
        raise ValueError("Unsupported candidate format")
    signed = branch == "s01"
    domain = range(-(1 << (bits - 1)), 1 << (bits - 1)) if signed else range(1 << bits)
    unique_bits = int(torch.unique(_bits(values)).numel())
    common = {"distinct_stored_bit_patterns": unique_bits, "stored_dtype": str(values.dtype),
              "minimum": float(values.min()),
              "maximum": float(values.max()), "zeros": int((values == 0).sum()),
              "negative": int((values < 0).sum()), "positive": int((values > 0).sum()),
              "code_range": [domain.start, domain.stop - 1], "scale_format": scale_dtype,
              "decoder": ("scale-dtype multiply, FP16 QDQ cast, stored-dtype export cast"
                          if pre_export_fp16 else "scale-dtype multiply then stored-dtype cast") ,
              "signed_zero_present": bool(((_bits(values) == -32768)).any()),
              "search_limit": search_limit}
    if unique_bits > 1 << bits:
        return {**common, "outcome": "not_representable_proven", "reason": "too_many_distinct_bit_patterns"}
    dtype = torch.float16 if scale_dtype == "fp16" else torch.float32
    # R02 is evaluated with integer zero point; all possible offsets are included.
    effective_domain = domain if signed else range(-(1 << bits) + 1, 1 << bits)
    scaled_codes = effective_domain
    candidates = (candidate_scales[:search_limit] if candidate_scales is not None else
                  _candidate_scales(values, scaled_codes, dtype, search_limit))
    tested = 0
    target = _bits(values)
    best = None
    for scale in candidates:
        if not signed and scale <= 0:
            continue
        effective = torch.tensor(list(effective_domain), dtype=dtype)
        product = effective * torch.tensor(scale, dtype=dtype)
        grid = product.to(torch.float16).to(values.dtype) if pre_export_fp16 else product.to(values.dtype)
        eq = _bits(grid)[:, None] == target[None, :]
        matches = eq.any(dim=0)
        positions = eq.to(torch.int8).argmax(dim=0)
        zero = 0
        feasible = bool(matches.all())
        if feasible and not signed:
            last = len(effective_domain) - 1 - eq.flip(0).to(torch.int8).argmax(dim=0)
            first_effective = positions + effective_domain.start
            last_effective = last + effective_domain.start
            maxq = (1 << bits) - 1
            low = max(0, int((-last_effective).max()))
            high = min(maxq, int((maxq - first_effective).min()))
            feasible = low <= high
            if feasible:
                zero = low
                valid = torch.tensor([0 <= q + zero <= maxq for q in effective_domain])
                positions = (eq & valid[:, None]).to(torch.int8).argmax(dim=0)
        chosen = positions + effective_domain.start
        count = int((~matches).sum()) + int(not feasible and bool(matches.all()))
        tested += 1
        if best is None or count < best["mismatching_elements"]:
            decoded = grid[positions]
            error = (decoded.float() - values.float()).abs()
            best = {"mismatching_elements": count, "scale": scale, "zero_point": zero,
                    "max_abs_error": float(error.max()),
                    "mse": float(error.square().mean()),
                    "max_stored_dtype_ulp_difference": _ulp_distance(decoded, values),
                    "numerically_equal": bool(torch.equal(decoded, values)),
                    "codes": (chosen + zero).tolist() if count == 0 else None}
        if count == 0:
            return {**common, **best, "outcome": "exact_constructive",
                    "mismatch_fraction": 0.0,
                    "code_bytes_sha256": hashlib.sha256(bytes(q & 255 for q in best["codes"])).hexdigest(),
                    "candidates_tested": tested}
    return {**common, **(best or {}), "outcome": "not_found_under_search",
            "mismatch_fraction": best["mismatching_elements"] / len(values) if best else None,
            "candidates_tested": tested,
            "reason": "bounded_scale_search_not_a_proof"}


def _positions(shape: list[int], mode: str):
    rows, columns = shape
    groups = math.ceil(columns / GROUP)
    if mode == "full":
        for row in range(rows):
            for group in range(groups):
                yield row, group
    else:
        # Three positions per tensor; row and input positions each cover early/middle/late.
        for row, group in zip((0, rows // 2, rows - 1), (0, groups // 2, groups - 1)):
            yield row, group


def _storage(item: dict) -> dict:
    bits = item["bits"]
    eligible = [t for t in item["tensors"].values() if t["eligible"]]
    groups = sum(t["shape"][0] * math.ceil(t["shape"][1] / GROUP) for t in eligible)
    code_bytes = sum(t["shape"][0] * math.ceil(t["shape"][1] * bits / 8) for t in eligible)
    excluded = sum(t["payload_bytes"] for t in item["tensors"].values() if not t["eligible"])
    sidecars = sum(entry["bytes"] for entry in item.get("required_sidecars", {}).values())
    return {"basis": "analytical_unserialized_payload_estimate", "code_bytes": code_bytes,
            "group_count": groups, "scale_fp16_bytes": 2 * groups,
            "scale_fp32_bytes": 4 * groups,
            "zero_point_uint8_bytes": 0 if item["branch"] == "s01" else groups,
            "excluded_tensor_payload_bytes": excluded, "required_existing_sidecar_bytes": sidecars,
            "framing_bytes": None, "estimated_total_excluding_future_framing_fp16_scale":
                code_bytes + 2 * groups + (0 if item["branch"] == "s01" else groups) + excluded + sidecars,
            "existing_fake_qdq_checkpoint_file_bytes": item["checkpoint_file_bytes"],
            "warning": "Not a measured packed artifact; framing and deployment metadata remain unimplemented"}


def inspect(item: dict, mode: str, group_path: Path | None = None) -> dict:
    if mode == "metadata":
        return {"inventory": item, "storage": _storage(item), "groups": [],
                "scope": "metadata_only_no_exactness_claim"}
    path = Path(item["checkpoint"])
    outcomes = ("exact_constructive", "not_representable_proven", "not_found_under_search", "unknown")
    counts = {fmt: {outcome: 0 for outcome in outcomes} for fmt in ("fp16", "fp32")}
    examples, total, exact_elements = [], 0, 0
    exact_by_format = {"fp16": 0, "fp32": 0, "mixed": 0}
    errors = {"zero": 0, "positive_under_1e-3": 0, "at_least_1e-3": 0, "unavailable": 0}
    if group_path is not None:
        group_path.parent.mkdir(parents=True, exist_ok=True)
    with (group_path.open("x") if group_path is not None else open(os.devnull, "w")) as stream:
        for name, info in sorted(item["tensors"].items()):
            if not info["eligible"]:
                continue
            with safe_open(path / info["shard"], framework="pt", device="cpu") as handle:
                sl = handle.get_slice(name)
                for row, group in _positions(info["shape"], mode):
                    start = group * GROUP
                    values = sl[row:row + 1, start:min(start + GROUP, info["shape"][1])].reshape(-1)
                    tested = {fmt: recover_group(values, item["bits"], item["branch"], fmt)
                              for fmt in ("fp16", "fp32")}
                    for fmt, candidate in tested.items():
                        counts[fmt][candidate["outcome"]] += 1
                        candidate.pop("codes", None)
                    error = min((c["max_abs_error"] for c in tested.values()
                                 if c.get("max_abs_error") is not None), default=None)
                    bucket = ("unavailable" if error is None else "zero" if error == 0 else
                              "positive_under_1e-3" if error < 1e-3 else "at_least_1e-3")
                    errors[bucket] += 1
                    entry = {"model_id": item["id"], "checkpoint_record_sha256": item["full_sha256"],
                             "tensor": name, "shape": info["shape"], "row": row,
                             "group": group, "valid_elements": len(values), "candidates": tested}
                    stream.write(json.dumps(entry, sort_keys=True) + "\n")
                    if len(examples) < 24:
                        examples.append(entry)
                    total += 1
                    for fmt in ("fp16", "fp32"):
                        if tested[fmt]["outcome"] == "exact_constructive":
                            exact_by_format[fmt] += len(values)
                    if any(c["outcome"] == "exact_constructive" for c in tested.values()):
                        exact_elements += len(values)
                        exact_by_format["mixed"] += len(values)
    return {"inventory": item, "storage": _storage(item), "group_file": str(group_path),
            "group_file_sha256": sha256(group_path) if group_path else None,
            "examples": examples, "group_count": total, "exact_elements": exact_elements,
            "counts": counts, "error_histogram": errors,
            "scope": "all_groups" if mode == "full" else "deterministic_sample",
            "exact_certificates": {fmt: mode == "full" and total > 0 and
                exact_by_format[fmt] == item["eligible_elements"]
                for fmt in ("fp16", "fp32", "mixed")}}


def _bar_figure(path: Path, title: str, rows: list[tuple[str, float]],
                unit: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    height = 75 + 40 * len(rows)
    high = max((value for _, value in rows), default=1) or 1
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="30" font-family="sans-serif" font-size="19">{title}</text>']
    for index, (label, value) in enumerate(rows):
        y = 55 + 40 * index
        width = int(490 * value / high)
        lines += [f'<text x="20" y="{y + 16}" font-family="sans-serif" font-size="13">{label}</text>',
                  f'<rect x="230" y="{y}" width="{width}" height="19" fill="#4169a8"/>',
                  f'<text x="{240 + width}" y="{y + 16}" font-family="sans-serif" font-size="13">{value:.4g} {unit}</text>']
    lines.append('</svg>')
    path.write_text("\n".join(lines))


def _figures(report: dict, output: Path) -> None:
    figures = output.with_name(output.stem + ".figures")
    if figures.exists():
        raise FileExistsError(figures)
    figures.mkdir()
    models = report["models"]
    _bar_figure(figures / "exact_fraction.svg", "Constructive exact group fraction (sample or full)",
                [(name, max((model.get("counts", {}).get(fmt, {}).get("exact_constructive", 0)
                             for fmt in ("fp16", "fp32")), default=0) / max(model.get("group_count", 0), 1))
                 for name, model in models.items()], "fraction")
    _bar_figure(figures / "group_errors.svg", "Best candidate error buckets (group counts)",
                [(f"{name}: {bucket}", count) for name, model in models.items()
                 for bucket, count in model.get("error_histogram", {}).items()], "groups")
    _bar_figure(figures / "storage_components.svg", "Analytical bytes by component, not serialized",
                [(f"{name}: {key}", model["storage"][key]) for name, model in models.items()
                 for key in ("code_bytes", "scale_fp16_bytes", "zero_point_uint8_bytes",
                             "excluded_tensor_payload_bytes")], "bytes")
    _bar_figure(figures / "multi_rate.svg", "Conditional multi-rate estimates; framing excluded",
                [(f"{branch}: independent", scenario["independent_models_analytical_lower_bound_bytes"])
                 for branch, scenario in report["multi_rate_scenarios"].items()] +
                [(f"{branch}: LP", scenario["lp_analytical_base_plus_measured_enhancements_bytes"])
                 for branch, scenario in report["multi_rate_scenarios"].items()
                 if scenario["lp_analytical_base_plus_measured_enhancements_bytes"] is not None], "bytes")


def run(source: Path, root: Path, output: Path, mode: str, code_sha: str,
        selected: str | None = None) -> dict:
    if mode not in ("metadata", "sample", "full") or not (len(code_sha) == 40):
        raise ValueError("Invalid audit mode or code SHA")
    if (output.exists() or output.with_suffix(".md").exists() or
            output.with_name(output.stem + ".figures").exists()):
        raise FileExistsError(f"Audit output already exists: {output}")
    models = [(branch, bits) for branch in BRANCHES for bits in BITS
              if selected is None or selected == f"{branch}_w{bits}"]
    if not models:
        raise ValueError("Unknown model ID")
    report = {"status": "completed", "mode": mode, "code_sha": code_sha,
              "source": str(source.resolve()), "models": {},
              "disclaimer": "Sampled exactness is not full-model packed feasibility"}
    for branch, bits in models:
        item = inventory(source, root, branch, bits)
        groups = output.with_name(output.stem + f".{item['id']}.groups.jsonl") if mode != "metadata" else None
        report["models"][item["id"]] = inspect(item, mode, groups)
    report["multi_rate_scenarios"] = {}
    for branch in BRANCHES:
        names = [f"{branch}_w{bits}" for bits in BITS]
        if not all(name in report["models"] for name in names):
            continue
        models_for_branch = [report["models"][name] for name in names]
        independent = sum(model["storage"]["estimated_total_excluding_future_framing_fp16_scale"]
                          for model in models_for_branch)
        lp_depth3 = root / f"progressive_llm/scaled_signed_v1/{branch}/evaluation/depth3/full.json"
        enhancement = None
        if lp_depth3.is_file():
            lp_record = json.loads(lp_depth3.read_text())
            layer_rows = lp_record.get("storage", {}).get("layers", [])
            if lp_record.get("status") == "completed_validated" and len(layer_rows) == 3:
                enhancement = sum(row["code_bytes"] + row["metadata_bytes"] +
                                  row["manifest_bytes"] for row in layer_rows)
                enhancement += lp_record["storage"].get("coefficient_reference_bytes", 0)
        base = models_for_branch[0]["storage"]["estimated_total_excluding_future_framing_fp16_scale"]
        report["multi_rate_scenarios"][branch] = {
            "independent_models_analytical_lower_bound_bytes": independent,
            "shared_excluded_tensors": "not_estimated_without_byte_identity_verification",
            "lp_measured_enhancement_bytes": enhancement,
            "lp_analytical_base_plus_measured_enhancements_bytes":
                base + enhancement if enhancement is not None else None,
            "status": "conditional_estimates_not_measured_packed_storage",
            "warning": "All analytical packed estimates exclude future framing and require full exactness verification"}
    atomic_json(output, report)
    lines = [f"# Storage feasibility: {mode}", "",
             "Analytical bytes below are not measured packed files. Exactness is group-level",
             "only; a sample cannot certify the full checkpoint.", "",
             "| Model | Bits | Family | Existing fake-QDQ bytes | Eligible | Excluded |",
             "|---|---:|---|---:|---:|---:|"]
    for name, model in report["models"].items():
        item, storage = model["inventory"], model["storage"]
        lines.append(f"| {name} | {item['bits']} | {item['family']} | "
                     f"{storage['existing_fake_qdq_checkpoint_file_bytes']} | "
                     f"{item['eligible_tensor_count']} | {item['excluded_tensor_count']} |")
    lines += ["", "| Model | Groups | Exact FP16 scale | Exact FP32 scale | Proven failures FP16/FP32 | Search misses FP16/FP32 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        counts = model.get("counts", {})
        a, b = counts.get("fp16", {}), counts.get("fp32", {})
        lines.append(f"| {name} | {model.get('group_count', 0)} | {a.get('exact_constructive', 0)} | "
                     f"{b.get('exact_constructive', 0)} | "
                     f"{a.get('not_representable_proven', 0)}/{b.get('not_representable_proven', 0)} | "
                     f"{a.get('not_found_under_search', 0)}/{b.get('not_found_under_search', 0)} |")
    lines += ["", "| Model | Declared scale | Zero point | Explicit packed metadata |",
              "|---|---|---|---|"]
    for name, model in report["models"].items():
        item = model["inventory"]
        explicit = item["explicit_codes_scales_zeros"]
        available = any(explicit.values())
        lines.append(f"| {name} | FP16 requested; FP32 alternative tested | "
                     f"{'none (signed codes)' if item['branch'] == 's01' else 'integer uint8 candidate'} | "
                     f"{'present' if available else 'absent in fake-QDQ checkpoint'} |")
    lines += ["", "| Model | Codes | FP16 scales | FP32 scales | Zero points | Excluded payload | Sidecars |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        s = model["storage"]
        lines.append(f"| {name} | {s['code_bytes']} | {s['scale_fp16_bytes']} | "
                     f"{s['scale_fp32_bytes']} | {s['zero_point_uint8_bytes']} | "
                     f"{s['excluded_tensor_payload_bytes']} | {s['required_existing_sidecar_bytes']} |")
    lines += ["", "Shared excluded weights are unverified and are not deducted.",
              "Future serialized framing bytes remain unknown."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    _figures(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("metadata", "sample", "full"), default="sample")
    parser.add_argument("--model", choices=[f"{b}_w{n}" for b in BRANCHES for n in BITS])
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    result = run(args.source, args.result_root, args.output, args.mode,
                 args.code_sha, args.model)
    print(json.dumps({"output": str(args.output), "models": list(result["models"]),
                      "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
