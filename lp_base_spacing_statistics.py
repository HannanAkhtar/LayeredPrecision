"""Base quantization step and centroid-spacing distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analyze_mse_layer_sensitivity import stage_of
from evaluate_lp_base_enhancement import FAMILIES, base_key
from evaluate_lp_quantizer_adaptivity import load_torch, save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    rows = []
    for family in FAMILIES:
        artifacts = load_torch(out / "codes" / (base_key(family) + ".pt"))
        groups = {stage: [] for stage in ("all", "conv1", "layer1", "layer2", "layer3", "layer4", "fc")}
        for name, params in artifacts.items():
            if family == "N":
                values = (params["centers"][:, 1:]-params["centers"][:, :-1]).flatten().numpy()
            elif family == "L":
                scale = params["scale"].reshape(-1).numpy()
                ratio = np.sqrt(2.0)
                positive = np.array([(ratio-1)*ratio**i for i in range(15)])
                values = (scale[:, None]*positive[None,:]).flatten()
            else:
                values = params["scale"].reshape(-1).numpy()
            groups["all"].append(values)
            groups[stage_of(name)].append(values)
        for stage, chunks in groups.items():
            values = np.concatenate(chunks).astype(np.float64)
            rows.append({"base": family, "stage": stage,
                         "definition": "adjacent learned centers" if family=="N" else
                                       "adjacent positive log levels" if family=="L" else "per-channel uniform step",
                         "count": len(values), "minimum": float(values.min()),
                         "p10": float(np.quantile(values,.1)), "median": float(np.median(values)),
                         "p90": float(np.quantile(values,.9)), "p99": float(np.quantile(values,.99)),
                         "maximum": float(values.max()),
                         "zero_fraction": float(np.mean(values==0))})
    save_csv(out / "base_step_distribution.csv", rows)
    print(f"Wrote {len(rows)} base spacing summaries", flush=True)


if __name__ == "__main__":
    main()
