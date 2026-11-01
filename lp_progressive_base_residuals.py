"""Residual evolution for the fifteen prespecified Experiment 5 hierarchies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_mse_layer_sensitivity import eligible_groups, stage_of
from evaluate_imagenet import build_model
from evaluate_lp_base_enhancement import FAMILIES, SCHEMES, base_key, extend
from evaluate_lp_quantizer_adaptivity import load_torch, save_csv
from lp_base_residual_statistics import STANDARD_EDGES, add_hist, summary, tensor_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    model = build_model("resnet50")
    names = {name for group in eligible_groups(model).values() for name in group}
    original = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if name in names}
    parents = []
    routes = []
    for base in FAMILIES:
        for scheme in SCHEMES:
            key = base_key(base)
            for level, family in enumerate(scheme, 1):
                child = extend(key, family)
                routes.append((base, scheme, level, key, child))
                if key not in parents:
                    parents.append(key)
                key = child
    unique = {}
    histograms = {}
    for parent in parents:
        weights = load_torch(out / "decoded_prefixes" / (parent + ".pt"))
        rows = []
        hists = {}
        for i, (name, target) in enumerate(original.items(), 1):
            part, hist = tensor_statistics(name, target-weights[name])
            rows.extend(part)
            stage = stage_of(name)
            hists[stage] = add_hist(hists.get(stage), hist)
            if i % 18 == 0:
                print(f"Residual statistics {parent}: {i}/54", flush=True)
        hists["all"] = None
        for stage in tuple(stage for stage in hists if stage != "all"):
            hists["all"] = add_hist(hists["all"], hists[stage])
        for stage in ("all", "conv1", "layer1", "layer2", "layer3", "layer4", "fc"):
            item = summary(rows, parent, stage)
            hist = hists[stage]
            item["standard_overflow_fraction"] = (hist["standard_below"]+hist["standard_above"])/item["weights"]
            unique[(parent,stage)] = item
        histograms[parent] = hists["all"]["standardized"]
    output = []
    for base, scheme, level, parent, child in routes:
        code = json.loads((out / "codes" / (child + ".json")).read_text())
        for stage in ("all", "conv1", "layer1", "layer2", "layer3", "layer4", "fc"):
            item = {"base": base, "scheme": scheme, "residual": f"R{level}",
                    "parent": parent, "next_prefix": child, **unique[(parent,stage)],
                    "parameter_mse": unique[(parent,stage)]["pooled_rms"]**2,
                    "next_code_entropy": code["conditional_code_entropy"],
                    "next_median_step_or_spacing": code["median_step_or_spacing"]}
            output.append(item)
    save_csv(out / "progressive_residual_statistics.csv", output)
    representative = (("A","AAA"),("A","UUU"),("A","NNN"),("N","UUU"),("U","NNN"))
    centers = .5*(STANDARD_EDGES[:-1]+STANDARD_EDGES[1:])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), sharey=True)
    for base, scheme in representative:
        key = base_key(base)
        for level, family in enumerate(scheme, 1):
            counts = histograms[key]
            axes[level-1].plot(centers, counts/counts.sum()/(STANDARD_EDGES[1]-STANDARD_EDGES[0]),
                               label=f"{base}-{scheme}")
            key = extend(key, family)
    for level, axis in enumerate(axes, 1):
        axis.set_title(f"R{level}")
        axis.set_yscale("log")
        axis.set_xlim(-5,5)
        axis.set_xlabel("Value / per-channel standard deviation")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Parameter-weighted density")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(out / "progressive_residual_evolution.png", dpi=220)
    plt.savefig(out / "progressive_residual_evolution.pdf")
    plt.close()
    print(f"Wrote {len(output)} residual evolution rows", flush=True)


if __name__ == "__main__":
    main()
