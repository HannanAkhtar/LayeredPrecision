"""Numerical-only 5-bit midpoint uniform control for Experiment 5."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from analyze_mse_layer_sensitivity import eligible_groups, write_json
from evaluate_imagenet import build_model
from lp_base_quantizers import quantize_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    model = build_model("resnet50")
    names = {name for group in eligible_groups(model).values() for name in group}
    sse = signal = count = 0.0
    started = time.perf_counter()
    for i, (name, weight) in enumerate(model.named_parameters(), 1):
        if name not in names:
            continue
        _, decoded = quantize_base(weight.detach(), "U0")
        sse += (weight.detach().double()-decoded.double()).square().sum().item()
        signal += weight.detach().double().square().sum().item()
        count += weight.numel()
    import math
    result = {"family": "U0", "bits": 5, "eligible_weights": int(count),
              "parameter_mse": sse/count, "parameter_sqnr_db": 10*math.log10(signal/sse),
              "calibration_seconds": round(time.perf_counter()-started, 3),
              "inference": "not evaluated; numerical control only"}
    write_json(out / "base_uniform_midpoint_control.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
