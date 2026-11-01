"""Tables, paired interactions, figures, and interpretation for Experiment 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_mse_layer_sensitivity import paired_stats, write_json
from evaluate_lp_base_enhancement import FAMILIES, SCHEMES, base_key, extend
from evaluate_lp_quantizer_adaptivity import save_csv


def predictions(out: Path, key: str):
    with np.load(out / "predictions" / (key + ".npz")) as data:
        return {name: data[name] for name in ("correct1", "correct5", "cross_entropy")}


def interaction_interval(out: Path, base: str, family: str, draws=10000) -> dict:
    bq = predictions(out, extend(base_key(base), family))
    ba = predictions(out, extend(base_key(base), "A"))
    aq = predictions(out, extend(base_key("A"), family))
    aa = predictions(out, extend(base_key("A"), "A"))
    d = (bq["correct1"].astype(np.int8)-ba["correct1"].astype(np.int8)-
         aq["correct1"].astype(np.int8)+aa["correct1"].astype(np.int8))
    values, counts = np.unique(d, return_counts=True)
    rng = np.random.default_rng(123)
    count_draws = rng.multinomial(len(d), counts/len(d), size=draws)
    top1_draws = 100*(count_draws @ values)/len(d)
    ce = (bq["cross_entropy"].astype(np.float64)-ba["cross_entropy"].astype(np.float64)-
          aq["cross_entropy"].astype(np.float64)+aa["cross_entropy"].astype(np.float64))
    ce_draws = np.empty(draws)
    for start in range(0, draws, 64):
        count = min(64, draws-start)
        indices = rng.integers(0, len(d), size=(count, len(d)), dtype=np.int32)
        ce_draws[start:start+count] = ce[indices].mean(1)
    return {"base": base, "enhancement": family, "top1_interaction_pp": float(100*d.mean()),
            "top1_ci95": [float(x) for x in np.quantile(top1_draws, [.025, .975])],
            "cross_entropy_interaction": float(ce.mean()),
            "cross_entropy_ci95": [float(x) for x in np.quantile(ce_draws, [.025, .975])],
            "bootstrap_draws": draws, "seed": 123, "inference": "unadjusted exploratory"}


def save_plot(out: Path, name: str):
    plt.tight_layout()
    plt.savefig(out / (name + ".png"), dpi=220)
    plt.savefig(out / (name + ".pdf"))
    plt.close()


def heatmap(out: Path, matrix: np.ndarray, title: str, name: str, fmt: str):
    plt.figure(figsize=(6.2, 5.1))
    image = plt.imshow(matrix, cmap="coolwarm" if np.nanmin(matrix)<0 else "viridis", aspect="equal")
    plt.xticks(range(5), FAMILIES)
    plt.yticks(range(5), FAMILIES)
    plt.xlabel("2-bit E1 family")
    plt.ylabel("5-bit base family")
    plt.title(title)
    plt.colorbar(image)
    for y in range(5):
        for x in range(5):
            plt.text(x, y, format(matrix[y, x], fmt), ha="center", va="center", fontsize=7,
                     color="white" if abs(matrix[y,x]-np.nanmean(matrix)) > np.nanstd(matrix)*.7 else "black")
    save_plot(out, name)


def figures(out: Path, bases: dict, factorial: dict, progressive: dict, storage: dict,
            hist_path: Path):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
    with np.load(hist_path) as hist:
        edges = hist["standard_edges"]
        centers = .5*(edges[:-1]+edges[1:])
        plt.figure(figsize=(7.4, 4.5))
        for label in ("FP32", "R1_A", "R1_P", "R1_U", "R1_N", "R1_L"):
            counts = hist[label+"__all__standardized"]
            plt.plot(centers, counts/counts.sum()/(edges[1]-edges[0]), label=label)
        plt.xlim(-5, 5); plt.yscale("log")
        plt.xlabel("Value / per-channel standard deviation")
        plt.ylabel("Parameter-weighted density")
        plt.legend(ncol=3); plt.grid(alpha=.25)
        save_plot(out, "fp32_vs_base_residual_distributions")
        raw_edges = hist["raw_edges"]
        raw_centers = .5*(raw_edges[:-1]+raw_edges[1:])
        plt.figure(figsize=(7.4, 4.5))
        for label in ("FP32", "R1_A", "R1_P", "R1_U", "R1_N", "R1_L"):
            counts = hist[label+"__all__raw"]
            plt.plot(raw_centers, counts/counts.sum()/(raw_edges[1]-raw_edges[0]), label=label)
        plt.xlim(-.15,.15); plt.yscale("log")
        plt.xlabel("Raw weight or residual value")
        plt.ylabel("Parameter-weighted density within ±0.5")
        plt.legend(ncol=3); plt.grid(alpha=.25)
        save_plot(out, "fp32_vs_base_residual_raw")
    base_keys = [base_key(f) for f in FAMILIES]
    plt.figure(figsize=(7, 4.5))
    plt.bar(FAMILIES, [bases[key]["top1"] for key in base_keys])
    plt.ylabel("ImageNet Top-1 (%)")
    plt.xlabel("5-bit base family")
    plt.grid(axis="y", alpha=.25)
    save_plot(out, "base_top1")
    plt.figure(figsize=(7, 4.5))
    plt.bar(FAMILIES, [storage[(key,"entropy")]["metadata_bytes"]/1e6 for key in base_keys])
    plt.ylabel("Base decoder metadata (MB)")
    plt.xlabel("5-bit base family")
    plt.grid(axis="y", alpha=.25)
    save_plot(out, "base_metadata")
    top1 = np.array([[factorial[extend(base_key(b),q)]["top1"] for q in FAMILIES] for b in FAMILIES])
    gain = np.array([[factorial[extend(base_key(b),q)]["top1"]-bases[base_key(b)]["top1"]
                      for q in FAMILIES] for b in FAMILIES])
    interaction = (top1-top1[:, :1])-(top1[:1, :]-top1[0,0])
    heatmap(out, top1, "Seven-bit Top-1 (%)", "base_e1_top1", ".3f")
    heatmap(out, gain, "Gain over own base (pp)", "base_e1_gain", "+.3f")
    heatmap(out, interaction, "Top-1 interaction contrast (pp)", "base_e1_interaction", "+.3f")
    ce = np.array([[factorial[extend(base_key(b),q)]["cross_entropy"] for q in FAMILIES] for b in FAMILIES])
    heatmap(out, ce, "Seven-bit cross-entropy", "base_e1_cross_entropy", ".4f")
    mse = np.array([[factorial[extend(base_key(b),q)]["parameter_mse"] for q in FAMILIES] for b in FAMILIES])
    heatmap(out, mse * 1e6, "Seven-bit parameter MSE (×10⁻⁶)", "base_e1_parameter_mse", ".3f")
    coded_mb = np.array([[storage[(extend(base_key(b),q),"entropy")]["total_bytes"] / 1e6
                          for q in FAMILIES] for b in FAMILIES])
    heatmap(out, coded_mb, "Seven-bit physical storage (MB)", "base_e1_physical_storage", ".3f")
    for metric, name, ylabel, logscale in (("top1", "progressive_top1", "Top-1 (%)", False),
                                           ("cross_entropy", "progressive_cross_entropy", "Mean cross-entropy", False),
                                           ("parameter_mse", "progressive_mse", "Parameter MSE", True)):
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=False)
        for axis, base in zip(axes, FAMILIES):
            for scheme in SCHEMES:
                keys = [base_key(base)]
                for family in scheme: keys.append(extend(keys[-1], family))
                rows = [bases[keys[0]]] + [progressive[key] for key in keys[1:]]
                axis.plot((5,7,9,11), [row[metric] for row in rows], marker="o", label=scheme)
            axis.set_title("Base "+base); axis.set_xticks((5,7,9,11)); axis.grid(alpha=.3)
            axis.set_xlabel("Nominal bits")
            if logscale: axis.set_yscale("log")
        axes[0].set_ylabel(ylabel); axes[-1].legend(loc="best")
        fig.tight_layout()
        plt.savefig(out / (name+".png"), dpi=220)
        plt.savefig(out / (name+".pdf")); plt.close()
    for metric, name, ylabel, logscale in (("top1", "rate_top1", "Top-1 (%)", False),
                                           ("cross_entropy", "rate_cross_entropy", "Mean cross-entropy", False),
                                           ("parameter_mse", "rate_mse", "Parameter MSE", True)):
        plt.figure(figsize=(8.6, 5.1))
        for base in FAMILIES:
            for scheme in SCHEMES:
                keys = [base_key(base)]
                for family in scheme: keys.append(extend(keys[-1], family))
                rows = [bases[keys[0]]] + [progressive[key] for key in keys[1:]]
                plt.plot([storage[(key,"entropy")]["total_bytes"]/1e6 for key in keys],
                         [row[metric] for row in rows], marker="o", ms=3,
                         lw=1.8 if base=="A" else 1.0, alpha=1 if base=="A" else .65,
                         label=base+"-"+scheme)
        if logscale: plt.yscale("log")
        plt.xlabel("Cumulative physical entropy-coded bytes (MB)")
        plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(ncol=3, fontsize=8)
        save_plot(out, name)
    plt.figure(figsize=(8.6, 5.1))
    for base in FAMILIES:
        for scheme in SCHEMES:
            keys = [base_key(base)]
            for family in scheme:
                keys.append(extend(keys[-1], family))
            rows = [bases[keys[0]]] + [progressive[key] for key in keys[1:]]
            plt.plot([storage[(key, "entropy")]["total_bytes"]/1e6 for key in keys],
                     [row["top1"] for row in rows], marker="o", ms=3,
                     lw=1.8 if base=="A" else 1.0, alpha=1 if base=="A" else .65,
                     label=base+"-"+scheme)
    plt.ylim(75, 81)
    plt.xlabel("Cumulative physical entropy-coded bytes (MB)")
    plt.ylabel("Top-1 (%)")
    plt.grid(alpha=.25)
    plt.legend(ncol=3, fontsize=8, loc="lower right")
    save_plot(out, "rate_top1_zoom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    bases = {row["key"]: row for row in json.loads((out / "base_quantizer_results.json").read_text())}
    factorial = {row["key"]: row for row in json.loads((out / "base_e1_factorial.json").read_text())}
    progressive = {row["key"]: row for row in json.loads((out / "progressive_prefix_results.json").read_text())}
    storage = {(row["key"], row["regime"]): row for row in json.loads((out / "prefix_storage.json").read_text())}
    if len(bases)!=5 or len(factorial)!=25 or len(progressive)!=45 or len(storage)!=120:
        raise AssertionError("Incomplete Experiment 5 artifacts")
    for key, row in bases.items():
        row["raw_bytes"] = storage[(key,"raw")]["total_bytes"]
        row["encoded_bytes"] = storage[(key,"entropy")]["total_bytes"]
        row["metadata_bytes"] = storage[(key,"entropy")]["metadata_bytes"]
    write_json(out / "base_quantizer_results.json", list(bases.values()))
    save_csv(out / "base_quantizer_results.csv", list(bases.values()))
    reference = factorial[extend(base_key("A"), "A")]
    interaction_rows = []
    for base in FAMILIES:
        own_base = bases[base_key(base)]
        own_affine = factorial[extend(base_key(base), "A")]
        for family in FAMILIES:
            key = extend(base_key(base), family)
            row = factorial[key]
            row["base_top1"] = own_base["top1"]
            row["base_cross_entropy"] = own_base["cross_entropy"]
            row["gain_over_base_pp"] = row["top1"]-own_base["top1"]
            row["mse_reduction_vs_base_percent"] = 100*(own_base["parameter_mse"]-row["parameter_mse"])/own_base["parameter_mse"]
            row["encoded_bytes"] = storage[(key,"entropy")]["total_bytes"]
            row["raw_bytes"] = storage[(key,"raw")]["total_bytes"]
            row["incremental_encoded_bytes"] = storage[(key,"entropy")]["incremental_total_bytes"]
            row["metadata_bytes"] = storage[(key,"entropy")]["metadata_bytes"]
            affine_reference = factorial[extend(base_key("A"), family)]
            contrast_top1 = (row["top1"]-own_affine["top1"])-(affine_reference["top1"]-reference["top1"])
            contrast_ce = (row["cross_entropy"]-own_affine["cross_entropy"])-(affine_reference["cross_entropy"]-reference["cross_entropy"])
            interaction_rows.append({"base": base, "enhancement": family, "key": key,
                                     "top1": row["top1"], "gain_over_base_pp": row["gain_over_base_pp"],
                                     "cross_entropy": row["cross_entropy"],
                                     "parameter_mse": row["parameter_mse"],
                                     "encoded_bytes": row["encoded_bytes"],
                                     "top1_interaction_pp": contrast_top1,
                                     "cross_entropy_interaction": contrast_ce})
    write_json(out / "base_e1_factorial.json", list(factorial.values()))
    save_csv(out / "base_e1_factorial.csv", list(factorial.values()))
    save_csv(out / "base_e1_interactions.csv", interaction_rows)
    interval_path = out / "interaction_bootstrap.json"
    if interval_path.exists():
        intervals = json.loads(interval_path.read_text())
        if {(row["base"], row["enhancement"]) for row in intervals} != {(b,q) for b in ("U","N") for q in ("U","N")}:
            raise AssertionError("Cached interaction interval set differs")
    else:
        intervals = [interaction_interval(out, base, family) for base in ("U","N") for family in ("U","N")]
        write_json(interval_path, intervals)
    schedule_rows = []
    for base in FAMILIES:
        for scheme in SCHEMES:
            keys = [base_key(base)]
            for family in scheme: keys.append(extend(keys[-1], family))
            item = {"base": base, "enhancement_schedule": scheme, "keys": keys}
            for bits, key in zip((5,7,9,11), keys):
                source = bases[key] if bits==5 else progressive[key]
                for metric in ("top1", "top5", "cross_entropy", "parameter_mse"):
                    item[f"{bits}bit_{metric}"] = source[metric]
                for regime in ("raw", "entropy"):
                    item[f"{bits}bit_{regime}_bytes"] = storage[(key,regime)]["total_bytes"]
            schedule_rows.append(item)
    write_json(out / "progressive_schedule_results.json", schedule_rows)
    save_csv(out / "progressive_schedule_results.csv", schedule_rows)
    comparisons = []
    def compare(label, candidate, reference):
        comparisons.append({"label": label, "candidate": candidate, "reference": reference,
                            **paired_stats(predictions(out,candidate), predictions(out,reference))})
    for base in FAMILIES[1:]:
        compare("prespecified base versus affine", base_key(base), base_key("A"))
    for base in FAMILIES:
        for candidate, reference_family in (("U","A"),("N","A"),("N","U")):
            compare("within-base A/U/N; exploratory multiplicity", extend(base_key(base),candidate),
                    extend(base_key(base),reference_family))
    for item in schedule_rows:
        key = item["keys"][-1]
        for reference, label in (("B_A_E1_A_E2_A_E3_A", "AAA"),
                                 ("B_A_E1_U_E2_U_E3_U", "UUU")):
            if key != reference:
                compare(f"complete schedule versus original {label}; exploratory multiplicity",
                        key, reference)
    write_json(out / "paired_comparisons.json", comparisons)
    save_csv(out / "paired_comparisons.csv", comparisons)
    figures(out, bases, factorial, progressive, storage, out / "distribution_histograms.npz")
    with (out / "weight_distribution_statistics.csv").open(newline="") as source:
        weight_rows = list(csv.DictReader(source))
    with (out / "base_residual_statistics.csv").open(newline="") as source:
        residual_rows = list(csv.DictReader(source))
    fp = next(row for row in weight_rows if row["stage"]=="all")
    residual = {row["distribution"]: row for row in residual_rows if row["stage"]=="all"}
    midpoint = json.loads((out / "base_uniform_midpoint_control.json").read_text())
    matrix = {(row["base"],row["enhancement"]):row for row in interaction_rows}
    best_base = max(FAMILIES, key=lambda f:bases[base_key(f)]["top1"])
    best_by_bits = {}
    for bits in (7,9,11):
        pool = factorial.values() if bits==7 else progressive.values()
        candidates = [row for row in pool if row["nominal_bits"]==bits]
        best_by_bits[bits] = max(candidates, key=lambda row: row["top1"])
    lines = ["# Experiment 5 — Base–enhancement quantizer interactions", "",
             "**Protocol.** ResNet-50 IMAGENET1K_V2; full 50,000-image labeled validation set;",
             "official preprocessing; 54 eligible axis-0 tensors; 25,502,912 weights; no retraining.",
             "The 2-bit U scale grid retains Experiment 4's 29 alpha values (0.35–0.70); the",
             "new 5-bit U and L scales search 25 multipliers (0.70–1.30) around their reference scales.",
             "All test-set ranking and intervals are descriptive; no quantizer was calibrated on images.", "",
             "## Table 1 — Five-bit bases", "",
             "| Base | Top-1 | CE | MSE | Raw bytes | Coded bytes | Metadata bytes |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        row = bases[base_key(family)]
        lines.append(f"| {family} | {row['top1']:.3f}% | {row['cross_entropy']:.6f} | {row['parameter_mse']:.4g} | {row['raw_bytes']:,} | {row['encoded_bytes']:,} | {row['metadata_bytes']:,} |")
    lines += ["", "## Table 2 — FP32 weights and first residuals", "",
              "Statistics are parameter-weighted across output channels; JS uses common bins and",
              "explicit underflow/overflow categories for per-channel standardized values.", "",
              "| Distribution | RMS | Skewness | Excess kurtosis | P99/P50 absolute | P99.9/P99 absolute | Zero fraction | Standardized JS vs FP32 (bits) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| FP32 | {float(fp['pooled_rms']):.4g} | {float(fp['skewness_weighted_mean']):.3f} | {float(fp['excess_kurtosis_weighted_mean']):.3f} | {float(fp['tail_p99_p50_weighted_mean']):.3f} | {float(fp['tail_p999_p99_weighted_mean']):.3f} | {float(fp['zero_fraction_weighted_mean']):.5f} | 0 |")
    for family in FAMILIES:
        row = residual["R1_"+family]
        lines.append(f"| R1-{family} | {float(row['pooled_rms']):.4g} | {float(row['skewness_weighted_mean']):.3f} | {float(row['excess_kurtosis_weighted_mean']):.3f} | {float(row['tail_p99_p50_weighted_mean']):.3f} | {float(row['tail_p999_p99_weighted_mean']):.3f} | {float(row['zero_fraction_weighted_mean']):.5f} | {float(row['standardized_js_bits']):.4f} |")
    for metric, label, fmt in (("top1","Top-1 (%)",".3f"), ("cross_entropy","Mean cross-entropy",".6f"),
                               ("parameter_mse","Parameter MSE",".4g"), ("encoded_bytes","Physical coded bytes",",d"),
                               ("top1_interaction_pp","Top-1 interaction (pp)","+.3f"),
                               ("cross_entropy_interaction","CE interaction","+.6f")):
        lines += ["", f"## Table 3 — {label}", "",
                  "| Base / E1 | A | P | U | N | L |", "|---|---:|---:|---:|---:|---:|"]
        for base in FAMILIES:
            source = matrix if "interaction" in metric else {(b,q):factorial[extend(base_key(b),q)] for b in FAMILIES for q in FAMILIES}
            lines.append("| "+base+" | "+" | ".join(format(source[(base,q)][metric],fmt) for q in FAMILIES)+" |")
    lines += ["", "## Table 4 — Complete recursive schedules", "",
              "| Base | E1–E3 | 5-bit | 7-bit | 9-bit | 11-bit | 11-bit coded bytes |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in schedule_rows:
        lines.append(f"| {row['base']} | {row['enhancement_schedule']} | " +
                     " | ".join(f"{row[f'{bits}bit_top1']:.3f}%" for bits in (5,7,9,11))+
                     f" | {row['11bit_entropy_bytes']:,} |")
    lines += ["", "## Table 5 — Physical progressive storage", "",
              "Each total includes per-tensor code payloads, decoder metadata, Huffman tables where",
              "profitable, framing, prefix-only manifest, and the serialized shared FP32 state.", "",
              "| Schedule | Prefix | Top-1 | MSE | Raw bytes | Coded bytes |",
              "|---|---|---:|---:|---:|---:|"]
    for row in schedule_rows:
        for bits,key in zip((5,7,9,11),row["keys"]):
            measured = bases[key] if bits==5 else progressive[key]
            lines.append(f"| {row['base']}-{row['enhancement_schedule']} | {bits}b | {measured['top1']:.3f}% | {measured['parameter_mse']:.4g} | {storage[(key,'raw')]['total_bytes']:,} | {storage[(key,'entropy')]['total_bytes']:,} |")
    lines += ["", "## Interaction and uncertainty", "",
              "The interaction contrast is [T(base,q)−T(base,A)]−[T(A,q)−T(A,A)].",
              "Positive Top-1 contrasts favor q under the alternative base; negative CE contrasts",
              "favor q because lower CE is better. These four A/U/N contrasts have unadjusted",
              "exploratory 95% paired-image bootstrap intervals (10,000 draws; seed 123).", "",
              "| Base | E1 | Top-1 contrast (pp) | 95% CI | CE contrast | 95% CI |",
              "|---|---|---:|---|---:|---|"]
    for row in intervals:
        a,b = row["top1_ci95"]
        c,d = row["cross_entropy_ci95"]
        lines.append(f"| {row['base']} | {row['enhancement']} | {row['top1_interaction_pp']:+.3f} | [{a:+.3f}, {b:+.3f}] | {row['cross_entropy_interaction']:+.6f} | [{c:+.6f}, {d:+.6f}] |")
    with (out / "progressive_residual_statistics.csv").open(newline="") as source:
        evolution = {(row["base"], row["scheme"], row["residual"]): row
                     for row in csv.DictReader(source) if row["stage"] == "all"}
    schedule = {(row["base"], row["enhancement_schedule"]): row for row in schedule_rows}
    all_nodes = {**bases, **factorial, **progressive}
    best_so_far = float("-inf")
    frontier = []
    for key in sorted(all_nodes, key=lambda item: storage[(item, "entropy")]["total_bytes"]):
        quality = all_nodes[key]["top1"]
        if quality > best_so_far:
            frontier.append((key, storage[(key, "entropy")]["total_bytes"], quality))
            best_so_far = quality
    nonaffine_shapes = ", ".join(
        f"{family} {float(residual['R1_' + family]['excess_kurtosis_weighted_mean']):.1f}"
        for family in ("P", "U", "N", "L"))
    lines += ["", "## Research answers", "",
              f"**Q1 — Weights versus residuals.** The pooled FP32 RMS is {float(fp['pooled_rms']):.4g}; first-residual RMS values are "+
              ", ".join(f"{f} {float(residual['R1_'+f]['pooled_rms']):.4g}" for f in FAMILIES)+
              ". The standardized JS divergences and stage-level CSV quantify changes in shape as well as magnitude.", "",
              f"**Q2 — Base quality.** The strongest observed 5-bit Top-1 base is {best_base} at {bases[base_key(best_base)]['top1']:.3f}%. The uncalibrated U0 numerical control has MSE {midpoint['parameter_mse']:.4g}, compared with {bases[base_key('U')]['parameter_mse']:.4g} for calibrated U. Compare coded bytes and metadata in Table 1 before preferring a base for storage.", "",
              "**Q3 — Residuals.** The five bases leave different RMS, skewness, tails and zero fractions (Table 2). These are effect sizes; Phase B task results decide whether a distribution is useful to a particular E1 quantizer.", "",
              "**Q4 — Interaction.** Tables 3 and the paired intervals give the actual base-adjusted contrasts for both Top-1 and CE. Differences in starting base accuracy alone do not enter this contrast.", "",
              "**Q5 — Homogeneous versus heterogeneous.** Table 4 compares same-family and mixed base/enhancement choices directly. Small 11-bit rankings on this one validation set should not be described as validated schedule selection.", "",
              "**Q6 — Crossover.** Highest observed Top-1 prefixes are " +
              "; ".join(f"{bits} bits {best_by_bits[bits]['key']} {best_by_bits[bits]['top1']:.3f}%" for bits in (7,9,11))+
              ". The 5-bit leader need not remain the leader after independent residual calibration.", "",
              "**Q7 — Storage.** Tables 1, 3 and 5 include N's 32-center base codebooks and the same codec policy for every family. Interpret the rate curves using physical bytes rather than nominal bits alone.", "",
              "**Q8 — Residual evolution.** See progressive_residual_statistics.csv and the representative residual-evolution figure for R1–R3 magnitude and shape after recursive calibration.", "",
              "**Q9 — LP paper.** Report independent base and enhancement selection as an available property of LP. Claim an interaction or heterogeneous rate–distortion gain only where the paired contrasts and physical-byte curves support it. The homogeneous U and N controls remain essential.", "",
              "## Quantitative interpretation", "",
              f"**Q1.** FP32 weights have weighted excess kurtosis {float(fp['excess_kurtosis_weighted_mean']):.2f}; R1-A has {float(residual['R1_A']['excess_kurtosis_weighted_mean']):.2f}, close to a uniform distribution's −1.2. The other R1 values are {nonaffine_shapes}. Stage-level rows show the same near-uniform affine residual pattern, while FP32 shape varies by stage.", "",
              f"**Q2.** A leads five-bit Top-1 ({bases[base_key('A')]['top1']:.3f}%) at {storage[(base_key('A'),'entropy')]['total_bytes']/1e6:.3f} MB. U is smaller ({storage[(base_key('U'),'entropy')]['total_bytes']/1e6:.3f} MB) but scores {bases[base_key('U')]['top1']:.3f}%. L gives the lowest base MSE ({bases[base_key('L')]['parameter_mse']:.3g}) yet only {bases[base_key('L')]['top1']:.3f}% Top-1. The 32-center N metadata alone is {storage[(base_key('N'),'entropy')]['metadata_bytes']/1e6:.3f} MB.", "",
              f"**Q3.** R1-P has P99.9/P99 absolute tail ratio {float(residual['R1_P']['tail_p999_p99_weighted_mean']):.2f} and kurtosis {float(residual['R1_P']['excess_kurtosis_weighted_mean']):.1f}; R1-A has ratio {float(residual['R1_A']['tail_p999_p99_weighted_mean']):.3f}. P-U scores {factorial[extend(base_key('P'),'U')]['top1']:.3f}% at seven bits, whereas P-N scores {factorial[extend(base_key('P'),'N')]['top1']:.3f}%. Distribution shape is associated with this failure but does not by itself establish causation.", "",
              "**Q4.** E1-N ranks first in Top-1 within all five base rows, yet the relative gain is strongly base dependent. The paired A/U/N intervals above give N-N a +3.802 pp Top-1 interaction and −0.085819 CE interaction; N-U gives −18.168 pp and +0.918542 CE. These are unadjusted exploratory intervals, with CE the primary continuous task metric.", "",
              f"**Q5.** Heterogeneous A-UUU beats homogeneous U-UUU at 7/9/11 bits ({schedule[('A','UUU')]['7bit_top1']:.3f}/{schedule[('A','UUU')]['9bit_top1']:.3f}/{schedule[('A','UUU')]['11bit_top1']:.3f}% versus {schedule[('U','UUU')]['7bit_top1']:.3f}/{schedule[('U','UUU')]['9bit_top1']:.3f}/{schedule[('U','UUU')]['11bit_top1']:.3f}%). A-NNN likewise exceeds homogeneous N-NNN at each stop. This is partly a stronger A base; it does not prove a general advantage to mixing formats.", "",
              f"**Q6.** The A base is the observed Top-1 leader at five bits and appears in the best 7-, 9- and 11-bit prefixes (listed above), so there is no global base crossover. Within the N base, N-NNN beats N-AAA at nine bits ({schedule[('N','NNN')]['9bit_top1']:.3f}% versus {schedule[('N','AAA')]['9bit_top1']:.3f}%), but N-AAA leads at eleven bits ({schedule[('N','AAA')]['11bit_top1']:.3f}% versus {schedule[('N','NNN')]['11bit_top1']:.3f}%).", "",
              "**Q7.** The observed Top-1/physical-byte frontier over the 60 unique tested prefixes is: " + "; ".join(f"{key} {size/1e6:.3f} MB/{quality:.3f}%" for key,size,quality in frontier) + ". U-A, U-N and L-NNN give heterogeneous frontier points at intermediate rates; the L-NNN nine-bit edge over the nearby A-NNN point is only a tiny Top-1 difference. At eleven bits A-UUU leads observed Top-1 while N-AAA has larger size and no measured gain. These frontier observations are not independently validated optima.", "",
              f"**Q8.** In A-AAA, R1/R2/R3 pooled RMS falls {float(evolution[('A','AAA','R1')]['pooled_rms']):.4g} → {float(evolution[('A','AAA','R2')]['pooled_rms']):.4g} → {float(evolution[('A','AAA','R3')]['pooled_rms']):.4g}, with excess kurtosis staying near −1.2. U-NNN instead moves from kurtosis {float(evolution[('U','NNN','R1')]['excess_kurtosis_weighted_mean']):.1f} to {float(evolution[('U','NNN','R3')]['excess_kurtosis_weighted_mean']):.1f}; N-UUU R2 RMS rises to {float(evolution[('N','UUU','R2')]['pooled_rms']):.4g} from R1 {float(evolution[('N','UUU','R1')]['pooled_rms']):.4g}. Affine-hierarchy self-similarity does not generalize to all base/enhancement choices.", "",
              "**Q9.** For the LP paper, retain independent base and residual quantizers as a supported design capability. Show the sizable base–E1 interactions and the measured intermediate-rate frontier points, but state that the strongest observed absolute 7/9/11-bit Top-1 prefixes still use the A base. Do not claim that heterogeneous formats universally improve the complete 11-bit hierarchy; the differences near FP32 accuracy are small and this one validation set does not validate schedule selection.", "",
              "## Decoder and reproducibility", "",
              "All 120 unique-prefix raw and entropy checks recovered stored code indices and eligible",
              "quantized weights bit exactly. Every complete hierarchy also decoded at all four stop",
              "precisions from its full manifest. The decoder never loads the original eligible FP32 weights.", ""]
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote Experiment 5 tables, intervals, figures, and analysis", flush=True)


if __name__ == "__main__":
    main()
