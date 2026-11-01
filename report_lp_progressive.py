#!/usr/bin/env python3
"""Produce Experiment 4 comparison tables, paired statistics, figures, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_mse_layer_sensitivity import paired_stats, write_json
from evaluate_lp_progressive_schedules import SCHEDULES, all_prefixes
from evaluate_lp_quantizer_adaptivity import save_csv


def _arrays(path: Path):
    with np.load(path) as source:
        return {key: source[key] for key in ("correct1", "correct5", "cross_entropy")}


def _percent(x):
    return f"{x:.3f}"


def _f(x):
    return f"{x:.4g}"


def plots(out: Path, isolated: list[dict], prefixes: dict, storage: dict):
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10})
    def save(name):
        plt.tight_layout()
        plt.savefig(out / (name + ".png"))
        plt.savefig(out / (name + ".pdf"))
        plt.close()
    families = ("A", "S", "U0", "U", "N")
    colors = {"A": "#555555", "S": "#b86b24", "U0": "#5681bd", "U": "#155b9a", "N": "#a5354a"}
    for metric, name, ylabel, logscale in (("parameter_mse", "uniform_vs_nonuniform_mse", "Parameter MSE", True),
                                           ("top1", "uniform_vs_nonuniform_accuracy", "ImageNet Top-1 (%)", False)):
        plt.figure(figsize=(7.2, 4.4))
        for family in families:
            rows = [next(row for row in isolated if row["enhancement"] == stage and row["family"] == family)
                    for stage in (1, 2, 3)]
            plt.plot((1, 2, 3), [row[metric] for row in rows], marker="o", label=family, color=colors[family])
        if logscale: plt.yscale("log")
        plt.xticks((1, 2, 3), ("E1", "E2", "E3"))
        plt.ylabel(ylabel); plt.xlabel("Isolated enhancement on B / BA / BAA")
        plt.grid(alpha=.3); plt.legend(ncol=5, loc="best")
        save(name)
    focus = ("AAA", "UUU", "NNN", "NNA", "NUA", "UNA")
    palette = dict(zip(focus, ("#555555", "#155b9a", "#a5354a", "#99509f", "#2a8a76", "#b27c1d")))
    for metric, name, ylabel, logscale in (("top1", "progressive_accuracy", "ImageNet Top-1 (%)", False),
                                           ("parameter_mse", "progressive_mse", "Parameter MSE", True)):
        plt.figure(figsize=(7.5, 4.8))
        for schedule in focus:
            rows = [prefixes["B"+schedule[:n]] for n in (0, 1, 2, 3)]
            plt.plot((5, 7, 9, 11), [row[metric] for row in rows], marker="o", label=schedule, color=palette[schedule])
        if logscale: plt.yscale("log")
        plt.xticks((5, 7, 9, 11)); plt.xlabel("Nominal cumulative bits per eligible weight")
        plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(ncol=3)
        save(name)
    all_keys = all_prefixes()
    nondominated = {key for key in all_keys if not any(
        other != key and storage[(other, "entropy")]["total_bytes"] <= storage[(key, "entropy")]["total_bytes"]
        and prefixes[other]["top1"] >= prefixes[key]["top1"]
        and (storage[(other, "entropy")]["total_bytes"] < storage[(key, "entropy")]["total_bytes"]
             or prefixes[other]["top1"] > prefixes[key]["top1"])
        for other in all_keys)}
    covered = {"B"+schedule[:n] for schedule in focus for n in (0,1,2,3)}
    extra = [schedule for schedule in SCHEDULES if schedule not in focus
             and any("B"+schedule[:n] in nondominated - covered for n in (1,2,3))]
    rate_schedules = focus + tuple(extra)
    for metric, name, ylabel, logscale in (("top1", "progressive_rate_accuracy", "ImageNet Top-1 (%)", False),
                                           ("cross_entropy", "progressive_rate_cross_entropy", "Mean cross-entropy", False),
                                           ("parameter_mse", "progressive_rate_mse", "Parameter MSE", True)):
        plt.figure(figsize=(7.8, 4.9))
        for schedule in rate_schedules:
            keys = ["B"+schedule[:n] for n in (0, 1, 2, 3)]
            plt.plot([storage[(key, "entropy")]["total_bytes"]/1e6 for key in keys],
                     [prefixes[key][metric] for key in keys], marker="o", label=schedule,
                     color=palette.get(schedule), linestyle="-" if schedule in focus else "--",
                     alpha=1 if schedule in focus else .7)
        if logscale: plt.yscale("log")
        plt.xlabel("Cumulative serialized bytes, per-tensor Huffman with raw fallback (MB)")
        plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(ncol=3)
        save(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_progressive_quantizer_study"))
    parser.add_argument("--saved-artifacts", type=Path, default=Path("outputs/lp_quantizer_adaptivity"))
    args = parser.parse_args()
    out = args.outdir.resolve(); old = args.saved_artifacts.resolve()
    isolated = json.loads((out / "isolated_quantizer_comparison.json").read_text())
    progressive_rows = json.loads((out / "progressive_prefixes.json").read_text())
    storage_rows = json.loads((out / "prefix_storage.json").read_text())
    run_meta_path = out / "run_metadata.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta.update({"metadata_precision": "FP32", "seed": 123,
                     "prespecified_schedules": list(SCHEDULES),
                     "entropy_coding_policy": "per-tensor canonical Huffman if payload plus table is smaller than raw; otherwise flagged raw fallback",
                     "shared_serialized_state_bytes": (out / "serialized/raw/shared_fp32_state.pt").stat().st_size})
    write_json(run_meta_path, run_meta)
    isolated_streams = {row["config"]: row for row in json.loads((out / "isolated_stream_sizes.json").read_text())}
    storage = {(row["prefix"], row["regime"]): row for row in storage_rows}
    prefix = {row["prefix"]: row for row in progressive_rows}
    reference_prefixes = {1: "B", 2: "BA", 3: "BAA"}
    for row in isolated:
        row["enhancement"] = row.get("enhancement", len(row["parent"]))
        stage, family = row["enhancement"], row["family"]
        baseline = next(item for item in isolated if item["enhancement"] == stage and item["family"] == "A")
        preceding = prefix[reference_prefixes[stage]]
        row["delta_top1_vs_A_pp"] = row["top1"]-baseline["top1"]
        row["delta_top1_vs_preceding_pp"] = row["top1"]-preceding["top1"]
        row["mse_reduction_vs_A_percent"] = 100*(baseline["parameter_mse"]-row["parameter_mse"])/baseline["parameter_mse"]
        stat = isolated_streams[f"E{stage}_{family}"]
        for key in ("raw_code_bytes", "metadata_bytes", "framing_bytes", "physical_bytes"):
            row[key] = stat[key]
        measured_prefix = stat["prefix"]
        row["measured_binary_stream"] = (f"serialized/raw/streams/{measured_prefix}.bin"
                                         if family in ("A", "U", "N") else
                                         f"serialized/raw/isolated/E{stage}_{family}.bin")
        row["shared_preceding_prefix"] = reference_prefixes[stage]
    for stage in (1, 2, 3):
        u0 = next(row for row in isolated if row["enhancement"] == stage and row["family"] == "U0")
        u = next(row for row in isolated if row["enhancement"] == stage and row["family"] == "U")
        if u["residual_mse"] > u0["residual_mse"] + 1e-12:
            raise AssertionError(f"U residual MSE exceeds U0 at E{stage}")
    write_json(out / "isolated_quantizer_comparison.json", isolated)
    save_csv(out / "isolated_quantizer_comparison.csv", isolated)
    uniform = []
    for row in isolated:
        if row["family"] not in ("U0", "U"):
            continue
        item = {key: row.get(key) for key in ("enhancement", "family", "parent", "parameter_mse", "residual_mse",
                                               "top1", "top5", "cross_entropy", "calibration_seconds", "alpha_distribution",
                                               "scale_min", "scale_median", "scale_max", "physical_bytes", "metadata_bytes")}
        distribution = row.get("alpha_distribution", {})
        total = sum(distribution.values())
        item["alpha_0_5_fraction"] = distribution.get("0.5", 0)/total if total else None
        item["mean_selected_alpha"] = sum(float(alpha)*count for alpha,count in distribution.items())/total if total else None
        uniform.append(item)
    write_json(out / "uniform_calibration.json", uniform)
    save_csv(out / "uniform_calibration.csv", uniform)
    for row in progressive_rows:
        key = row["prefix"]
        parent = key[:-1] if key != "B" else None
        aaa = "B" + "A"*(len(key)-1)
        row["delta_top1_vs_previous_pp"] = row["top1"]-prefix[parent]["top1"] if parent else None
        row["mse_reduction_vs_previous_percent"] = (100*(prefix[parent]["parameter_mse"]-row["parameter_mse"])/
                                                      prefix[parent]["parameter_mse"]) if parent else None
        row["delta_top1_vs_AAA_pp"] = row["top1"]-prefix[aaa]["top1"]
        row["raw_total_bytes"] = storage[(key, "raw")]["total_bytes"]
        row["entropy_total_bytes"] = storage[(key, "entropy")]["total_bytes"]
        row["raw_incremental_bytes"] = storage[(key, "raw")]["incremental_total_bytes"]
        row["entropy_incremental_bytes"] = storage[(key, "entropy")]["incremental_total_bytes"]
    write_json(out / "progressive_prefixes.json", progressive_rows)
    save_csv(out / "progressive_prefixes.csv", progressive_rows)
    schedule_rows = []
    for schedule in SCHEDULES:
        keys = ["B"+schedule[:n] for n in (0, 1, 2, 3)]
        row = {"schedule": schedule, "prefixes": keys}
        for bits, key in zip((5, 7, 9, 11), keys):
            for metric in ("top1", "top5", "cross_entropy", "parameter_mse", "raw_total_bytes", "entropy_total_bytes"):
                row[f"{bits}bit_{metric}"] = prefix[key][metric]
        schedule_rows.append(row)
    write_json(out / "schedule_summary.json", schedule_rows)
    save_csv(out / "schedule_summary.csv", schedule_rows)
    comparisons = []
    def prediction_path(key):
        if key == "B": return None
        old_config = {"BA":"E1_A", "BAA":"E2_A", "BAAA":"E3_A", "BN":"E1_N", "BAN":"E2_N", "BAAN":"E3_N"}.get(key)
        if key.startswith("E") and "_" in key:
            stage, family = key.split("_", 1)
            if family in ("S", "N", "A"):
                return old / "predictions" / (f"full_{key}.npz")
        if old_config: return old / "predictions" / (f"full_{old_config}.npz")
        return out / "predictions" / (key + ".npz")
    def compare(label, candidate, reference):
        if candidate == reference:
            comparisons.append({"comparison": label, "candidate": candidate, "reference": reference,
                                "paired_gain_images": 0, "paired_loss_images": 0, "paired_top1_delta": 0,
                                "paired_ce_delta": 0, "paired_bootstrap_top1_ci95": [0, 0]})
            return
        a, b = _arrays(prediction_path(candidate)), _arrays(prediction_path(reference))
        comparisons.append({"comparison": label, "candidate": candidate, "reference": reference,
                            **paired_stats(a, b)})
    for candidate, reference in (("BU", "E1_S"), ("BU", "E1_N"),
                                 ("BAU", "E2_N"), ("BAAU", "E3_N")):
        compare("primary isolated", candidate, reference)
    for candidate, reference in (("BU", "E1_U0"), ("BAU", "E2_U0"),
                                 ("BAAU", "E3_U0")):
        compare("secondary calibrated versus midpoint", candidate, reference)
    for schedule in ("UUU", "NNN", "NNA"):
        for n in (0, 1, 2, 3):
            key, aaa = "B"+schedule[:n], "B"+"A"*n
            compare(f"primary {schedule} versus AAA {5+2*n}bit", key, aaa)
    for n in (2, 3):
        compare(f"primary NUA versus NNA {5+2*n}bit", "B"+"NUA"[:n], "B"+"NNA"[:n])
    compare("exploratory UAA versus AAA 11bit", "BUAA", "BAAA")
    write_json(out / "paired_comparisons.json", comparisons)
    save_csv(out / "paired_comparisons.csv", comparisons)
    plots(out, isolated, prefix, storage)
    lines = ["# Experiment 4 — Uniform calibration and progressive LP", "",
             "**Protocol.** ResNet-50 IMAGENET1K_V2, 50,000 labeled validation images, official preprocessing,",
             "54 eligible tensors (25,502,912 weights), axis-0 5-2-2-2 LP, no retraining. All 12 schedules",
             "were prespecified. A, S and N isolated evaluations and exact A/N prefixes reuse verified",
             "Experiment 3 predictions on the same checkpoint, dataset order and transform.", "",
             "## Table 1 — Uniform quantizer comparison", "",
             "| Enhancement | Quantizer | MSE | Top-1 | Δ versus A (pp) | Entropy (bits/code) | Metadata (bytes) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in isolated:
        lines.append(f"| E{row['enhancement']} | {row['family']} | {_f(row['parameter_mse'])} | {_percent(row['top1'])}% | {row['delta_top1_vs_A_pp']:+.3f} | {row['parameter_weighted_conditional_code_entropy']:.3f} | {row['metadata_bytes']:,} |")
    lines += ["", "## Table 2 — Complete LP schedules", "",
              "| Schedule | 5-bit Top-1 | 7-bit Top-1 | 9-bit Top-1 | 11-bit Top-1 |",
              "|---|---:|---:|---:|---:|"]
    for row in schedule_rows:
        lines.append("| " + row["schedule"] + " | " + " | ".join(f"{row[f'{bits}bit_top1']:.3f}%" for bits in (5,7,9,11)) + " |")
    lines += ["", "## Table 3 — Progressive practical storage", "",
              "Each total includes all earlier streams, decoder metadata, frame bytes, a prefix-only manifest,",
              "and the serialized shared FP32 state. Huffman uses a per-tensor canonical table where worthwhile",
              "and an explicitly flagged raw fallback otherwise.", "",
              "| Schedule | Prefix | Top-1 | MSE | Raw bytes | Entropy-coded bytes |",
              "|---|---|---:|---:|---:|---:|"]
    for schedule in SCHEDULES:
        for n in (0,1,2,3):
            key = "B"+schedule[:n]
            if n == 0 and schedule != "AAA": continue
            lines.append(f"| {schedule} | {5+2*n}b {key} | {prefix[key]['top1']:.3f}% | {_f(prefix[key]['parameter_mse'])} | {storage[(key,'raw')]['total_bytes']:,} | {storage[(key,'entropy')]['total_bytes']:,} |")
    lines += ["", "## Table 4 — Quantizer complexity and benefits", "",
              "| Quantizer | Decoder FP32 parameters/channel | Full E1 Top-1 | Full E1 MSE | Raw enhancement bytes |",
              "|---|---:|---:|---:|---:|"]
    for family, parameters in (("A", 1), ("S", 1), ("U0", 1), ("U", 1), ("N", 4)):
        row = next(x for x in isolated if x["enhancement"] == 1 and x["family"] == family)
        lines.append(f"| {family} | {parameters}{' + 1 uint8 zero point' if family=='A' else ''} | {row['top1']:.3f}% | {_f(row['parameter_mse'])} | {row['physical_bytes']:,} |")
    get = lambda stage, family: next(x for x in isolated if x["enhancement"] == stage and x["family"] == family)
    e1s,e1u0,e1u,e1n = [get(1,x) for x in ("S","U0","U","N")]
    e2u,e2n = get(2,"U"),get(2,"N")
    e3u,e3n = get(3,"U"),get(3,"N")
    mse_gain = lambda a,b: 100*(a["parameter_mse"]-b["parameter_mse"])/a["parameter_mse"]
    mse_gap_closed = 100*(e1s["parameter_mse"]-e1u0["parameter_mse"])/(e1s["parameter_mse"]-e1n["parameter_mse"])
    n_metadata_premium = storage[("BN","entropy")]["incremental_stream_bytes"] - storage[("BU","entropy")]["incremental_stream_bytes"]
    e1_u_n = next(row for row in comparisons if row["candidate"] == "BU" and row["reference"] == "E1_N")
    e1_interval = e1_u_n["paired_bootstrap_top1_ci95"]
    lines += ["", "## Answers to the research questions", "",
              f"**Q1 — Scale calibration.** U0 reduces E1 MSE by {mse_gain(e1s,e1u0):.2f}% versus S, closing {mse_gap_closed:.2f}% of the S-to-N MSE gap; U adds {mse_gain(e1u0,e1u):.2f}% versus U0. The 0.5 scale is selected in {next(row for row in uniform if row['enhancement']==1 and row['family']=='U')['alpha_0_5_fraction']*100:.1f}% of E1 channels. E1 Top-1 changes from {e1s['top1']:.3f}% (S) to {e1u0['top1']:.3f}% (U0) and {e1u['top1']:.3f}% (U). Thus the original grid calibration explains most of N's MSE advantage, but the task metric must be judged separately.",
              "",
              f"**Q2 — Uniform versus Lloyd–Max.** At E1, N reduces MSE by {mse_gain(e1u,e1n):.2f}% relative to U, while U minus N Top-1 is {e1u['top1']-e1n['top1']:+.3f} pp (paired 95% interval [{e1_interval[0]:+.3f}, {e1_interval[1]:+.3f}]). At E2 and E3, U minus N Top-1 is {e2u['top1']-e2n['top1']:+.3f} and {e3u['top1']-e3n['top1']:+.3f} pp. At E3, A scores {get(3,'A')['top1']:.3f}% despite substantially higher MSE than U or N. Weight MSE and accuracy do not rank these quantizers identically.",
              "",
              f"**Q3 — Lloyd–Max metadata.** At E1, standalone raw enhancement streams measure {storage[('BU','raw')]['incremental_stream_bytes']:,} bytes (U) and {storage[('BN','raw')]['incremental_stream_bytes']:,} bytes (N); entropy streams measure {storage[('BU','entropy')]['incremental_stream_bytes']:,} and {storage[('BN','entropy')]['incremental_stream_bytes']:,} bytes. N adds {n_metadata_premium:,} bytes ({100*n_metadata_premium/storage[('BU','entropy')]['incremental_stream_bytes']:.2f}% of U's E1 stream). A's entropy-coded E1 stream measures {storage[('BA','entropy')]['incremental_stream_bytes']:,} bytes because its code frequencies differ. N's extra storage accompanies a prespecified E1 accuracy advantage over U; at E2/E3 the isolated Top-1 differences are smaller and their paired intervals should guide the choice.",
              "",
              f"**Q4 — Entire hierarchy.** At 11 bits AAA / UUU / NNN Top-1 = {prefix['BAAA']['top1']:.3f}% / {prefix['BUUU']['top1']:.3f}% / {prefix['BNNN']['top1']:.3f}% (FP32 reference 80.746%); MSE = {_f(prefix['BAAA']['parameter_mse'])} / {_f(prefix['BUUU']['parameter_mse'])} / {_f(prefix['BNNN']['parameter_mse'])}. All are recursive residual calibrations.",
              ""]
    best_by_precision = {}
    for n in (1,2,3):
        candidates = sorted({"B"+s[:n] for s in SCHEDULES})
        best = max(candidates, key=lambda key: prefix[key]["top1"])
        best_by_precision[5+2*n] = (best,prefix[best]["top1"])
    budget_choices = []
    for bits, aaa in ((7,"BA"),(9,"BAA"),(11,"BAAA")):
        budget = storage[(aaa,"entropy")]["total_bytes"]
        candidates = [key for key in all_prefixes() if storage[(key,"entropy")]["total_bytes"] <= budget]
        winner = max(candidates, key=lambda key: prefix[key]["top1"])
        budget_choices.append((bits, budget, winner, prefix[winner]["top1"]))
    nine_bit_keys = sorted({"B"+schedule[:2] for schedule in SCHEDULES})
    lowest_ce_9 = min(nine_bit_keys, key=lambda key: prefix[key]["cross_entropy"])
    mixed = {name:prefix['B'+name]['top1'] for name in ("NNA","NUA","UNA")}
    mixed_frontier = []
    for n in (2, 3):
        keys = sorted({"B"+schedule[:n] for schedule in SCHEDULES})
        for key in keys:
            if len(set(key[1:])) == 1:
                continue
            x = storage[(key,"entropy")]["total_bytes"]
            y = prefix[key]["top1"]
            if not any(other != key and storage[(other,"entropy")]["total_bytes"] <= x
                       and prefix[other]["top1"] >= y
                       and (storage[(other,"entropy")]["total_bytes"] < x or
                            prefix[other]["top1"] > y) for other in keys):
                mixed_frontier.append(key)
    nua_nna_9 = next(row for row in comparisons if row["candidate"] == "BNU" and row["reference"] == "BNN")
    nua_nna_11 = next(row for row in comparisons if row["candidate"] == "BNUA" and row["reference"] == "BNNA")
    uaa_aaa_11 = next(row for row in comparisons if row["candidate"] == "BUAA" and row["reference"] == "BAAA")
    lines += [f"**Q5 — Mixed schedules.** Eleven-bit Top-1: " + ", ".join(f"{name} {value:.3f}%" for name,value in mixed.items()) +
              f"; AAA {prefix['BAAA']['top1']:.3f}%, UUU {prefix['BUUU']['top1']:.3f}%, NNN {prefix['BNNN']['top1']:.3f}%. Mixed encoded-byte/Top-1 nondominated prefixes at 9 or 11 bits: {', '.join(mixed_frontier) if mixed_frontier else 'none'}. NUA minus NNA is {nua_nna_9['paired_top1_delta']:+.3f} pp at 9 bits and {nua_nna_11['paired_top1_delta']:+.3f} pp at 11 bits, with intervals [{nua_nna_9['paired_bootstrap_top1_ci95'][0]:+.3f}, {nua_nna_9['paired_bootstrap_top1_ci95'][1]:+.3f}] and [{nua_nna_11['paired_bootstrap_top1_ci95'][0]:+.3f}, {nua_nna_11['paired_bootstrap_top1_ci95'][1]:+.3f}]. The NUA comparisons were prespecified. " +
              f"The observed UAA advantage over AAA at 11 bits is {uaa_aaa_11['paired_top1_delta']:+.3f} pp with exploratory paired interval [{uaa_aaa_11['paired_bootstrap_top1_ci95'][0]:+.3f}, {uaa_aaa_11['paired_bootstrap_top1_ci95'][1]:+.3f}]; its coded representation is {storage[('BAAA','entropy')]['total_bytes']-storage[('BUAA','entropy')]['total_bytes']:,} bytes smaller. These small differences and exploratory frontier membership do not establish an independently validated schedule-selection method.",
              "",
              "**Q6 — Desired stop precision.** Best observed Top-1 among tested prefixes: " +
              "; ".join(f"{bits} bits {key} {value:.3f}%" for bits,(key,value) in best_by_precision.items()) +
              ". At the entropy-coded byte budgets of AAA's 7/9/11-bit prefixes, the best observed eligible prefixes are " +
              "; ".join(f"{bits}-bit AAA budget {budget:,} bytes: {winner} {value:.3f}%" for bits,budget,winner,value in budget_choices) +
              f". Among 9-bit prefixes, {lowest_ce_9} has the lowest mean cross-entropy ({prefix[lowest_ce_9]['cross_entropy']:.6f}), which can differ from the highest Top-1 prefix. These are descriptive rankings on the same validation set; the size and cross-entropy plots show the practical tradeoffs.",
              "",
              "**Q7 — Paper.** Include the controlled S/U0/U/N calibration comparison, the recursive prefix",
              "curves for AAA, UUU and NNN, and measured serialized bytes with metadata and shared state.",
              "Treat schedule rankings on this evaluation set as comparative evidence, not a validated adaptive",
              "selection algorithm. A subsequent LP versus BPC/monolithic comparison should serialize all methods",
              "with the same codec, metadata precision, FP32 shared-state accounting, and prefix decoder checks.",
              "", "## Prespecified paired comparisons", "",
              "| Candidate | Reference | Corrected | Newly incorrect | Δ Top-1 (pp) | 95% paired bootstrap CI (pp) | Δ CE |",
              "|---|---|---:|---:|---:|---|---:|"]
    for row in comparisons:
        ci = row["paired_bootstrap_top1_ci95"]
        lines.append(f"| {row['candidate']} | {row['reference']} | {row['paired_gain_images']:,} | {row['paired_loss_images']:,} | {row['paired_top1_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {row['paired_ce_delta']:+.6f} |")
    lines += ["", "The 95% intervals are paired image bootstrap intervals with 10,000 multinomial draws,",
              "seed 123. They quantify sampling uncertainty for each prespecified comparison; they do not",
              "correct for multiple comparisons or establish independently selected schedule performance.",
              "", "## Decoder and reproducibility", "",
              f"All {len(json.loads((out/'decode_verification.json').read_text()))} raw and entropy unique-prefix checks",
              "reconstructed code indices and eligible FP32 decoded weights bit exactly from serialized data.",
              "Each prefix manifest lists only streams through its own precision. The decoder loads the shared",
              "unquantized state and no original eligible FP32 checkpoint weights.", ""]
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote complete Experiment 4 report to {out}")


if __name__ == "__main__":
    main()
