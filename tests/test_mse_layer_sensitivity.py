"""Correctness checks for fixed-code LP substitution and paired evaluation."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

from analyze_mse_layer_sensitivity import (
    STAGES, aggregate_all, aggregate_diagnostics, apply_configuration,
    balanced_indices, decode_cache, eligible_groups, evaluate_details,
    metric_sums, paired_stats, stage_of, tensor_diagnostic,
)
from evaluate_imagenet import EXCLUDE_PATTERNS
from layered_quantization import IntegerFormat, parameter_is_quantized
from quantizer_variants import dequantize_variant, quantize_variant


class TinyStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(3, 3, bias=False)
        self.layer1 = nn.Linear(3, 3, bias=False)
        self.layer2 = nn.Linear(3, 3, bias=False)
        self.layer3 = nn.Linear(3, 3, bias=False)
        self.layer4 = nn.Linear(3, 3, bias=False)
        self.fc = nn.Linear(3, 6)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        return self.fc(self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x))))))


class LayerSensitivityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyStageModel().eval()
        self.groups = eligible_groups(self.model)
        self.names = [name for group in self.groups.values() for name in group]
        self.original = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.artifacts = {"base_5": {}, "affine_minmax": {}, "affine_mse": {}}
        for name in self.names:
            weight = self.original[name]
            base_params, base = IntegerFormat("channel", 0).quantize(weight, 5)
            self.artifacts["base_5"][name] = base_params
            for method in ("affine_minmax", "affine_mse"):
                self.artifacts[method][name], _ = quantize_variant(weight - base, method)
        self.cache = decode_cache(self.artifacts, self.names, self.original)

    def test_real_resnet_all_eligible_assigned_once(self):
        model = models.resnet50(weights=None)
        groups = eligible_groups(model)
        assigned = [name for names in groups.values() for name in names]
        expected = [name for name, p in model.named_parameters()
                    if parameter_is_quantized(name, p, EXCLUDE_PATTERNS, False)]
        self.assertEqual(set(groups), set(STAGES))
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(expected))
        self.assertEqual(len(assigned), 54)
        self.assertIn("layer3.0.downsample.0.weight", groups["layer3"])
        for name in assigned:
            self.assertIn(name, groups[stage_of(name)])

    def test_substitution_changes_only_target_and_preserves_excluded(self):
        apply_configuration(self.model, self.cache, set())
        affine = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        target = {"layer3.weight"}
        apply_configuration(self.model, self.cache, target, set())
        for name, tensor in self.model.state_dict().items():
            expected = self.cache["mse"][name] if name in target else affine[name]
            self.assertTrue(torch.equal(tensor, expected), name)
        self.assertTrue(torch.equal(self.model.bn.running_mean, self.original["bn.running_mean"]))
        self.assertTrue(torch.equal(self.model.fc.bias, self.original["fc.bias"]))

    def test_empty_all_and_restored_all(self):
        all_names = set(self.names)
        apply_configuration(self.model, self.cache, set())
        for name, p in self.model.named_parameters():
            if name in all_names:
                self.assertTrue(torch.equal(p, self.cache["affine"][name]))
        apply_configuration(self.model, self.cache, all_names, set())
        for name, p in self.model.named_parameters():
            if name in all_names:
                self.assertTrue(torch.equal(p, self.cache["mse"][name]))
        apply_configuration(self.model, self.cache, set(), all_names)
        for name, p in self.model.named_parameters():
            if name in all_names:
                self.assertTrue(torch.equal(p, self.cache["affine"][name]))

    def test_saved_dequantization_matches_cache(self):
        for name in self.names:
            base = IntegerFormat.dequantize(self.artifacts["base_5"][name])
            for target, method in (("affine", "affine_minmax"), ("mse", "affine_mse")):
                self.assertTrue(torch.equal(base + dequantize_variant(self.artifacts[method][name]),
                                            self.cache[target][name]))

    def test_diagnostic_sse_aggregation(self):
        rows = []
        for name in self.names:
            base = IntegerFormat.dequantize(self.artifacts["base_5"][name])
            rows.append(tensor_diagnostic(name, stage_of(name), self.original[name], base,
                                          self.cache["affine"][name], self.cache["mse"][name],
                                          self.artifacts["affine_minmax"][name],
                                          self.artifacts["affine_mse"][name]))
        all_row = aggregate_all(rows)
        self.assertEqual(all_row["weights"], sum(r["weights"] for r in rows))
        self.assertAlmostEqual(all_row["affine_residual_sse"],
                               sum(r["affine_residual_sse"] for r in rows))
        self.assertAlmostEqual(all_row["mse_residual_mse"],
                               sum(r["mse_residual_sse"] for r in rows)/all_row["weights"])
        stage = aggregate_diagnostics(rows, "layer3")
        self.assertEqual(stage["weights"], next(r["weights"] for r in rows if r["stage"] == "layer3"))

    def test_balanced_subset_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [(str(root / str(label) / f"image_{i:03d}.png"), label)
                       for label in range(10) for i in range(20)]
            order = list(reversed(samples))
            first, manifest = balanced_indices(samples, root, 5, 123)
            _, repeated = balanced_indices(order, root, 5, 123)
            self.assertEqual(manifest, repeated)
            self.assertEqual(len(first), 50)
            self.assertEqual(set(Counter(e["class_id"] for e in manifest["entries"]).values()), {5})
            self.assertNotEqual(manifest["manifest_sha256"], balanced_indices(samples, root, 5, 124)[1]["manifest_sha256"])

    def test_evaluation_and_reconstruction_reproducible(self):
        images = torch.randn(11, 3)
        labels = torch.randint(0, 6, (11,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        apply_configuration(self.model, self.cache, set())
        first, arrays1, _ = evaluate_details(self.model, loader, torch.device("cpu"), 11)
        apply_configuration(self.model, self.cache, set(self.names), set())
        apply_configuration(self.model, self.cache, set(), set(self.names))
        second, arrays2, _ = evaluate_details(self.model, loader, torch.device("cpu"), 11)
        self.assertEqual(first, second)
        for key in arrays1:
            np.testing.assert_array_equal(arrays1[key], arrays2[key])
        self.assertEqual(metric_sums(**arrays1), first)
        paired = paired_stats(arrays1, arrays2)
        self.assertEqual(paired["paired_top1_delta"], 0)
        self.assertEqual(paired["paired_gain_images"], 0)


if __name__ == "__main__":
    unittest.main()
