"""Synthetic checks for the four-code families and isolated LP prefixes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lp_quantizer_families import (FAMILIES, dequantize_family, pack_two_bit_codes,
                                   quantize_family, read_binary_codebook,
                                   unpack_two_bit_codes, write_binary_codebook)
from quantizer_variants import quantize_variant
from lp_residual_evolution import characterize_tensor, summarize_channels
from evaluate_lp_quantizer_adaptivity import (calibrate_candidate, numerical_metrics,
                                               reconstruct_candidate)


class QuantizerAdaptivityTests(unittest.TestCase):
    def test_all_families_finite_and_deterministic_on_edge_cases(self):
        cases = [torch.zeros(3, 41), torch.full((3, 41), 1.25),
                 torch.tensor([[-2., -1., 0., 1., 2.]]*3),
                 torch.tensor([[-.2, -.1, -.02, .01, 4.]]*3),
                 torch.full((3, 41), 1e-18)]
        for family in FAMILIES:
            for case in cases:
                params, decoded = quantize_family(case, family)
                repeat, repeated = quantize_family(case, family)
                self.assertTrue(torch.equal(params["quantized"], repeat["quantized"]), family)
                self.assertTrue(torch.equal(decoded, repeated), family)
                self.assertTrue(torch.equal(decoded, dequantize_family(params)), family)
                self.assertTrue(torch.isfinite(decoded).all(), family)
                self.assertEqual(decoded.shape, case.shape)
                self.assertGreaterEqual(int(params["quantized"].min()), 0)
                self.assertLessEqual(int(params["quantized"].max()), 3)

    def test_a_p_exact_original_implementations(self):
        residual = torch.tensor([[-2., -.1, 0., .2, 3., 7.],
                                 [-.1, -.01, 0., .1, .2, 4.]])
        for family, method in (("A", "affine_minmax"), ("P", "affine_percentile")):
            original, original_decoded = quantize_variant(residual, method)
            params, decoded = quantize_family(residual, family)
            for key in ("quantized", "scale", "zero_point"):
                self.assertTrue(torch.equal(original[key], params[key]), (family, key))
            self.assertTrue(torch.equal(original_decoded, decoded))

    def test_signed_grids_have_four_reachable_codes_and_zero_bias(self):
        residual = torch.tensor([[-8., -2., 0., 2., 8.]])
        for family in ("S", "L"):
            params, decoded = quantize_family(residual, family)
            self.assertEqual(set(params["quantized"].flatten().tolist()), {0, 1, 2, 3})
            self.assertLess(decoded[0, 2].item(), 0)
            self.assertNotEqual(decoded[0, 2].item(), 0)

    def test_lloyd_nonincreasing_objective(self):
        generator = torch.Generator().manual_seed(7)
        residual = torch.randn(8, 300, generator=generator)
        residual[0, 0] = 15
        params, decoded = quantize_family(residual, "N")
        history = params["objective_history"]
        self.assertTrue(all(next_value <= value + 1e-8*max(1, value)
                            for value, next_value in zip(history, history[1:])))
        self.assertTrue(torch.equal(decoded, dequantize_family(params)))
        self.assertTrue((params["centers"][:, 1:] >= params["centers"][:, :-1]).all())

    def test_bitpack_and_binary_roundtrip(self):
        values = np.arange(35, dtype=np.uint8) % 4
        self.assertTrue(np.array_equal(values, unpack_two_bit_codes(pack_two_bit_codes(values), len(values))))
        residual = torch.tensor([[-2., -1., 0., 1., 2., 3.], [0., 0., 0., 0., 0., 0.]])
        for family in FAMILIES:
            params, decoded = quantize_family(residual, family)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "codes.bin"
                sizes = write_binary_codebook(path, family, {"layer.weight": params})
                read_family, loaded = read_binary_codebook(path)
                self.assertEqual(read_family, family)
                self.assertEqual(sizes["physical_bytes"], path.stat().st_size)
                self.assertEqual(sizes["raw_code_bytes"], 3)
                self.assertTrue(torch.equal(decoded, dequantize_family(loaded["layer.weight"], family)))

    def test_residual_statistics_handle_degenerate_channels_and_weighting(self):
        residual = torch.tensor([[0., 0., 0., 0.], [-2., -1., 1., 2.]])
        affine, _ = quantize_family(residual, "A")
        clipped, _ = quantize_family(residual, "P")
        rows, raw_hist, normal_hist = characterize_tensor("conv1.weight", 1, residual,
                                                          affine, clipped)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["degenerate_std"])
        self.assertIsNone(rows[0]["normalized_mean"])
        self.assertEqual(rows[0]["zero_fraction"], 1.0)
        self.assertTrue(np.isfinite(raw_hist).all())
        self.assertTrue(np.isfinite(normal_hist).all())
        # Summary uses total squared magnitude / total weight count for RMS.
        aggregate = summarize_channels(rows, 1, "all")
        expected = np.sqrt(sum(r["weights"]*r["rms"]**2 for r in rows)/8)
        self.assertAlmostEqual(aggregate["pooled_residual_rms"], expected)
        self.assertEqual(sum(aggregate[f"code{k}_count"] for k in range(4)), 8)

    def test_isolated_candidates_share_incoming_decoded_residual(self):
        original = {"conv1.weight": torch.tensor([[.1, -.3, .5, 1.2],
                                                   [-.7, .2, .4, .9]])}
        preceding = {"conv1.weight": torch.tensor([[.08, -.28, .48, 1.18],
                                                    [-.68, .18, .38, .88]])}
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            stats = [calibrate_candidate(outdir, 2, family, original, preceding,
                                         device=torch.device("cpu"))
                     for family in FAMILIES]
            self.assertEqual(len({row["incoming_residual_mse"] for row in stats}), 1)
            for row in stats:
                decoded = reconstruct_candidate(outdir, 2, row["family"], preceding)
                metric = numerical_metrics(original, decoded)
                self.assertAlmostEqual(metric["parameter_mse"], row["parameter_mse"])
                self.assertEqual(row["raw_code_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
