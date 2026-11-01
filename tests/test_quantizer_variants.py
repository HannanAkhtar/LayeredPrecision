"""Deterministic synthetic checks for the first LP residual quantizers."""

import unittest

import torch

from layered_quantization import IntegerFormat, LayeredQuantizer
from quantizer_variants import VARIANTS, dequantize_variant, quantize_variant


class QuantizerVariantTests(unittest.TestCase):
    def test_symmetric_levels_and_zero(self):
        values = torch.tensor([[-2., -1.2, 0., 1.2, 2.]])
        params, result = quantize_variant(values, "symmetric_minmax")
        self.assertEqual(params["quantized"].tolist(), [[-1, -1, 0, 1, 1]])
        self.assertEqual(result.tolist(), [[-2., -2., 0., 2., 2.]])
        self.assertEqual(params["effective_levels"], 3)
        self.assertNotIn(-2, params["quantized"].unique().tolist())

    def test_affine_codes_and_zero_point(self):
        values = torch.tensor([[-1., 0., 1., 2.]])
        params, result = quantize_variant(values, "affine_minmax")
        self.assertEqual(params["quantized"].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(params["zero_point"].item(), 1)
        self.assertEqual(result.tolist(), [[-1., 0., 1., 2.]])

    def test_channel_axis_zero(self):
        values = torch.tensor([[-2., 0., 2.], [-20., 0., 20.]])
        params, result = quantize_variant(values, "symmetric_minmax")
        self.assertEqual(params["scale"].flatten().tolist(), [2., 20.])
        self.assertEqual(params["axis"], 0)
        self.assertTrue(torch.equal(result, values))

    def test_percentile_clips_outliers(self):
        values = torch.tensor([[0., 0., 0., 1., 1., 1., 100.]])
        symmetric, sym_recon = quantize_variant(values, "symmetric_percentile",
                                                 clip_percentile=80)
        affine, aff_recon = quantize_variant(values, "affine_percentile",
                                             clip_percentile=80)
        self.assertLess(symmetric["scale"].item(), 100)
        self.assertLess(affine["scale"].item(), 100 / 3)
        self.assertLess(sym_recon[0, -1].item(), 100)
        self.assertLess(aff_recon[0, -1].item(), 100)
        self.assertTrue(torch.equal(affine["quantized"].max(), torch.tensor(3, dtype=torch.uint8)))

    def test_constant_and_zero(self):
        for variant in VARIANTS:
            for values in (torch.zeros(2, 5), torch.full((2, 5), 1e-30),
                           torch.full((2, 5), 1e-45),
                           torch.full((2, 5), -3.0)):
                params, result = quantize_variant(values, variant)
                self.assertTrue(torch.isfinite(result).all(), variant)
                self.assertTrue(torch.equal(result, dequantize_variant(params)), variant)
                self.assertEqual(result.shape, values.shape)
                self.assertEqual(result.dtype, values.dtype)
                self.assertEqual(result.device, values.device)
                codes = params["quantized"]
                if variant.startswith("symmetric"):
                    self.assertTrue(((codes >= -1) & (codes <= 1)).all(), variant)
                else:
                    self.assertTrue(((codes >= 0) & (codes <= 3)).all(), variant)

    def test_mse_calibration_never_worse_than_baseline(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(4, 37, generator=generator)
        values[0, 0] = 15
        base, base_recon = quantize_variant(values, "affine_minmax")
        optimized, opt_recon = quantize_variant(values, "affine_mse")
        base_error = (values - base_recon).double().square().mean(dim=1)
        opt_error = (values - opt_recon).double().square().mean(dim=1)
        self.assertTrue(torch.all(opt_error <= base_error + 1e-12))
        self.assertTrue(((optimized["zero_point"] >= 0) &
                         (optimized["zero_point"] <= 3)).all())
        self.assertEqual(base["quantized"].dtype, torch.uint8)

    def test_roundtrip_and_determinism(self):
        values = torch.tensor([[-7., -1., 0., 1., 2., 20.],
                               [-0.5, -0.1, 0., 0.2, 0.4, 0.7]])
        for variant in VARIANTS:
            first, first_recon = quantize_variant(values, variant)
            second, second_recon = quantize_variant(values, variant)
            self.assertTrue(torch.equal(first_recon, dequantize_variant(first)))
            self.assertTrue(torch.equal(first_recon, second_recon))
            for key in ("quantized", "scale", "zero_point"):
                self.assertTrue(torch.equal(first[key], second[key]), (variant, key))

    def test_affine_is_original_implementation(self):
        values = torch.tensor([[-2., -0.1, 0., 3.], [1., 2., 4., 7.]])
        original, original_recon = IntegerFormat("channel", 0).quantize(values, 2)
        variant, variant_recon = quantize_variant(values, "affine_minmax")
        for key in ("quantized", "scale", "zero_point", "t_min"):
            self.assertTrue(torch.equal(original[key], variant[key]))
        self.assertTrue(torch.equal(original_recon, variant_recon))

    def test_shared_base_matches_original_lp(self):
        weights = torch.tensor([[-2.3, -1.2, 0.4, 3.7], [0.1, 1.4, 3.1, 6.2]])
        base_params, base = IntegerFormat("channel", 0).quantize(weights, 5)
        residual = weights - base
        lp = LayeredQuantizer("channel", 0).quantize_layered(weights, [5, 2])
        self.assertTrue(torch.equal(base_params["quantized"],
                                    lp["layers_info"][0]["quant_params"]["quantized"]))
        for variant in VARIANTS:
            _, contribution = quantize_variant(residual, variant)
            self.assertTrue(torch.equal(weights - base, residual))
            if variant == "affine_minmax":
                self.assertTrue(torch.equal(base + contribution,
                    LayeredQuantizer.reconstruct_from_layers(lp["layers_info"])))


if __name__ == "__main__":
    unittest.main()
