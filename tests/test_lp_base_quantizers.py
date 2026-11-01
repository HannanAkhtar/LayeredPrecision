import torch

from lp_base_quantizers import GAMMAS, dequantize_base, quantize_base


def test_five_bit_roundtrip_and_determinism():
    torch.manual_seed(7)
    examples = (torch.randn(3, 81), torch.zeros(2, 13),
                torch.stack((torch.full((13,), .25), torch.full((13,), -.75))),
                torch.tensor([[-3., -1., 0., .1, .5, 2.], [0., 0., 0., 2., 5., 9.]]))
    for family in ("A", "P", "U0", "U", "N", "L"):
        for weight in examples:
            params, decoded = quantize_base(weight, family)
            again, decoded_again = quantize_base(weight, family)
            assert params["bits"] == 5
            assert params["quantized"].dtype == torch.uint8
            assert 0 <= params["quantized"].min() and params["quantized"].max() < 32
            assert torch.equal(decoded, dequantize_base(params))
            assert torch.equal(decoded, decoded_again)
            assert torch.equal(params["quantized"], again["quantized"])


def test_calibrated_uniform_includes_midpoint():
    torch.manual_seed(9)
    weight = torch.randn(4, 150)
    _, fixed = quantize_base(weight, "U0")
    _, calibrated = quantize_base(weight, "U")
    assert len(GAMMAS) == 25 and abs(GAMMAS[12]-1) < 1e-12
    assert ((weight-calibrated).double().square().sum(1) <=
            (weight-fixed).double().square().sum(1)+1e-12).all()


def test_log_and_lloyd_metadata():
    weight = torch.linspace(-1, 1, 512).reshape(2, 256)
    log, _ = quantize_base(weight, "L")
    lloyd, _ = quantize_base(weight, "N")
    assert log["scale"].numel() == 2
    assert lloyd["centers"].shape == (2, 32)
    assert (lloyd["centers"][:, 1:] >= lloyd["centers"][:, :-1]).all()


def test_five_bit_families_define_all_32_codes():
    weight = torch.linspace(-1, 1, 4096).reshape(1, -1)
    for family in ("A", "P", "U", "N", "L"):
        params, _ = quantize_base(weight, family)
        params["quantized"] = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
        levels = dequantize_base(params).reshape(-1)
        assert len(levels) == 32
        assert (levels[1:] > levels[:-1]).all()
