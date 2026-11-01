import torch

from evaluate_lp_progressive_schedules import SCHEDULES, all_prefixes, decode
from layered_quantization import IntegerFormat
from lp_quantizer_families import quantize_family
from lp_uniform_calibration import quantize_uniform


def test_prespecified_prefix_tree_is_deduplicated():
    prefixes = all_prefixes()
    assert len(SCHEDULES) == 12
    assert len(prefixes) == 25
    assert [sum(len(prefix) == depth for prefix in prefixes) for depth in (1, 2, 3, 4)] == [1, 3, 9, 12]
    assert set(prefixes) == {"B"} | {"B" + schedule[:n] for schedule in SCHEDULES for n in (1, 2, 3)}


def test_recursive_residuals_are_specific_to_parent():
    weight = torch.tensor([[-1.2, -0.8, -0.1, 0.4, 0.8, 1.2],
                           [-0.4, -0.2, -0.1, 0.3, 0.6, 0.9]])
    _, base = IntegerFormat("channel", 0).quantize(weight, 5)
    decoded = {}
    for family in ("A", "U", "N"):
        params, _ = (quantize_uniform(weight-base, family) if family == "U"
                     else quantize_family(weight-base, family))
        decoded[family] = base + decode(params, family)
    assert not torch.equal(decoded["A"], decoded["N"])
    _, aa = quantize_family(weight-decoded["A"], "A")
    _, na = quantize_family(weight-decoded["N"], "A")
    assert not torch.equal(aa, na)
