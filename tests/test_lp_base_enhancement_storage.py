import json

import torch

from evaluate_lp_base_enhancement import decode
from lp_base_enhancement_storage import manifest, read_prefix, read_stream, write_stream
from lp_base_quantizers import quantize_base
from lp_quantizer_families import quantize_family
from lp_uniform_calibration import quantize_uniform


def test_all_stream_families_and_precisions(tmp_path):
    weight = torch.stack((torch.linspace(-1, 1, 193),
                          torch.linspace(-.1, 2, 193)))
    for regime in ("raw", "entropy"):
        for family in ("A", "P", "U", "N", "L"):
            base, decoded = quantize_base(weight, family)
            path = tmp_path / f"{regime}-{family}-5.bin"
            stats = write_stream(path, {"w": base}, family, 5, regime)
            restored_family, bits, restored = read_stream(path)
            assert (restored_family, bits) == (family, 5)
            assert torch.equal(base["quantized"], restored["w"]["quantized"])
            assert torch.equal(decoded, decode(restored["w"], family, True))
            assert stats["physical_bytes"] == path.stat().st_size
            residual = weight-decoded
            if family == "U":
                enhancement, corrected = quantize_uniform(residual, "U")
            else:
                enhancement, corrected = quantize_family(residual, family)
            path = tmp_path / f"{regime}-{family}-2.bin"
            write_stream(path, {"w": enhancement}, family, 2, regime)
            restored_family, bits, restored = read_stream(path)
            assert (restored_family, bits) == (family, 2)
            assert torch.equal(enhancement["quantized"], restored["w"]["quantized"])
            assert torch.equal(corrected, decode(restored["w"], family, False))


def test_prefix_stops_without_future_stream(tmp_path):
    root = tmp_path / "raw"
    (root / "streams").mkdir(parents=True)
    torch.save({"bias": torch.tensor([2.])}, root / "shared_fp32_state.pt")
    weight = torch.linspace(-1, 1, 97).reshape(1, -1)
    base, first = quantize_base(weight, "U")
    residual, second = quantize_uniform(weight-first, "U")
    write_stream(root / "streams/B_U.bin", {"w": base}, "U", 5, "raw")
    write_stream(root / "streams/B_U_E1_U.bin", {"w": residual}, "U", 2, "raw")
    full = manifest(root, "B_U_E1_U")
    info = json.loads(full.read_text())
    assert len(info["streams"]) == 2
    stopped = read_prefix(full, stop_key="B_U")
    assert torch.equal(stopped["w"], first)
    assert torch.equal(read_prefix(full)["w"], first+second)
