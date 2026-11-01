import pytest
import torch

from lp_llm.encode import encode
from lp_llm.reconstruct import reconstruct


def test_tiny_qwen3_transformers_reload_and_forward(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Qwen3 requires Transformers >= 4.51")
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True, max_shard_size="10KB")
    assert (source / "model.safetensors.index.json").exists()
    del model
    for method, bits, granularity in (("lp", [2, 2, 2, 2], "channel"),
                                       ("lp", [2, 2, 2, 2], "group"),
                                       ("monolithic", [4], "channel")):
        artifact = tmp_path / f"{method}-{granularity}"
        manifest = encode(source, artifact, {"method": method, "bits": bits,
                                              "granularity": granularity, "group_size": 128})
        for depth in range(1, len(manifest["bits"]) + 1):
            out = tmp_path / f"decoded-{method}-{granularity}-{depth}"
            reconstruct(artifact, out, depth)
            loaded = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
            with torch.no_grad():
                logits = loaded(torch.tensor([[1, 2, 3]])).logits
            assert logits.shape == (1, 3, 64) and torch.isfinite(logits).all()
