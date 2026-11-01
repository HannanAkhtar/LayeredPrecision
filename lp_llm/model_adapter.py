"""Checkpoint inventory and Qwen3 parameter selection, independent of quantization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


@dataclass(frozen=True)
class SelectionPolicy:
    projections: tuple[str, ...] = PROJECTIONS

    def eligible(self, name: str) -> bool:
        match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.weight", name)
        return bool(match and match[3] in self.projections and
                    ((match[2] == "self_attn") == (match[3] in _ATTN)))


def expected_qwen3(config: dict) -> dict[str, tuple[int, ...]]:
    if config.get("model_type") != "qwen3" or config.get("tie_word_embeddings") is not False:
        raise ValueError("Expected untied Qwen3 causal LM configuration")
    layers, hidden, intermediate = (int(config[k]) for k in
                                    ("num_hidden_layers", "hidden_size", "intermediate_size"))
    heads, kv_heads, head_dim = (int(config[k]) for k in
                                ("num_attention_heads", "num_key_value_heads", "head_dim"))
    vocab = int(config["vocab_size"])
    shapes = {"model.embed_tokens.weight": (vocab, hidden),
              "model.norm.weight": (hidden,), "lm_head.weight": (vocab, hidden)}
    for i in range(layers):
        p = f"model.layers.{i}."
        shapes.update({p + "self_attn.q_proj.weight": (heads * head_dim, hidden),
                       p + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                       p + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                       p + "self_attn.o_proj.weight": (hidden, heads * head_dim),
                       p + "self_attn.q_norm.weight": (head_dim,),
                       p + "self_attn.k_norm.weight": (head_dim,),
                       p + "mlp.gate_proj.weight": (intermediate, hidden),
                       p + "mlp.up_proj.weight": (intermediate, hidden),
                       p + "mlp.down_proj.weight": (hidden, intermediate),
                       p + "input_layernorm.weight": (hidden,),
                       p + "post_attention_layernorm.weight": (hidden,)})
    return shapes


def inventory(source: Path, policy: SelectionPolicy = SelectionPolicy()) -> tuple[dict, dict]:
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    expected = expected_qwen3(config)
    if set(mapping) != set(expected):
        raise ValueError(f"Checkpoint key mismatch: missing={sorted(set(expected)-set(mapping))}, "
                         f"unexpected={sorted(set(mapping)-set(expected))}")
    for shard in sorted(set(mapping.values())):
        path = source / shard
        if not path.is_file() or path.parent.resolve() != source.resolve():
            raise ValueError(f"Missing or invalid shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            assigned = {name for name, file in mapping.items() if file == shard}
            if actual != assigned:
                raise ValueError(f"Shard {shard} keys differ from index: {actual ^ assigned}")
            for name in sorted(assigned):
                sl = handle.get_slice(name)
                if tuple(sl.get_shape()) != expected[name]:
                    raise ValueError(f"Shape mismatch for {name}: {sl.get_shape()} != {expected[name]}")
                if str(sl.get_dtype()) not in ("torch.bfloat16", "BF16"):
                    raise ValueError(f"Expected BF16 tensor: {name}, got {sl.get_dtype()}")
    return config, mapping


def read_tensor(source: Path, mapping: dict[str, str], name: str):
    # safe_open maps the shard and materializes only this tensor on get_tensor.
    with safe_open(Path(source) / mapping[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)
