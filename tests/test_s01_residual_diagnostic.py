import json

import pytest

from lp_llm import s01_residual_diagnostic as diagnostic
from test_progressive_lp import fixture


def test_exact_endpoint_streams_and_checks_all_projections(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=130)
    monkeypatch.setattr(diagnostic, "validate_repaired_base", lambda *_: {})
    report_path = tmp_path / "diagnostic/exact.json"
    report = diagnostic.exact_endpoint(source, base, report_path, "a" * 40, rows=1)
    assert report["projection_count"] == 252
    assert report["excluded_count"] == 1
    assert report["eligible_elements"] == 252 * 2 * 130
    assert report["fp16_differing_elements"] == 0
    assert report["exact_mse_after_fp16_cast"] == 0
    assert len(report["by_block"]) == 36
    assert diagnostic.exact_endpoint(source, base, report_path, "a" * 40) == report
    with pytest.raises(ValueError, match="Existing exact endpoint"):
        diagnostic.exact_endpoint(source, base, report_path, "b" * 40)


def test_stored_eval_protocol_is_required(tmp_path):
    root = tmp_path / "results"
    paths = [root / "quantizer_followups/s01_fp16_load_repair/full.json",
             root / "progressive_llm/evaluation/s01_lp_2p2/full.json",
             root / "progressive_llm/ablations/s01_autoround_signed_rtn/evaluation/full.json"]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": "completed_validated", "code_sha": str(index),
            "corpus": {"token_sha256": diagnostic.TOKEN_SHA,
                       "scored_tokens": diagnostic.SCORED},
            "evaluation": {"perplexity": 12.47 + index}}))
    assert set(diagnostic._records(root)) == {"w2", "ternary_e2", "signed_e2"}
    changed = json.loads(paths[2].read_text())
    changed["corpus"]["scored_tokens"] -= 1
    paths[2].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="protocol differs"):
        diagnostic._records(root)


def test_validation_sweep_requires_exact_endpoint(tmp_path):
    root = tmp_path / "results"
    with pytest.raises(FileNotFoundError):
        diagnostic.validation_sweep(tmp_path / "source", tmp_path / "base", root,
                                    tmp_path / "screen.json", "a" * 40)
    with pytest.raises(ValueError, match="fixed 8-block"):
        diagnostic.validation_sweep(tmp_path / "source", tmp_path / "base", root,
                                    tmp_path / "screen.json", "a" * 40, blocks=3)


def test_original_baseline_uses_official_protocol_once(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=128)
    ident = diagnostic.identity("s01", source, base)
    pinned_source = {"checkpoint": "original", "config_sha256": "config",
                     "index_sha256": "index", "shard_sha256": {"shard": "hash"},
                     "sidecar_sha256": {"tokenizer.json": "token-hash"}}
    prior = {key: {"identity": {"source": ident["source"],
                                "repaired_base": ident["repaired_base"]}}
             for key in ("ternary_e2", "signed_e2")}
    prior["w2"] = {"identity": {"id": "s01_fp16_load_repair",
                              "original_run_identity": {"source": pinned_source}}}
    monkeypatch.setattr(diagnostic, "_records", lambda *_: prior)
    monkeypatch.setattr("lp_llm.rrq_pr2308.source_identity", lambda *_: pinned_source)
    monkeypatch.setattr(diagnostic, "check_revision", lambda *_: None)
    monkeypatch.setattr(diagnostic, "validate_repaired_base", lambda *_: {})
    monkeypatch.setattr(diagnostic, "corpus_identity", lambda path: {
        "token_sha256": diagnostic.TOKEN_SHA, "scored_tokens": diagnostic.SCORED})
    calls = []
    def evaluate(_checkout, checkpoint, log):
        calls.append(checkpoint)
        log.write_text("wikitext2: 7.01")
        return {"perplexity": 7.01}
    monkeypatch.setattr(diagnostic, "evaluate_matgptq", evaluate)
    output = tmp_path / "baseline"
    first = diagnostic.baseline(source, base, tmp_path, tmp_path, output, "a" * 40)
    assert first["evaluation"]["perplexity"] == 7.01
    assert calls == [source]
    assert diagnostic.baseline(source, base, tmp_path, tmp_path, output, "a" * 40) == first
    assert calls == [source]
    (output / "evaluation.log").write_text("modified")
    with pytest.raises(ValueError, match="Existing original baseline"):
        diagnostic.baseline(source, base, tmp_path, tmp_path, output, "a" * 40)
    prior["w2"]["identity"]["original_run_identity"]["source"] = {
        **pinned_source, "config_sha256": "wrong"}
    with pytest.raises(ValueError, match="source or base identity differs"):
        diagnostic.baseline(source, base, tmp_path, tmp_path, output, "a" * 40)
