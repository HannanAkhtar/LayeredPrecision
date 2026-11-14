import json
import os
import sys
import types

import pytest

from lp_llm import progressive_downstream as ds
from lp_llm.storage import sha256


def test_registry_has_exact_fifteen_models():
    assert len(ds.MODELS) == 15
    assert len(set(ds.MODELS)) == 15
    assert ds.PILOTS == ("original_fp16", "s01_lp6", "r02_w6")
    assert ds.MODELS[0] == "original_fp16"
    assert {f"{branch}_{suffix}" for branch in ("s01", "r02")
            for suffix in ("w2", "lp4", "lp6", "lp8", "w4", "w6", "w8")} == set(ds.MODELS[1:])


def test_task_metrics_keep_accuracy_normalization_and_stderr():
    manifest = {"datasets": {name: {"samples": 5} for name in ds.TASKS}}
    raw = {"results": {name: {"acc,none": 0.6, "acc_stderr,none": 0.1,
                               "acc_norm,none": 0.7 if name != "winogrande" else None}
                       for name in ds.TASKS},
           "n-samples": {name: {"effective": 5} for name in ds.TASKS}}
    result = ds._metrics(raw, manifest, ds.TASKS)
    assert result["arc_easy"]["acc_stderr"] == 0.1
    assert result["piqa"]["acc_norm"] == 0.7
    assert result["winogrande"]["acc_norm"] is None
    raw["n-samples"]["piqa"]["effective"] = 4
    with pytest.raises(ValueError, match="sample count"):
        ds._metrics(raw, manifest, ds.TASKS)


def test_pilot_is_separate_idempotent_and_never_quantizes(tmp_path, monkeypatch):
    source, root, output = (tmp_path / name for name in ("source", "results", "output"))
    source.mkdir(); root.mkdir(); output.mkdir()
    checkpoint = source
    monkeypatch.setattr(ds, "_historical", lambda *_: (checkpoint, {"verified": True}))
    fake = types.ModuleType("lm_eval")
    fake.simple_evaluate = lambda **kw: {"results": {task: {"acc,none": 0.5}
                                                for task in kw["tasks"]}}
    utils = types.ModuleType("lm_eval.utils")
    utils.handle_non_serializable = str
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    monkeypatch.setitem(sys.modules, "lm_eval.utils", utils)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    (output / "manifest.json").write_text("{}")
    manifest = {"code_sha": "a" * 40}
    first = ds.evaluate(source, root, output, "original_fp16", manifest,
                        pilot=True, scratch=tmp_path)
    assert first["pilot"] is True
    assert set(first["tasks"]) == set(ds.TASKS[:2])
    assert (output / "pilot/original_fp16/full.json").is_file()
    assert not (output / "models/original_fp16").exists()
    assert ds.evaluate(source, root, output, "original_fp16", manifest,
                       pilot=True, scratch=tmp_path) == first
    with pytest.raises(ValueError, match="unsupported pilot"):
        ds.evaluate(source, root, output, "s01_w8", manifest, pilot=True, scratch=tmp_path)


def test_aggregate_missing_rows_and_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "perplexity_matrix", lambda _: {"rows": [
        {"id": "original_fp16", "perplexity": 7.01}]})
    output = tmp_path / "out"
    target = output / "models/original_fp16"
    target.mkdir(parents=True)
    (output / "manifest.json").write_text("{}")
    (target / "harness.json").write_text("{}")
    full = {"status": "completed_validated", "pilot": False,
            "manifest_sha256": sha256(output / "manifest.json"),
            "harness_sha256": sha256(target / "harness.json"),
            "tasks": {name: {"acc": 0.2 * (i + 1), "acc_norm": None}
                      for i, name in enumerate(ds.TASKS)}}
    (target / "full.json").write_text(json.dumps(full))
    rows = ds.aggregate(tmp_path, output)["rows"]
    assert len(rows) == 15
    assert rows[0]["mean_acc"] == pytest.approx(0.6)
    assert rows[1]["status"] == "missing"


def test_historical_layer_sha_is_scoped_and_tampering_fails(tmp_path, monkeypatch):
    source = tmp_path / "source"
    root = tmp_path / "results"
    old_sha = "a" * 40
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", "b" * 40)
    monkeypatch.setattr(ds, "scaled_identity", lambda *_: {"verified": True})
    checked = []
    def layers(branch, depth, source, base, root, ident):
        assert os.environ["FOLLOWUP_CODE_SHA"] == old_sha
        checked.append(depth)
        return [{"manifest": {"code_sha": old_sha}} for _ in range(depth)]
    monkeypatch.setattr(ds, "layers", layers)
    folder = root / "progressive_llm/scaled_signed_v1/r02/evaluation/depth2"
    folder.mkdir(parents=True)
    (folder / "evaluation.log").write_text("historical")
    full = {"status": "completed_validated", "identity": {"verified": True},
            "depth": 2, "code_sha": old_sha, "evaluation_log_sha256": sha256(folder / "evaluation.log"),
            "materialization": {"shard_sha256": {"a": "hash"}}}
    (folder / "full.json").write_text(json.dumps(full))
    checkpoint, provenance = ds._historical(root, "r02_lp6", source)
    assert checkpoint is None and provenance["historical_producer_sha"] == old_sha
    assert checked == [2] and os.environ["FOLLOWUP_CODE_SHA"] == "b" * 40
    monkeypatch.setattr(ds, "layers", lambda *_: [{"manifest": {"code_sha": "wrong"}}] * 2)
    with pytest.raises(ValueError, match="producer differs"):
        ds._historical(root, "r02_lp6", source)


def test_preflight_seal_requires_all_checks_and_resumes(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    for name in ("manifest.json", "environment.json", "task_manifest.json"):
        (root / name).write_text("{}")
    manifest = {"code_sha": "c" * 40}
    with pytest.raises(FileNotFoundError, match="seal required"):
        ds._preflight_seal(root, manifest)
    entries = {name: (None, {"id": name}) for name in ds.MODELS}
    seal = ds._preflight_seal(root, manifest, entries)
    assert seal["status"] == "completed_validated"
    assert ds._preflight_seal(root, manifest, entries) == seal
    (root / "task_manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="seal differs"):
        ds._preflight_seal(root, manifest)


def test_cgroup_snapshot_reads_job_path_not_mount_root(tmp_path, monkeypatch):
    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/slurm/job123\n")
    mount = tmp_path / "sys/fs/cgroup"
    job = mount / "slurm/job123"
    job.mkdir(parents=True)
    (job / "memory.stat").write_text("anon 100\nfile 200\n")
    (job / "memory.peak").write_text("400")
    (job / "memory.max").write_text("500")
    monkeypatch.setattr(ds, "Path", lambda value: tmp_path / str(value).lstrip("/"))
    value = ds._cgroup_snapshot()
    assert value["cgroup_path"] == str(job)
    assert value["anon_bytes"] == 100 and value["file_bytes"] == 200
    assert value["peak_bytes"] == 400 and value["limit_bytes"] == 500


def test_preflight_reuses_hashes_then_rechecks_inputs(tmp_path, monkeypatch):
    from lp_llm import progressive_lp
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("original")
    calls = []
    def original(directory):
        calls.append(1)
        return {"config_sha256": sha256(source / "config.json")}
    monkeypatch.setattr(progressive_lp, "_files", original)
    monkeypatch.setattr(ds, "registry", lambda *_: [ds._files(source) for _ in range(15)])
    ds._preflight_registry(source, tmp_path)
    assert len(calls) == 2  # initial hash and final independent recheck


def test_interrupted_preflight_manifest_resumes_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_revision", lambda *_: None)
    monkeypatch.setattr(ds, "_task_configs", lambda *_: {"arc_easy": {"sha256": "pinned"}})
    monkeypatch.setattr(ds, "_versions", lambda: {"torch": "2.9.1+cu128"})
    path = tmp_path / "manifest.json"
    manifest = {"code_sha": "a" * 40, "harness_sha": ds.HARNESS_SHA,
                "tasks": ds._task_configs(tmp_path), "versions": ds._versions(),
                "protocol": {"model": "hf", "dtype": "float16", "num_fewshot": 0,
                             "batch_size": 1, "apply_chat_template": False,
                             "limit": None, "seeds": [0, 1234, 1234, 1234]},
                "datasets": {"arc_easy": {"fingerprint": "cached"}}}
    path.write_text(json.dumps(manifest))
    saved = ds._manifest(path, "a" * 40, tmp_path, create=True)
    assert saved == manifest
    assert json.loads((tmp_path / "environment.json").read_text()) == manifest["versions"]
    assert not (tmp_path / "preflight_complete.json").exists()
    (tmp_path / "environment.json").write_text('{"torch":"wrong"}')
    with pytest.raises(ValueError, match="environment.json changed"):
        ds._manifest(path, "a" * 40, tmp_path, create=True)


def test_lp_materialization_uses_historical_sha_and_exact_shard_hashes(tmp_path, monkeypatch):
    class Usage:
        free = 30 * 1024**3
    monkeypatch.setattr(ds.shutil, "disk_usage", lambda _: Usage())
    def materialize(*args, **kwargs):
        assert kwargs["historical_layer_sha"] == "a" * 40
        return {"shard_sha256": {"model.safetensors": "exact"},
                "tensor_count": 399, "projection_count": 252}
    monkeypatch.setattr(ds, "materialize", materialize)
    provenance = {"branch": "r02", "depth": 2, "base": str(tmp_path),
                  "lp_root": str(tmp_path), "historical_producer_sha": "a" * 40,
                  "historical_materialization": {"shard_sha256": {"model.safetensors": "exact"},
                      "tensor_count": 399, "projection_count": 252}}
    assert ds._materialize_lp(tmp_path, "r02_lp6", provenance, tmp_path / "checkpoint") == \
        tmp_path / "checkpoint"
    provenance["historical_materialization"]["shard_sha256"] = {"model.safetensors": "tampered"}
    with pytest.raises(ValueError, match="differs from evaluated weights"):
        ds._materialize_lp(tmp_path, "r02_lp6", provenance, tmp_path / "checkpoint")
