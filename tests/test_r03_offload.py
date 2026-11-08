import json
from types import SimpleNamespace

import pytest

from lp_llm.r03_offload import GIB, OffloadLease, offload_root, quota_free_bytes


def test_r03_offload_cleans_only_owned_directory(tmp_path, monkeypatch):
    root = tmp_path / "r03_offload"
    root.mkdir()
    sibling = root / "other-job"
    sibling.mkdir()
    (sibling / "weights").write_text("preserve")
    monkeypatch.setenv("R03_SHARED_QUOTA_FREE_GIB", "100")
    monkeypatch.setenv("SLURM_JOB_ID", "53899")
    monkeypatch.setattr("lp_llm.r03_offload.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=100 * GIB))
    assert offload_root(root, allowed_root=tmp_path) == root.resolve()
    with OffloadLease.create(root) as lease:
        owned = lease.directory
        assert owned.parent == root.resolve()
        (owned / "weights").mkdir()
        (owned / "weights/data").write_text("temporary")
    assert not owned.exists()
    assert (sibling / "weights").read_text() == "preserve"


def test_r03_offload_refuses_quota_and_tampered_ownership(tmp_path, monkeypatch):
    root = tmp_path / "r03_offload"
    root.mkdir()
    monkeypatch.setattr("lp_llm.r03_offload.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=100 * GIB))
    monkeypatch.setenv("R03_SHARED_QUOTA_FREE_GIB", "10")
    with pytest.raises(ValueError, match="capacity or declared free quota"):
        OffloadLease.create(root)
    monkeypatch.setenv("R03_SHARED_QUOTA_FREE_GIB", "100")
    lease = OffloadLease.create(root)
    marker = lease.directory / ".owner.json"
    value = json.loads(marker.read_text())
    value["token"] = "not-owned"
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ownership marker changed"):
        lease.cleanup()
    assert lease.directory.exists()
    another = OffloadLease.create(root)
    another.cleanup()
    assert lease.directory.exists()  # Orphaned job state is never removed by a new job.


def test_r03_offload_root_and_quota_validation(tmp_path, monkeypatch):
    root = tmp_path / "r03_offload"
    root.mkdir()
    monkeypatch.setenv("R03_SHARED_QUOTA_FREE_GIB", "not-known")
    with pytest.raises(ValueError, match="available-quota"):
        quota_free_bytes()
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="must remain under"):
        offload_root(outside, allowed_root=tmp_path)
