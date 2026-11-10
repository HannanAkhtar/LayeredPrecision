import pytest

from lp_llm import progressive_job_resources as resources


def test_full_job_resource_record_keeps_stage_rss_and_cgroup_peak(tmp_path, monkeypatch):
    time_dir = tmp_path / "times"
    time_dir.mkdir()
    (time_dir / "encode.rss_kib").write_text(str(8 * 1024**2))
    (time_dir / "evaluate.rss_kib").write_text(str(9 * 1024**2))
    monkeypatch.setattr(resources, "slurm_cgroup_peak", lambda *_:
                        (10 * 1024**3, "cgroup_v2_memory.peak"))
    output = tmp_path / "resource.json"
    result = resources.record(output, time_dir, "123", 12 * 1024, 0, "s01_lp_2p2")
    assert result["stage_process_peak_rss_kib"]["evaluate"] == 9 * 1024**2
    assert result["slurm_cgroup_peak_bytes"] == 10 * 1024**3
    assert result["host_memory_10pct_headroom_passed"]
    with pytest.raises(ValueError, match="already exists"):
        resources.record(output, time_dir, "123", 12 * 1024, 0, "s01_lp_2p2")
    monkeypatch.setattr(resources, "slurm_cgroup_peak", lambda *_:
                        (11 * 1024**3, "cgroup_v2_memory.peak"))
    failed = resources.record(tmp_path / "failed.json", time_dir,
                              "124", 12 * 1024, 1, "s01_w4")
    assert not failed["host_memory_10pct_headroom_passed"]
