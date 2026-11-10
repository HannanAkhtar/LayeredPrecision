import pytest
import json
import sys

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
    assert result["host_memory_within_allocation"]
    with pytest.raises(ValueError, match="already exists"):
        resources.record(output, time_dir, "123", 12 * 1024, 0, "s01_lp_2p2")
    monkeypatch.setattr(resources, "slurm_cgroup_peak", lambda *_:
                        (11 * 1024**3, "cgroup_v2_memory.peak"))
    failed = resources.record(tmp_path / "failed.json", time_dir,
                              "124", 12 * 1024, 1, "s01_w4")
    assert not failed["host_memory_10pct_headroom_passed"]
    assert failed["host_memory_within_allocation"]
    import sys
    monkeypatch.setattr(sys, "argv", ["resources", "--output", str(tmp_path / "within.json"),
        "--time-dir", str(time_dir), "--job-id", "125", "--allocation-mib", "12288",
        "--exit-code", "0", "--point", "s01_w4"])
    resources.main()  # 11 GiB within 12 GiB passes despite less than 10% headroom.


def test_failed_time_output_keeps_numeric_rss_and_original_exit(tmp_path, monkeypatch):
    times = tmp_path / "times"
    times.mkdir()
    (times / "encode.rss_kib").write_text("Command exited with non-zero status 7\n15000000\n")
    (times / "evaluate.rss_kib").write_text("Command was terminated by signal 9\n")
    monkeypatch.setattr(resources, "slurm_cgroup_peak", lambda *_:
                        (15_439_818_752, "cgroup_v2_memory.peak"))
    result = resources.record(tmp_path / "resources.json", times, "55386", 12 * 1024,
                              7, "s01_w4")
    assert result["stage_process_peak_rss_kib"] == {"encode": 15000000}
    assert "evaluate" in result["stage_rss_parse_diagnostics"]
    assert result["exit_code_before_resource_check"] == 7
    assert not result["host_memory_10pct_headroom_passed"]
    assert not result["host_memory_within_allocation"]
    monkeypatch.setattr(sys, "argv", ["resources", "--output", str(tmp_path / "main.json"),
        "--time-dir", str(times), "--job-id", "55386", "--allocation-mib", "12288",
        "--exit-code", "7", "--point", "s01_w4"])
    with pytest.raises(SystemExit) as error:
        resources.main()
    assert error.value.code == 7
    assert json.loads((tmp_path / "main.json").read_text())["exit_code_before_resource_check"] == 7
