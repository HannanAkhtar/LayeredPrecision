import json

from lp_llm.progressive_scaled_aggregate import aggregate
from lp_llm.progressive_scaled_aggregate import TOKEN_SHA
from lp_llm.storage import sha256


def test_missing_matrix_rows_remain_missing(tmp_path):
    result = aggregate(tmp_path)
    assert len(result["rows"]) == 15  # original plus two seven-point branches
    assert result["rows"][0]["perplexity"] == 7.01
    assert all(row["status"] == "missing" and row["perplexity"] is None
               for row in result["rows"][1:])


def test_controls_report_existing_verified_distortion_and_storage(tmp_path):
    paths = (
        tmp_path / "progressive_llm/controls/r02_w4/full.json",
        tmp_path / "progressive_llm/scaled_signed_v1/control_recovery/s01_w4/evaluation/full.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True)
        log = path.parent / "evaluation.log"
        log.write_text("ppl output")
        path.write_text(json.dumps({"status": "completed_validated",
            "corpus": {"token_sha256": TOKEN_SHA, "scored_tokens": 298862},
            "evaluation": {"perplexity": 13.38}, "evaluation_log_sha256": sha256(log),
            "checkpoint_validation": {"distortion": {"mse_fp32": 0.125,
                "sqnr_db_fp32": 27.0}, "actual_checkpoint_file_bytes": 456}}))
    rows = {row["id"]: row for row in aggregate(tmp_path)["rows"]}
    for name in ("r02_w4", "s01_w4"):
        assert rows[name]["mse_fp32"] == 0.125
        assert rows[name]["sqnr_db_fp32"] == 27.0
        assert rows[name]["physical_reconstruction_bytes"] == 456
        assert rows[name]["storage_basis"] == "fake_qdq_checkpoint"
        assert rows[name]["perplexity"] == 13.38
