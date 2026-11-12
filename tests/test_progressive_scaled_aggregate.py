from lp_llm.progressive_scaled_aggregate import aggregate


def test_missing_matrix_rows_remain_missing(tmp_path):
    result = aggregate(tmp_path)
    assert len(result["rows"]) == 15  # original plus two seven-point branches
    assert result["rows"][0]["perplexity"] == 7.01
    assert all(row["status"] == "missing" and row["perplexity"] is None
               for row in result["rows"][1:])
