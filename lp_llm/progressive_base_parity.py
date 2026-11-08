"""Independent zero-enhancement W2 materialization and MatGPTQ parity gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .progressive_lp import BRANCHES, identity, validate_repaired_base
from .progressive_run import TOKEN_SHA, materialize
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import atomic_json, sha256
from .progressive_resources import quota_status


def run(branch: str, source: Path, base: Path, root: Path, result: Path,
        matgptq: Path, code_sha: str, *, python: str = sys.executable) -> dict:
    if branch not in BRANCHES or len(code_sha) != 40:
        raise ValueError("Invalid base-parity identity")
    check_revision(matgptq, MATGPTQ_COMMIT)
    validate_repaired_base(branch, source, base)
    ident = identity(branch, source, base)
    marker = result / "base_parity.json"
    if marker.is_file():
        if (result / "checkpoint").exists() or (result / "checkpoint.partial").exists():
            raise FileExistsError("Completed base parity retains an unexpected temporary checkpoint")
        record = json.loads(marker.read_text())
        if (record.get("status") != "base_parity_validated" or record.get("identity") != ident or
                record.get("code_sha") != code_sha or
                sha256(result / "evaluation.log") != record.get("evaluation_log_sha256")):
            raise ValueError("Existing zero-enhancement parity differs")
        return record
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("Incomplete zero-enhancement evaluation requires inspection")
    result.mkdir(parents=True)
    checkpoint = result / "checkpoint"
    try:
        reconstructed = materialize(branch, 0, source, base, root, checkpoint)
        corpus = corpus_identity(checkpoint)
        if corpus["token_sha256"] != TOKEN_SHA or corpus["scored_tokens"] != 298862:
            raise ValueError("Zero-enhancement corpus identity differs")
        measurement = evaluate_matgptq(matgptq, checkpoint, result / "evaluation.log", python=python)
        if round(measurement["perplexity"], 2) != {"s01": 12.47, "r02": 32.22}[branch]:
            raise ValueError("Independent W2 materialization lost perplexity parity")
        record = {"status": "base_parity_validated", "identity": ident,
                  "quota_status": quota_status(),
                  "code_sha": code_sha, "checkpoint_validation": reconstructed,
                  "corpus": corpus, "evaluation": measurement,
                  "evaluation_log_sha256": sha256(result / "evaluation.log")}
        atomic_json(marker, record)
        shutil.rmtree(checkpoint)  # only this job's new, verified temporary directory
        return record
    except Exception as error:
        atomic_json(result / "failure.json", {"status": "failed",
                                               "error_type": type(error).__name__,
                                               "error": str(error), "identity": ident})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--matgptq", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.branch, args.source, args.base, args.root, args.result,
                         args.matgptq, args.code_sha), indent=2))


if __name__ == "__main__":
    main()
