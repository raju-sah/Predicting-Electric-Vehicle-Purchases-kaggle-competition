from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_submission.py <submission.csv>")

    candidate_path = Path(sys.argv[1])
    sample_path = Path("sample_submission.csv")
    candidate = pd.read_csv(candidate_path)
    sample = pd.read_csv(sample_path)

    expected_columns = ["id", "Will_Buy_EV"]
    if candidate.columns.tolist() != expected_columns:
        raise ValueError(f"columns must be {expected_columns}, got {candidate.columns.tolist()}")
    if len(candidate) != len(sample):
        raise ValueError(f"row count mismatch: {len(candidate)} != {len(sample)}")
    if not candidate["id"].equals(sample["id"]):
        raise ValueError("candidate IDs do not exactly match sample_submission.csv order")

    predictions = candidate["Will_Buy_EV"].to_numpy()
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain missing or infinite values")
    if not ((predictions >= 0.0) & (predictions <= 1.0)).all():
        raise ValueError("predictions must all be in [0, 1]")

    print(
        f"valid: {candidate_path} rows={len(candidate)} "
        f"min={predictions.min():.9f} max={predictions.max():.9f}"
    )


if __name__ == "__main__":
    main()
