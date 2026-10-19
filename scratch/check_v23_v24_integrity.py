import pandas as pd
import numpy as np

sample = pd.read_csv("sample_submission.csv")

for f in ["subs/v23_pure_10model_meta_stacker.csv", "subs/v24_meta_stack_50_50_balanced.csv"]:
    df = pd.read_csv(f)
    assert list(df.columns) == ["id", "Will_Buy_EV"]
    assert len(df) == 286571
    assert (df["id"] == sample["id"]).all()
    assert not df["Will_Buy_EV"].isna().any()
    assert not np.isinf(df["Will_Buy_EV"]).any()
    assert df["Will_Buy_EV"].nunique() == 286571
    print(f"VERIFIED {f}: 100% VALID! Zero ties, identical IDs.")
