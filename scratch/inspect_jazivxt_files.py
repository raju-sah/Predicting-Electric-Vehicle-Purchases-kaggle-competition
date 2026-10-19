import json
import pandas as pd
import numpy as np

print("--- Inspecting Jazivxt single-model-zoom-zoom outputs ---")

with open("scratch/kernels_pulled/jazivxt_zoom_out/submission_provenance.json") as f:
    prov = json.load(f)
print("Provenance JSON:")
print(json.dumps(prov, indent=2))

cv_df = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/cv_scores_lightgbm.csv")
print("\nCV scores:")
print(cv_df)

sub = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")
print("\nsubmission.csv shape:", sub.shape)
print("Unique values in submission.csv:", sub["Will_Buy_EV"].nunique())
print("Ties in submission.csv:", len(sub) - sub["Will_Buy_EV"].nunique())

if "submission_latest_adjusted.csv" in [f for f in dir()]:
    pass
sub_adj = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission_latest_adjusted.csv")
print("\nsubmission_latest_adjusted.csv shape:", sub_adj.shape)
print("Unique values in submission_latest_adjusted.csv:", sub_adj["Will_Buy_EV"].nunique())
print("Ties in submission_latest_adjusted.csv:", len(sub_adj) - sub_adj["Will_Buy_EV"].nunique())
