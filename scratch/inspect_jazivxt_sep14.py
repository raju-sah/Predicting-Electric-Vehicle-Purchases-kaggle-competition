import json
import pandas as pd

with open("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_provenance.json") as f:
    prov = json.load(f)

print("Jazivxt Sep 14 Provenance:")
print(json.dumps(prov, indent=2))

cv_df = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/cv_scores_lightgbm.csv")
print("\nCV scores:")
print(cv_df)
