import zipfile
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata

archive = "scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip"
catalog = pd.read_csv("scratch/pull_sep16/jz_meta/latest_scored_results.csv")

# Get unique submissions at 0.94650 and 0.94651
top = catalog[catalog["public_score"] >= 0.94650].copy()
print(f"Total entries >= 0.94650: {len(top)}")
print(f"Unique sha256: {top['sha256'].nunique()}")

# Let's inspect the files in nested zip
with zipfile.ZipFile(archive, 'r') as z:
    with z.open("ranked_predictions_latest.zip.bin") as nested:
        with zipfile.ZipFile(nested) as inner_zip:
            all_files = set(inner_zip.namelist())
            print(f"Total files in inner zip: {len(all_files)}")
            
            # Check availability of top files
            available = []
            for idx, row in top.iterrows():
                path = row["csv_path"]
                if path in all_files:
                    available.append((row["submission_id"], row["public_score"], row["sha256"], path))
                else:
                    print(f"Missing: {path}")
            print(f"Available top files: {len(available)}")

            # Load predictions for unique sha256
            unique_preds = {}
            for sub_id, score, sha, path in available:
                if sha not in unique_preds:
                    df = pd.read_csv(inner_zip.open(path))
                    col = [c for c in df.columns if c != "id"][0]
                    unique_preds[sha] = (sub_id, score, df[col].to_numpy(float))
            
            print(f"Loaded {len(unique_preds)} unique predictions.")
            
            # Check correlations
            shas = list(unique_preds.keys())
            for i in range(len(shas)):
                for j in range(i+1, len(shas)):
                    s1, sc1, p1 = unique_preds[shas[i]]
                    s2, sc2, p2 = unique_preds[shas[j]]
                    rho = spearmanr(p1, p2).statistic
                    print(f"Sub {s1} ({sc1}) vs Sub {s2} ({sc2}): rho={rho:.8f}")
