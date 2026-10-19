"""Analyze the JZ scored archive to find optimal combinations.
The archive has 485 scored public predictions. Finding the right
subset to average could produce a stronger anchor than any single one.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
import zipfile, os, glob, time

BASE = "/home/raju/AI-ML Projects/Kaggle Competition/Predicting Electric Vehicle Purchases"
T0 = time.time()

# Load the catalog
catalog = pd.read_csv(f"{BASE}/scratch/pull_sep16/jz_meta/ranked_catalog_latest.csv")
print(f"Catalog: {catalog.shape}")
print(f"Score distribution: {catalog['public_score'].describe()}")
print(f"\nTop 20 scores:")
print(catalog.sort_values("public_score", ascending=False).head(20)[["origin", "author", "public_score", "sha256", "csv_path"]].to_string())

# Find the exact 0.94651 files
top_files = catalog[catalog["public_score"] >= 0.94650]
print(f"\n=== Files scoring >= 0.94650: {len(top_files)} ===")
print(top_files[["origin", "author", "public_score", "sha256"]].to_string())

# Check if the JZ full archive exists
archive_path = f"{BASE}/scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip"
if os.path.exists(archive_path):
    print(f"\n=== Archive found: {os.path.getsize(archive_path)/1e9:.2f} GB ===")
    
    # Load a sample of top-scoring predictions
    with zipfile.ZipFile(archive_path, 'r') as zf:
        all_files = zf.namelist()
        csv_files = [f for f in all_files if f.endswith('.csv')]
        print(f"CSV files in archive: {len(csv_files)}")
        
        # Load the best submission (the 0.94651 anchor)
        anchor_path = None
        for f in csv_files:
            if "own_001" in f:
                anchor_path = f
                break
        
        if anchor_path:
            print(f"\nLoading anchor: {anchor_path}")
            anchor_df = pd.read_csv(zf.open(anchor_path))
            print(f"Anchor shape: {anchor_df.shape}")
            print(f"Anchor head: {anchor_df.head()}")
else:
    print("Archive not found, checking for extracted files...")
    
# Also check the JZ meta directory for pre-extracted sibling files
extra_files = glob.glob(f"{BASE}/scratch/pull_sep16/jz_meta/extra_*.csv")
print(f"\nExtra sibling files: {len(extra_files)}")
for f in extra_files:
    print(f"  {os.path.basename(f)}")

# Load the submission_latest_best.csv
best_path = f"{BASE}/scratch/pull_sep16/jz_meta/submission_latest_best.csv"
if os.path.exists(best_path):
    best_df = pd.read_csv(best_path)
    print(f"\nBest submission: {best_df.shape}")
    print(f"Columns: {best_df.columns.tolist()}")
    print(f"Head:\n{best_df.head()}")

# Load the existing best v57 submission for comparison
v57_path = f"{BASE}/subs/v57_public_band_swap_exact.csv"
if os.path.exists(v57_path):
    v57_df = pd.read_csv(v57_path)
    print(f"\nv57 band-swap submission: {v57_df.shape}")
    
    if os.path.exists(best_path):
        jz_preds = best_df["Will_Buy_EV"].to_numpy()
        v57_preds = v57_df["Will_Buy_EV"].to_numpy()
        
        rho, _ = spearmanr(jz_preds, v57_preds)
        print(f"Spearman ρ(JZ anchor, v57): {rho:.8f}")
        
        # How many rows differ significantly?
        diff = np.abs(jz_preds - v57_preds)
        print(f"Rows with >0.01 diff: {(diff > 0.01).sum()}")
        print(f"Rows with >0.001 diff: {(diff > 0.001).sum()}")
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.8f}")

# Now: analyze the scored predictions to understand band distribution
print("\n=== Score distribution analysis ===")
score_counts = catalog["public_score"].value_counts().sort_index(ascending=False).head(20)
print(score_counts)

# Key question: can we find a subset of 0.94650 files to average
# that produces a better ranking than any single 0.94651 file?
top_0_94650 = catalog[(catalog["public_score"] >= 0.94649) & (catalog["public_score"] <= 0.94651)]
print(f"\nFiles in [0.94649, 0.94651]: {len(top_0_94650)}")
for _, row in top_0_94650.iterrows():
    print(f"  {row['public_score']:.5f} | {row['origin']} | {row['author']} | {row['csv_path']}")

print(f"\nTotal analysis time: {time.time()-T0:.1f}s")
