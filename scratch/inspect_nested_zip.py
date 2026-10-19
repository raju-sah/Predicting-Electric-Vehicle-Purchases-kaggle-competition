import zipfile
import io
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata

archive = "scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip"
print("Opening outer archive...")
with zipfile.ZipFile(archive, 'r') as z:
    print("Reading nested zip.bin...")
    with z.open("ranked_predictions_latest.zip.bin") as nested:
        with zipfile.ZipFile(nested) as inner_zip:
            names = inner_zip.namelist()
            print(f"Total files in nested zip: {len(names)}")
            print("First 10 files:", names[:10])
