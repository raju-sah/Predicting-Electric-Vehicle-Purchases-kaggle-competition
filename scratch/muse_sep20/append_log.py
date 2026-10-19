"""MUSE-SEP20 exp_log append (append-only; no prior-row edits)."""
import csv

rows = [
    ["v75", "CatBoost-ordered-boosting digit-frame solo", "0.9433034", "",
     "First CatBoost on digit frame + 8 novel fit-free ratios; native cats; "
     "seed-42 5-fold; 2500iter lr0.03 d6; folds 0.94222/0.94307/0.94422/0.94372/0.94334; "
     "NOT submitted; script scratch/muse_sep20/train_catA.py"],
    ["v76", "HGB-native-cats digit-frame solo", "0.9432323", "",
     "First HGB on digit frame (only raw HGB baseline existed); native categoricals; "
     "seed-42 5-fold; folds 0.94211/0.94298/0.94417/0.94365/0.94329; "
     "NOT submitted; script scratch/muse_sep20/train_hgbB.py"],
    ["v77", "Fixed prob blend mem40/lgb15/xgb15/cat15/hgb15", "0.9460407", "",
     "Weights fixed a priori (build_candidates.py predates results); complete OOF; "
     "NOT submitted; subs/v75..v78 family"],
    ["v78", "Fixed rank blend mem40/lgb15/xgb15/cat15/hgb15", "0.9460654", "",
     "Weights fixed a priori; complete OOF; strongest of final four; "
     "NOT submitted; subs/v75..v78 family"],
    ["muse_stage4way", "Staged 4-way blends mem45/lgb20/xgb20/hgb15 (no catA)", "0.9462450", "",
     "Logit 0.9462450 / rank 0.9462240 / prob 0.9462151 complete OOF; fully validated; "
     "kept as documented alternates in scratch/muse_sep20/staging/; NOT submitted"],
]
with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows(rows)
print("appended", len(rows), "rows")
