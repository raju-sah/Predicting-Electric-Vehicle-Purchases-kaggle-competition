"""Round2 model D: gated rank-student (pseudo15 mechanism, OUR gate folds).

Student: shallow LGBM (7 leaves, min 4000 rows/leaf) on 6 frozen OOF ranks
(memory/lgb6/xgb5/catA/hgbB/disw) + raw numeric + native cats, one model per
outer fold. Teacher: fixed rank blend mem40/lgb15/xgb15/cat15/hgb15 (v78).
Blend 20% student rank into teacher rank INSIDE each valid fold.
Acceptance (same bar as source): pooled delta > 0 AND >=4/5 fold wins.
Else the student is killed and logged as such. No tuning, no peeking.

Leak safety: student features are frozen out-of-fold predictions + raw inputs;
labels only from its own outer-train rows. Standard stacked generalization.
Folds: exact StratifiedKFold(5, shuffle=True, random_state=42).
Outputs: scratch/muse_sep20_round2/oof_stu.npy, test_stu.npy, run_stu.json
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "Will_Buy_EV"
OUT = "scratch/muse_sep20_round2"
RAW = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
W_T = [0.40, 0.15, 0.15, 0.15, 0.15]  # teacher weights, pre-registered (v78)
W_S = 0.20  # student dose, frozen per source screen


def r01(a):
    return (rankdata(a, method="average") - 0.5) / len(a)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()

mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")
assert (mem["id"] == train["id"]).all()
lgb6 = pd.read_csv("scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_oof.csv")["OOF_Pred"].to_numpy(float)
xgb5 = pd.read_csv("scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_oof.csv")["OOF_Pred"].to_numpy(float)
catA = np.load("scratch/muse_sep20/oof_catA.npy")
hgbB = np.load("scratch/muse_sep20/oof_hgbB.npy")
disw = np.load(f"{OUT}/oof_disw.npy")
t_mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/submission_model_only.csv")["Will_Buy_EV"].to_numpy(float)
t_lgb6 = pd.read_csv("scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_test.csv")["Will_Buy_EV"].to_numpy(float)
t_xgb5 = pd.read_csv("scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_test.csv")["Will_Buy_EV"].to_numpy(float)
t_catA = np.load("scratch/muse_sep20/test_catA.npy")
t_hgbB = np.load("scratch/muse_sep20/test_hgbB.npy")
t_disw = np.load(f"{OUT}/test_disw.npy")

OOF6 = [mem["oof_final"].to_numpy(float), lgb6, xgb5, catA, hgbB, disw]
TEST6 = [t_mem, t_lgb6, t_xgb5, t_catA, t_hgbB, t_disw]
R_OOF = np.column_stack([r01(a) for a in OOF6])
R_TEST = np.column_stack([r01(a) for a in TEST6])
T5_OOF = [r01(a) for a in [OOF6[0], OOF6[1], OOF6[2], OOF6[3], OOF6[4]]]
T5_TEST = [r01(a) for a in [TEST6[0], TEST6[1], TEST6[2], TEST6[3], TEST6[4]]]
teacher_oof = sum(w * a for w, a in zip(W_T, T5_OOF))
teacher_test = sum(w * a for w, a in zip(W_T, T5_TEST))

levels = {c: pd.Index(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATS}
Xraw = train[RAW].copy()
Xpraw = test[RAW].copy()
for c in CATS:
    Xraw[c] = pd.Categorical(train[c].fillna(levels[c][0]), categories=levels[c])
    Xpraw[c] = pd.Categorical(test[c].fillna(levels[c][0]), categories=levels[c])
feat_names = [f"r{i}" for i in range(6)] + RAW + CATS
Xp = pd.concat([pd.DataFrame(R_TEST, columns=feat_names[:6]), Xpraw.reset_index(drop=True)], axis=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(cv.split(y, y)):
    fold[va] = k
oof_s = np.zeros(len(y), np.float64)
test_s = np.zeros(len(test), np.float64)
acc_oof = np.zeros(len(y), np.float64)
fold_rows = []
for kf, (fit_idx, valid_idx) in enumerate(cv.split(y, y)):
    Xf = pd.concat([pd.DataFrame(R_OOF[fit_idx], columns=feat_names[:6]),
                    Xraw.iloc[fit_idx].reset_index(drop=True)], axis=1)
    Xv = pd.concat([pd.DataFrame(R_OOF[valid_idx], columns=feat_names[:6]),
                    Xraw.iloc[valid_idx].reset_index(drop=True)], axis=1)
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05,
                               num_leaves=7, min_child_samples=4000,
                               subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5.0,
                               random_state=42 + kf, verbose=-1)
    model.fit(Xf, y[fit_idx], eval_set=[(Xv, y[valid_idx])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_s[valid_idx] = model.predict_proba(Xv)[:, 1]
    test_s += model.predict_proba(Xp)[:, 1] / cv.n_splits
    t_r = r01(teacher_oof[valid_idx])
    s_r = r01(oof_s[valid_idx])
    acc_oof[valid_idx] = (1 - W_S) * t_r + W_S * s_r
    print(f"fold {kf}: student_trees={model.best_iteration_} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)

t_auc = roc_auc_score(y, teacher_oof)
c_auc = roc_auc_score(y, acc_oof)
deltas = [roc_auc_score(y[fold == k], acc_oof[fold == k])
          - roc_auc_score(y[fold == k], teacher_oof[fold == k]) for k in range(5)]
accepted = bool(c_auc > t_auc and sum(d > 0 for d in deltas) >= 4)
np.save(f"{OUT}/oof_stu.npy", oof_s)
np.save(f"{OUT}/test_stu.npy", test_s)
np.save(f"{OUT}/oof_stu_accepted_blend.npy", acc_oof)
json.dump({"teacher_oof_auc": t_auc, "candidate_oof_auc": c_auc,
           "delta": c_auc - t_auc, "fold_deltas": deltas,
           "wins": int(sum(d > 0 for d in deltas)), "accepted": accepted,
           "folds": "StratifiedKFold(5, shuffle=True, random_state=42)"},
          open(f"{OUT}/run_stu.json", "w"), indent=2)
print(f"STU teacher={t_auc:.7f} candidate={c_auc:.7f} "
      f"delta={c_auc-t_auc:+.7f} wins={sum(d > 0 for d in deltas)}/5 "
      f"accepted={accepted}", flush=True)
