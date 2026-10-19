"""S6E9 v37 - modulo-leak + digit + triple-TE LGBM seed 123. OOF 0.94642 local. CPU ~10 min."""
import glob, os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

def find_input(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if hits:
        print(f"using {hits[0]}")
        return hits[0]
    return f'/tmp/kaggle-grandmaster/s6e9-data/{name}'

train = pd.read_csv(find_input('train.csv'))
test_raw = pd.read_csv(find_input('test.csv'))
TARGET = 'Will_Buy_EV'
train[TARGET] = train[TARGET].map({'Yes': 1, 'No': 0})
orig = train.sample(frac=0.15, random_state=42).copy()
train['is_train'] = 1
test = test_raw.copy(); test['is_train'] = 0; test[TARGET] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined.drop(columns=['Number_of_Cars_Owned'], inplace=True, errors='ignore')
cat_cols = combined.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = [c for c in combined.columns if c not in cat_cols + ['id', 'is_train', TARGET]]
print("digit feats...")
digit_features = []
for c in num_cols:
    for k in range(-4, 4):
        cn = f"{c}_digit{k}"
        combined[cn] = (combined[c].fillna(0) // (10**k) % 10).astype('int8')
        digit_features.append(cn)
num_cols = num_cols + digit_features
orig_global_mean = orig[TARGET].mean()
for col in cat_cols + [c for c in num_cols if c in orig.columns]:
    if col in orig.columns:
        m = orig.groupby(col, observed=False)[TARGET].mean()
        combined[f"{col}_org_mean"] = combined[col].map(m).fillna(orig_global_mean).astype(float)
num_to_cat = []
for col in num_cols:
    cn = f"{col}_cat"
    combined[cn] = combined[col].fillna('NaN').astype(str)
    num_to_cat.append(cn)
all_cats = cat_cols + num_to_cat
for col in all_cats:
    fm = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_fe"] = combined[col].map(fm).astype(float).fillna(0.0)
combined['is_30k_spike'] = (combined['Annual_Income_USD'] == 30000.0).astype('int8')
combined['is_millionaire_cliff'] = (combined['Annual_Income_USD'] >= 170537.0).astype('int8')
combined['is_dead_zone'] = ((combined['Annual_Income_USD'] >= 31004.0) & (combined['Annual_Income_USD'] <= 41970.0)).astype('int8')
combined['is_env_hater'] = (combined['Environmental_Concern_Level'] == 1).astype('int8')
combined['income_exact_int'] = np.floor(combined['Annual_Income_USD']).astype(str)
combined['income100_floor'] = np.floor(combined['Annual_Income_USD'] / 100.0).astype(str)
combined['income1000_floor'] = np.floor(combined['Annual_Income_USD'] / 1000.0).astype(str)
combined['commute_integer'] = np.floor(combined['Daily_Commute_km']).astype(str)
all_cats.extend(['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer'])
# v36 NEW: modulo-leak feats (Deotte-style synthetic generator fingerprints)
combined['inc_mod1000'] = (combined['Annual_Income_USD'] % 1000).astype(str)
combined['inc_mod100'] = (combined['Annual_Income_USD'] % 100).astype(str)
combined['inc_mod10'] = (combined['Annual_Income_USD'] % 10).astype(str)
combined['commute_mod1'] = (combined['Daily_Commute_km'] % 1).round(4).astype(str)
combined['age_mod5'] = (combined['Age'] % 5).astype(str)
all_cats.extend(['inc_mod1000', 'inc_mod100', 'inc_mod10', 'commute_mod1', 'age_mod5'])
train_df = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test_df = combined[combined['is_train'] == 0].drop(columns=['is_train', TARGET])
eval_cols = [c for c in train_df.columns if c not in ['id', TARGET] and pd.api.types.is_numeric_dtype(train_df[c])]
corr = train_df[eval_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] == 1.0)]
to_drop += [c for c in train_df.columns if train_df[c].nunique() == 1] + [c for c in test_df.columns if test_df[c].nunique() == 1]
DROP = [c for c in set(to_drop) if c not in ['id', TARGET]]
train_df.drop(columns=DROP, inplace=True, errors='ignore')
test_df.drop(columns=DROP, inplace=True, errors='ignore')
FEATURES = [c for c in test_df.columns if c != 'id']
TARGET_ENCODE_COLS = [c for c in all_cats if c not in DROP and c in train_df.columns]
print(f"FEATURES {len(FEATURES)} TE_COLS {len(TARGET_ENCODE_COLS)}")

X = train_df[FEATURES]; y = train_df[TARGET]; Xt = test_df[FEATURES]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
for fold, (a, b) in enumerate(skf.split(X, y), 1):
    Xa, ya = X.iloc[a].copy(), y.iloc[a]
    Xb, yb = X.iloc[b].copy(), y.iloc[b]
    Xtf = Xt.copy()
    Xa0 = Xa[TARGET_ENCODE_COLS].copy(); Xb0 = Xb[TARGET_ENCODE_COLS].copy(); Xt0 = Xtf[TARGET_ENCODE_COLS].copy()
    encs = {}
    for smooth, tag in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
        te = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=123)
        encs[tag] = (te.fit_transform(Xa0, ya), te.transform(Xb0), te.transform(Xt0))
    for i, col in enumerate(TARGET_ENCODE_COLS):
        for tag in ['auto', '10', '100']:
            ea, eb, et = encs[tag]
            Xa[f"{col}_TE_{tag}"] = ea[:, i].astype('float32')
            Xb[f"{col}_TE_{tag}"] = eb[:, i].astype('float32')
            Xtf[f"{col}_TE_{tag}"] = et[:, i].astype('float32')
    Xa.drop(columns=TARGET_ENCODE_COLS, inplace=True, errors='ignore')
    Xb.drop(columns=TARGET_ENCODE_COLS, inplace=True, errors='ignore')
    Xtf.drop(columns=TARGET_ENCODE_COLS, inplace=True, errors='ignore')
    clf = lgb.LGBMClassifier(n_estimators=15000, learning_rate=0.02, max_depth=5,
        num_leaves=32, min_child_samples=10, subsample=0.8, colsample_bytree=0.3,
        reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, random_state=123,
        feature_pre_filter=False, metric='auc', n_jobs=8, verbose=-1)
    clf.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[b] = clf.predict_proba(Xb)[:, 1]; tp += clf.predict_proba(Xtf)[:, 1] / 5
    print(f"fold {fold} iter {clf.best_iteration_} AUC {roc_auc_score(yb, oof[b]):.5f}", flush=True)
print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")

# calibration (cell 12 replica)
t = test_raw
inc = pd.to_numeric(t['Annual_Income_USD'], errors='coerce').to_numpy()
com = pd.to_numeric(t['Daily_Commute_km'], errors='coerce').to_numpy()
sub_no = (t['Subsidy_Available'].astype(str) == 'No').to_numpy()
env1 = (pd.to_numeric(t['Environmental_Concern_Level'], errors='coerce') == 1.0).to_numpy()
anx = t['Range_Anxiety_Level'].isin(['Medium', 'High']).to_numpy()
order = np.lexsort((tp, tp)); ranks = np.empty(len(order), dtype=np.int64); ranks[order] = np.arange(1, len(order)+1)
cal = (ranks - 0.5) / len(ranks)
cal[inc >= 170537.0] += 10.0
cal[(inc >= 31004.0) & (inc <= 41970.0)] -= 10.0
cal[com >= 83.0] -= 5.0
cal[(inc == 30000.0) & sub_no & (env1 | anx)] -= 5.0
of = np.lexsort((tp, cal)); rf = np.empty(len(of), dtype=np.int64); rf[of] = np.arange(1, len(of)+1)
final = (rf - 0.5) / len(rf)
final = np.round(final, 5) + np.arange(len(final)) * 1e-9
final = np.clip(final, 1e-6, 1-1e-6)
pd.DataFrame({'id': t['id'], 'Will_Buy_EV': final}).to_csv('/kaggle/working/submission.csv', index=False)
print("saved /kaggle/working/submission.csv - OOF printed above")
