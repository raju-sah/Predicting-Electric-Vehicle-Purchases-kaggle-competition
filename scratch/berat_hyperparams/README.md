# 🏛️ Kaggle Grandmaster Tabular Hyperparameter Zoo (2018–2026)

[![Docs: Full Data Dictionary](https://img.shields.io/badge/Docs-Full_Data_Dictionary-2ea44f.svg)]()
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Models: LightGBM_CatBoost_XGBoost](https://img.shields.io/badge/Models-GBDT_SOTA-blue.svg)]()

An exhaustive, standardized reference dataset compiling **verified Optuna hyperparameter distributions, learning rates, tree capacities, and regularization formulas** from 1st-place and top-tier Grandmaster solutions across 50+ major Kaggle competitions.

Curated and maintained by **Berat Erol Çelik** ([@beraterolelk](https://www.kaggle.com/beraterolelk) | [GitHub](https://github.com/BerattCelikk)).

---

## 📊 Dataset Contents & Schema

| Column | Type | Description |
| :--- | :---: | :--- |
| `competition_slug` | String | Kaggle competition URL identifier |
| `competition_name` | String | Full formal competition title |
| `model_family` | String | Model framework (`LightGBM`, `CatBoost`, `XGBoost`) |
| `task_type` | String | Task category (`Binary Classification`, `Regression`) |
| `eval_metric` | String | Optimization objective (`ROC-AUC`, `RMSE`, `LogLoss`) |
| `n_estimators` | Integer | Optimal number of boosting iterations |
| `learning_rate` | Float | Shrinkage learning rate |
| `max_depth_or_num_leaves` | String | Tree capacity bounds |
| `subsample_colsample` | String | Row and column subsampling fractions |
| `regularization_l1_l2` | String | L1/L2 penalty parameters |
| `optuna_search_space_code` | String | Drop-in Python code snippet for Optuna objective |
| `winning_placement` | String | Placement achieved (1st Place, Top 1% GM) |
| `key_tuning_insight` | String | Decisive architectural takeaway |

---

## 💻 Quickstart: Drop-in Optuna Objective

```python
import optuna
import lightgbm as lgb
import pandas as pd

# Load hyperparameter zoo
zoo = pd.read_csv("kaggle_grandmaster_tabular_hyperparameters.csv")
s6e9_recipe = zoo[zoo["competition_slug"].str.contains("s6e9")].iloc[0]

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    return 0.94638 # Evaluate against your CV fold
```

---

## ⭐ Community & Upvoting

If this hyperparameter zoo accelerates your competition sweeps, please **consider upvoting (⭐)** to support reproducible data science!
