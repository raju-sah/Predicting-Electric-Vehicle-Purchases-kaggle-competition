import pandas as pd
import numpy as np
from scipy.special import expit, ndtr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Let's inspect the exact formula from Hayashi on train.csv:
income = train["Annual_Income_USD"].values
concern = train["Environmental_Concern_Level"].values
subsidy = (train["Subsidy_Available"] == "Yes").astype(float).values
anxiety_low = (train["Range_Anxiety_Level"] == "Low").astype(float).values
anxiety_med = (train["Range_Anxiety_Level"] == "Medium").astype(float).values

# Hayashi raw formula:
buy_score_hayashi = (
    -8.5
    + 0.12 * (income / 10_000.0)
    + 0.6 * concern
    + 2.0 * subsidy
    + 3.0 * anxiety_low
    + 2.0 * anxiety_med
)
prob_hayashi = ndtr(buy_score_hayashi)
auc_hayashi = roc_auc_score(y, prob_hayashi)
print(f"Hayashi Exact Reconstruction Formula AUC on train.csv: {auc_hayashi:.6f}")

# Logistic regression on these 4 variables:
X_core = np.column_stack([
    income / 10000.0,
    concern,
    subsidy,
    anxiety_low,
    anxiety_med
])

lr = LogisticRegression(max_iter=1000, C=10.0)
lr.fit(X_core, y)
lr_preds = lr.predict_proba(X_core)[:, 1]
print("Logistic Regression Coefs:", lr.intercept_, lr.coef_)
print(f"Logistic Regression AUC on train.csv (just 4 core features): {roc_auc_score(y, lr_preds):.6f}")

# Probit fit using scipy minimize (negative log-likelihood)
def probit_loss(weights):
    b0 = weights[0]
    b = weights[1:]
    z = b0 + X_core @ b
    p = np.clip(ndtr(z), 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss

res = minimize(probit_loss, [-4.0, 0.08, 0.4, 1.2, 1.8, 1.2], method='BFGS')
print("Probit fitted weights:", res.x)
probit_preds = ndtr(res.x[0] + X_core @ res.x[1:])
print(f"Probit AUC on train.csv (just 4 core features): {roc_auc_score(y, probit_preds):.6f}")

# Now check income frequency feature:
all_incomes = pd.concat([train["Annual_Income_USD"], test["Annual_Income_USD"]])
inc_counts = all_incomes.value_counts()
train_inc_freq = train["Annual_Income_USD"].map(inc_counts).values
test_inc_freq = test["Annual_Income_USD"].map(inc_counts).values

print(f"Income frequency summary: min={train_inc_freq.min()}, median={np.median(train_inc_freq)}, max={train_inc_freq.max()}, unique values={len(inc_counts)}")
print(f"Correlation between income_freq and y: {np.corrcoef(train_inc_freq, y)[0, 1]:.4f}")
