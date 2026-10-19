import pandas as pd
import numpy as np

print("Auditing Annual_Income_USD frequency and generator properties...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = (train["Will_Buy_EV"] == "Yes").astype(int).values

all_incomes = pd.concat([train["Annual_Income_USD"], test["Annual_Income_USD"]])
inc_counts = all_incomes.value_counts()
train_inc_freq = train["Annual_Income_USD"].map(inc_counts).values
test_inc_freq = test["Annual_Income_USD"].map(inc_counts).values

print(f"Total rows: train={len(train):,}, test={len(test):,}, total={len(all_incomes):,}")
print(f"Unique income values: {len(inc_counts):,}")
print(f"Incomes appearing in both train and test: {len(set(train['Annual_Income_USD']).intersection(set(test['Annual_Income_USD']))):,} / {len(set(test['Annual_Income_USD'])):,}")
print(f"Train income freq stats: min={train_inc_freq.min()}, median={np.median(train_inc_freq)}, max={train_inc_freq.max()}, mean={train_inc_freq.mean():.2f}")
print(f"Test income freq stats: min={test_inc_freq.min()}, median={np.median(test_inc_freq)}, max={test_inc_freq.max()}, mean={test_inc_freq.mean():.2f}")

# Check correlation with y
corr = np.corrcoef(train_inc_freq, y)[0, 1]
print(f"Pearson correlation with Will_Buy_EV: {corr:.5f}")

# Group by income frequency deciles
df_audit = pd.DataFrame({"freq": train_inc_freq, "y": y, "income": train["Annual_Income_USD"]})
df_audit["freq_bin"] = pd.qcut(df_audit["freq"], q=10, duplicates="drop")
print("\nTarget rate by income frequency bin:")
print(df_audit.groupby("freq_bin", observed=True)["y"].agg(["count", "mean"]))

# Look at top and bottom frequencies
print("\nTop 10 most frequent incomes and their EV purchase rate:")
top_incomes = inc_counts.head(10).index
for inc in top_incomes:
    sub = df_audit[df_audit["income"] == inc]
    print(f"  Income: ${inc:,.0f} | Total count (tr+te): {inc_counts[inc]:4d} | Train count: {len(sub):4d} | EV rate: {sub['y'].mean():.4f}")
