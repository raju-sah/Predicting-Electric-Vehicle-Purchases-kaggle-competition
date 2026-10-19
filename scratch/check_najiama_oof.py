import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = (train['Will_Buy_EV'] == 'Yes').astype(int).values

naji = pd.read_csv('scratch/najiama_out/oof_XGBOOST.csv')
print("Columns in najiama OOF:", naji.columns.tolist())
print("Row count:", len(naji))
print("Match ID:", (naji['id'] == train['id']).all())

for col in naji.columns:
    if col != 'id':
        try:
            score = roc_auc_score(y_true, naji[col])
            print(f"Column '{col}' AUC: {score:.7f}")
        except Exception as e:
            print(f"Col {col} error: {e}")
