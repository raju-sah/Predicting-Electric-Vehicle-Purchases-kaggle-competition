from pathlib import Path
import pandas as pd
import numpy as np
root = Path('scratch/pull_sep19/najiama_refresh')
for directory in ['najiama_lgb_output', 'najiama_xgb_output']:
    for output in Path('scratch/research_next', directory).glob('*.csv'):
        df = pd.read_csv(output)
        print('SOURCE', output, df.shape, list(df.columns))
        pred = df.iloc[:, -1].to_numpy()
        for candidate in root.rglob('*.csv'):
            other = pd.read_csv(candidate)
            if len(other) != len(pred):
                continue
            v = other.iloc[:, -1].to_numpy()
            if np.array_equal(pred, v):
                print('EXACT MATCH', candidate)
            elif np.issubdtype(v.dtype, np.number):
                print(candidate.name, 'max_abs', np.max(np.abs(pred-v)), 'corr', np.corrcoef(pred,v)[0,1])
