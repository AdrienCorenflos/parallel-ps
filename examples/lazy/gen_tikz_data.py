import numpy as np
import pandas as pd

# This should be read from the results, but it's easier this way for now.
SVS = [0.2, 0.3, 0.4, 0.5]
TS = [32, 64, 128, 256, 512]
NS = [25, 50, 100, 250, 500, 1_000, 2_500, 5_000]

lazy_res = np.load(f"./output/rare-events-False-True.npz")
seq_res = np.load(f"./output/rare-events-True-True.npz")
par_res = np.load(f"./output/rare-events-False-False.npz")

par_runtime_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)
lazy_runtime_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)
seq_runtime_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)

for i in range(len(SVS)):
    par_runtime_df.iloc[i] = par_res['runtimes'][i].T.ravel()
    lazy_runtime_df.iloc[i] = lazy_res['runtimes'][i].T.ravel()
    seq_runtime_df.iloc[i] = seq_res['runtimes'][i].T.ravel()

par_var_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)
lazy_var_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)
seq_var_df = pd.DataFrame(columns=pd.MultiIndex.from_product([TS, NS]), index=SVS)

for i in range(len(SVS)):
    par_var_df.iloc[i] = par_res['variances'][i].T.ravel()
    lazy_var_df.iloc[i] = lazy_res['variances'][i].T.ravel()
    seq_var_df.iloc[i] = seq_res['variances'][i].T.ravel()

print(lazy_runtime_df)
print(lazy_var_df)

print(seq_runtime_df)
print(seq_var_df)
