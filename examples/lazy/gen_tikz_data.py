import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from examples.lazy.rare_events import NS, TS, SVS

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


