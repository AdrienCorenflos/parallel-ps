import numpy as np
import pandas as pd
import seaborn
import seaborn as sns

# This should be read from the results, but it's easier this way for now.
import tikzplotlib
from matplotlib import pyplot as plt

SVS = [0.2, 0.3, 0.4, 0.5]
TS = [32,  64, 128, 256, 512]
NS = [25, 50, 100, 250, 500, 1_000, 2_500, 5_000]



lazy_res = np.load("./output/rare-events-False-True.npz")
seq_res = np.load("./output/rare-events-True-True.npz")
par_res = np.load("./output/rare-events-False-False.npz")


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


runtime_df = pd.concat({"sys-dSMC": par_runtime_df, "FFBS": seq_runtime_df, "rs-dSMC": lazy_runtime_df})
var_df = pd.concat({"sys-dSMC": par_var_df, "FFBS": seq_var_df, "rs-dSMC": lazy_var_df})


df = pd.concat({"runtime": runtime_df, "variance": var_df}, axis=1)
df.columns.names = ["statistic", "T", "N"]
df.index.names = ["algorithm", "sigma"]
df = df.stack([1, 2]).reset_index()
df[df["variance"] < 1e-6] = np.nan


g = seaborn.FacetGrid(data=df, col="T", row="sigma", hue="algorithm", margin_titles=True)

def lineplot(x, y, twinx=False, **kwargs):
    if not twinx:
        ax = plt.gca()
    else:
        ax = plt.twinx()
    seaborn.lineplot(x=x, y=y, **kwargs, ax=ax)

g.map(lineplot, "N", "runtime")
g.add_legend()
g.set(xscale="log", yscale="log")
tikzplotlib.save("./output/runtime_groupplot.tikz")

g = seaborn.FacetGrid(data=df, col="T", row="sigma", hue="algorithm", margin_titles=True)
g.map(lineplot, "N", "variance", linestyle="--")
g.add_legend()
g.set(xscale="log", yscale="log")

tikzplotlib.save("./output/variance_groupplot.tikz")

