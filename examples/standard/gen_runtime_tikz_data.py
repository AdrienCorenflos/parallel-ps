import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib as tikzplotlib

ffbs_res = np.load("./output/cox-True-False-gpu.npz")
parallel_res = np.load("./output/cox-False-False-gpu.npz")

ffbs_index = ffbs_res["indices"]
parallel_index = parallel_res["indices"]

ffbs_runtime_mean = ffbs_res["runtime_means"]
parallel_runtime_mean = parallel_res["runtime_means"]

flat_T_index = np.ravel(parallel_index["T"])
flat_N_index = np.ravel(parallel_index["N"])
index = pd.MultiIndex.from_arrays([flat_T_index, flat_N_index], names=["T", "N"])

ffbs_runtime_mean = np.reshape(ffbs_runtime_mean, (len(flat_T_index), -1))
parallel_runtime_mean = np.reshape(parallel_runtime_mean, (len(flat_T_index), -1))

parallel_runtime_df = pd.DataFrame(index=index, data=parallel_runtime_mean)
sequential_runtime_df = pd.DataFrame(index=index, data=ffbs_runtime_mean)

parallel_runtime_df = parallel_runtime_df.stack()
parallel_runtime_df.index = parallel_runtime_df.index.set_names("batch", level=-1)
parallel_runtime_df.name = "Parallel"
parallel_runtime_df = parallel_runtime_df.reset_index()

sequential_runtime_df = sequential_runtime_df.stack()
sequential_runtime_df.index = sequential_runtime_df.index.set_names("batch", level=-1)
sequential_runtime_df.name = "Sequential"
sequential_runtime_df = sequential_runtime_df.reset_index()

fig, ax = plt.subplots(figsize=(12, 6))

colordict = {25: 1, 50: 2, 100: 3, 250: 4, 500: 5, 1000: 6}

sns.lineplot(x=parallel_runtime_df["T"], y=parallel_runtime_df["Parallel"], hue=parallel_runtime_df["N"].map(colordict),
             marker="o", markersize=7, ax=ax, ci=95)

sns.lineplot(x=sequential_runtime_df["T"], y=sequential_runtime_df["Sequential"],
             hue=sequential_runtime_df["N"].map(colordict), marker="s",
             markersize=7, legend=False, ax=ax)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_ylabel("Runtime (s)")
# plt.show()
tikzplotlib.save("./output/runtime.tex")
