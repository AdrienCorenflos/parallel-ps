import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ffbs_res = np.load("./output/cox-True-False-gpu.npz")
parallel_stationary_res = np.load("./output/cox-False-False-gpu.npz")

# I created the indices poorly...
index = parallel_stationary_res["indices"]

stationary_runtime = parallel_stationary_res["runtime_means"]
ffbs_runtime = ffbs_res["runtime_means"]
#
# print(index["T"])
# print(index["N"])
#
# print("Stationary Runtime")
#
# print(stationary_runtime)
#
# print("FFBS Runtime")
#
# print(ffbs_runtime)

stationary_scores = parallel_stationary_res["batch_scores"]
ffbs_scores = ffbs_res["batch_scores"]

flat_T_index = index["T"].reshape(-1)
flat_N_index = index["N"].reshape(-1)


multi_index = pd.MultiIndex.from_arrays([flat_T_index, flat_N_index], names=["T", "N"])
stationary_scores_df = pd.DataFrame(data=stationary_scores.reshape(flat_T_index.shape[0], -1), index=multi_index).stack()
ffbs_scores_df = pd.DataFrame(data=ffbs_scores.reshape(flat_T_index.shape[0], -1), index=multi_index).stack()
stationary_scores_df.name = "Parallel"
ffbs_scores_df.name = "Sequential"

scores_df = pd.concat([stationary_scores_df, ffbs_scores_df], 1)
scores_df.index.name = "idx"

stationary_runtime_df = pd.DataFrame(data=stationary_runtime.reshape(flat_T_index.shape[0], -1), index=multi_index)
ffbs_runtime_df = pd.DataFrame(data=ffbs_runtime.reshape(flat_T_index.shape[0], -1), index=multi_index)
stationary_runtime_df.columns.name = "idx"
ffbs_runtime_df.columns.name = "idx"

print(np.unique(index["T"]))
print(np.unique(index["N"]))

stationary_runtime_df = stationary_runtime_df.stack()
ffbs_runtime_df = ffbs_runtime_df.stack()

stationary_runtime_df.name = "Parallel"
ffbs_runtime_df.name = "Sequential"

scores_df = pd.concat([stationary_scores_df, ffbs_scores_df], axis=1)
scores_std = scores_df.groupby(level=[0, 1]).std()

std_dev_ratio = scores_std["Parallel"] / scores_std["Sequential"]
std_dev_ratio = std_dev_ratio.to_frame(name="ratio")

runtime_df = pd.concat([stationary_runtime_df, ffbs_runtime_df], axis=1)
runtime_mean = runtime_df.groupby(level=[0, 1]).mean()

df = pd.concat([std_dev_ratio, runtime_mean], axis=1).reset_index()
df = df.rename(columns={"Parallel": "dSMC", "Sequential": "FFBS"})
df.to_csv("./output/cox-results.csv", index=False)


runtime_mean = runtime_df.groupby(level=[0, 1]).mean()
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(data=std_dev_ratio.reset_index(), y=0, x="N", hue="T", ax=ax)
ax.set_ylabel("$\sigma_{dSMC} / \sigma_{SMC}$")
ax.set_title("Parallel")

plt.show()

fig, axes = plt.subplots(ncols=2, figsize=(15, 6), sharex=True, sharey=True)
sns.lineplot(data=runtime_df.reset_index(), x="T", y="Parallel", hue="N", ax=axes[0])
sns.lineplot(data=runtime_df.reset_index(), x="T", y="Sequential", hue="N", ax=axes[1])
axes[0].set_ylabel("Mean runtime")
axes[1].set_ylabel("Mean runtime")
axes[0].set_title("Parallel")
axes[1].set_title("Sequential")
axes[0].set_xscale("log", base=2)
axes[1].set_xscale("log", base=2)
axes[0].set_yscale("log")
axes[1].set_yscale("log")

plt.show()

