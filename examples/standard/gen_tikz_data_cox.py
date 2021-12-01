import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

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

runtime_df = pd.concat([stationary_runtime_df, ffbs_runtime_df], axis=1)
runtime_mean = runtime_df.groupby(level=[0, 1]).mean()

fig, axes = plt.subplots(ncols=2, figsize=(15, 6), sharex=True, sharey=True)
sns.lineplot(data=scores_std.reset_index(), x="N", y="Parallel", hue="T", ax=axes[0])
sns.lineplot(data=scores_std.reset_index(), x="N", y="Sequential", hue="T", ax=axes[1])
axes[0].set_ylabel("Standard deviation")
axes[1].set_ylabel("Standard deviation")
axes[0].set_title("Parallel")
axes[1].set_title("Sequential")
axes[0].set_xscale("log")
axes[1].set_xscale("log")


plt.show()

fig, axes = plt.subplots(ncols=2, figsize=(15, 6), sharex=True, sharey=True)
sns.lineplot(data=scores_std.reset_index(), x="T", y="Parallel", hue="N", ax=axes[0])
sns.lineplot(data=scores_std.reset_index(), x="T", y="Sequential", hue="N", ax=axes[1])
axes[0].set_ylabel("Standard deviation")
axes[1].set_ylabel("Standard deviation")
axes[0].set_title("Parallel")
axes[1].set_title("Sequential")
axes[0].set_xscale("log")
axes[1].set_xscale("log")

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


# stationary_stats = stats.describe(stationary_scores, axis=-1)
# print(stationary_stats.mean)
# print(stationary_stats.variance ** 0.5)
#
# print("FFBS stats")
# ffbs_stats = stats.describe(ffbs_scores, axis=-1)
#
# print(ffbs_stats.mean)
# print(ffbs_stats.variance ** 0.5)



#
# # parallel_stddev = parallel_res["ps_ell_stds"]
#
# print(parallel_index["N"])
# print(parallel_index["N"].shape)
# print(parallel_runtime.shape)
#
# T_index_shape = parallel_index["T"][..., -1].shape
# plt.scatter(np.log2(parallel_index["T"][..., -1]) + 0.1 * np.random.randn(*T_index_shape),
#             parallel_runtime / ffbs_runtime, c=parallel_index["N"][..., -1], alpha=0.5)
# plt.yscale("log")
# plt.xticks()
# plt.show()
#
# fig, ax = plt.subplots()
#
# scatter = ax.scatter(np.log2(parallel_index["T"][..., -1]) + 0.1 * np.random.randn(*T_index_shape),
#                      ffbs_stddev / parallel_stddev, alpha=0.5, c=parallel_index["dim_y"][..., -1])
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="dim_y")
# ax.add_artist(legend1)
# plt.legend()
# plt.show()
