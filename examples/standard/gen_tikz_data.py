import numpy as np
from matplotlib import pyplot as plt

ffbs_res = np.load("./output/result_degeneracy-True-gpu.npz")
parallel_res = np.load("./output/result_degeneracy-False-gpu.npz")

# I created the indices poorly...
ffbs_index = ffbs_res["indices"]
parallel_index = parallel_res["indices"]

ffbs_runtime = ffbs_res["runtime_medians"]
parallel_runtime = parallel_res["runtime_medians"]

ffbs_stddev = ffbs_res["ps_ell_stds"]
parallel_stddev = parallel_res["ps_ell_stds"]

print(parallel_index["N"][..., -1])
print(parallel_index["N"].shape)
print(parallel_runtime.shape)

T_index_shape = parallel_index["T"][..., -1].shape
plt.scatter(np.log2(parallel_index["T"][..., -1]) + 0.1 * np.random.randn(*T_index_shape),
            parallel_runtime / ffbs_runtime, c=parallel_index["N"][..., -1], alpha=0.5)
plt.yscale("log")
plt.xticks()
plt.show()

fig, ax = plt.subplots()

scatter = ax.scatter(np.log2(parallel_index["T"][..., -1]) + 0.1 * np.random.randn(*T_index_shape),
                     ffbs_stddev / parallel_stddev, alpha=0.5, c=parallel_index["dim_y"][..., -1])
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="dim_y")
ax.add_artist(legend1)
plt.legend()
plt.show()
