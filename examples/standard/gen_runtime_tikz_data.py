import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib as tikzplotlib

ffbs_res = np.load("./output/result_runtime-True-gpu.npz")
parallel_res = np.load("./output/result_runtime-False-gpu.npz")

ffbs_index = ffbs_res["indices"]
parallel_index = parallel_res["indices"]

ffbs_runtime = ffbs_res["runtime_means"]
parallel_runtime = parallel_res["runtime_means"]

run_time_data = list(map(np.ravel, [parallel_index["T"],
                                    parallel_index["N"],
                                    ffbs_runtime,
                                    parallel_runtime]))

runtime_df = pd.DataFrame(columns=["T", "N", "Sequential", "Parallel"])
runtime_df["T"] = run_time_data[0]
runtime_df["N"] = run_time_data[1]
runtime_df["Sequential"] = run_time_data[2]
runtime_df["Parallel"] = run_time_data[3]

fig, ax = plt.subplots(figsize=(12, 6))

colordict = {25: 1, 50: 2, 100: 3, 250: 4}

# ax.scatter(x=runtime_df["T"], y=runtime_df["Sequential"], marker="x", c=runtime_df["N"].map(colordict), cmap="viridis")
# scatter = ax.plot(runtime_df["T"], runtime_df["Parallel"], marker="o", c=runtime_df["N"].map(colordict),
#                      cmap="viridis")
sns.lineplot(x=runtime_df["T"], y=runtime_df["Parallel"], hue=runtime_df["N"].map(colordict), marker="o", markersize=7,
             ax=ax)
sns.lineplot(x=runtime_df["T"], y=runtime_df["Sequential"], hue=runtime_df["N"].map(colordict), marker="s",
             markersize=7, legend=False, ax=ax)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_ylabel("Runtime (s)")

tikzplotlib.save("./output/runtime.tex")
