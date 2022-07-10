import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib as tikzplotlib

ffbs_res = np.load("./output/cox-True-False-gpu.npz")
parallel_res = np.load("./output/cox-False-False-gpu.npz")

ffbs_index = ffbs_res["indices"]
parallel_index = parallel_res["indices"]

ffbs_ells = ffbs_res["batch_scores"]
parallel_ells = parallel_res["batch_scores"]

ffbs_ells_std = ffbs_ells.std(axis=-2)
parallel_ells_std = parallel_ells.std(axis=-2)

ratio = parallel_ells_std / ffbs_ells_std

flat_T_index = np.ravel(parallel_index["T"])
flat_N_index = np.ravel(parallel_index["N"])

degeneracy_data = list(map(np.ravel, [parallel_index["T"],
                                    parallel_index["N"],
                                    ]))
flat_ratio = ratio.reshape(degeneracy_data[0].shape + (-1,))

degeneracy_df_index = pd.MultiIndex.from_arrays([flat_T_index, flat_N_index], names=["T", "N"])
degeneracy_df = pd.DataFrame(index = degeneracy_df_index, data=flat_ratio)
degeneracy_df = degeneracy_df.stack()
degeneracy_df.name = "ratio"
degeneracy_df.index = degeneracy_df.index.set_names('batch', level=2)

degeneracy_df = degeneracy_df.reset_index()


fig, ax = plt.subplots(figsize=(12, 6))

colordict = {32: 1, 64: 2, 128: 3, 256: 4, 512: 5}

sns.lineplot(x=degeneracy_df["N"], y=degeneracy_df["ratio"], hue=degeneracy_df["T"].map(colordict), marker="o", markersize=7,
             ax=ax, ci=90)

ax.set_xscale("log", base=2)
ax.set_ylabel("ratio")
plt.show()


tikzplotlib.save("./output/degeneracy.tex")
