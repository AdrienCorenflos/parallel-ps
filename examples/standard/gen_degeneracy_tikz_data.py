import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ffbs_res = np.load("./output/result_degeneracy-True-gpu.npz")
parallel_res = np.load("./output/result_degeneracy-False-gpu.npz")

ffbs_index = ffbs_res["indices"]
parallel_index = parallel_res["indices"]

ffbs_ells = ffbs_res["ells"]
parallel_ells = parallel_res["ells"]

rs = np.tile(ffbs_res["rs"][:], parallel_index["T"].shape + (1,))
kf_ells = ffbs_res["kf_ells"]

ffbs_ells_diff = ffbs_ells - kf_ells[..., None]
parallel_ells_diff = parallel_ells - kf_ells[..., None]

ffbs_ells_std = ffbs_ells_diff.std(axis=-1)
parallel_ells_std = parallel_ells_diff.std(axis=-1)

T = np.tile(parallel_index["T"], (1, 1, rs.shape[-1]))
N = np.tile(parallel_index["N"], (1, 1, rs.shape[-1]))

T, N, ffbs_ells_std, parallel_ells_std, kf_ells, rs = list(map(np.ravel, [T,
                                                                          N,
                                                                          ffbs_ells_std,
                                                                          parallel_ells_std,
                                                                          kf_ells,
                                                                          rs]))

df = pd.DataFrame(columns=["T", "N", "rs" "Sequential", "Parallel", "kf"])
df["T"] = T
df["N"] = N

df["Sequential"] = ffbs_ells_std
df["Parallel"] = parallel_ells_std
df["rs"] = rs

colordict = {25: 1, 50: 2, 100: 3, 250: 4}

sns.relplot(x=df["T"], y=df["Parallel"], col=df["rs"],
            hue=df["N"], kind="scatter", legend=False,)

plt.savefig("here.png")

sns.relplot(x=df["rs"], y=df["Sequential"], col=df["rs"],
            legend=False, hue=df["N"], kind="scatter")

plt.savefig("here.png")

#
# ax.set_xscale("log", base=2)
# ax.set_yscale("log", base=10)
# ax.set_ylabel("Runtime (s)")

# tikzplotlib.save("./output/degeneracy.tex")
