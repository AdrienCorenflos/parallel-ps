import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf

lazy_res = np.load(f"./output/rare-events-False-True.npz")
seq_res = np.load(f"./output/rare-events-True-True.npz")
par_res = np.load(f"./output/rare-events-False-False.npz")

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
axes[0].plot(par_res["indices"][4]["T"].T, par_res['runtimes'][2].T)
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log", base=10)

axes[1].plot(lazy_res["indices"][4]["T"].T, lazy_res['runtimes'][2].T)
axes[1].set_xscale("log", base=2)
axes[1].set_yscale("log", base=10)

axes[2].plot(seq_res["indices"][4]["T"].T, seq_res['runtimes'][2].T)
axes[2].set_xscale("log", base=2)
axes[2].set_yscale("log", base=10)
plt.show()


fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
axes[0].plot(par_res["indices"][4]["T"].T, par_res['variances'][2].T)
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log", base=10)

axes[1].plot(lazy_res["indices"][4]["T"].T, lazy_res['variances'][2].T)
axes[1].set_xscale("log", base=2)
axes[1].set_yscale("log", base=10)

axes[2].plot(seq_res["indices"][4]["T"].T, seq_res['variances'][2].T)
axes[2].set_xscale("log", base=2)
axes[2].set_yscale("log", base=10)
plt.show()

print(f"Sequential var/time: \n {seq_res['runtimes'][0] / seq_res['variances'][0]}")
# print(f"Parallel run took: \n {par_res['runtimes']}")


print(f"Lazy var/time: \n {lazy_res['runtimes'][0] / lazy_res['variances'][0]}")
# print(f"Parallel run variance: \n {par_res['variances']}")


