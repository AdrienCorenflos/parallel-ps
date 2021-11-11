import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

seq_res = np.load(f"./output/theta-logistic-experiment-False-True.npz")
par_res = np.load(f"./output/theta-logistic-experiment-True-False.npz")

print(f"Sequential run took: {seq_res['runtime']:2f}s")
print(f"Parallel run took: {par_res['runtime']:2f}s")

# dump csv for update rate
seq_update_rate = seq_res["rejuvenated_logs"]
par_update_rate = par_res["rejuvenated_logs"]

update_rate = pd.DataFrame(data=np.stack([seq_update_rate, par_update_rate], -1),
                           columns=["Sequential", "Parallel"])
update_rate.index.rename('time', inplace=True)
update_rate.to_csv("./output/update_rate.csv")

# dump csvs for ACF
sample_names = ["x_prec", "y_prec", "tau0", "tau1", "tau2"]

for name in sample_names:
    seq_sample_data = seq_res[name + "_samples"]
    par_sample_data = par_res[name + "_samples"]
    seq_acf_data = acf(seq_sample_data, nlags=100)
    par_acf_data = acf(par_sample_data, nlags=100)

    acf_df = pd.DataFrame(data=np.stack([seq_acf_data, par_acf_data], -1),
                          columns=["Sequential", "Parallel"])
    acf_df.index.rename("lag", inplace=True)
    acf_df.to_csv(f"./output/acf-{name}.csv")

# For x_0
seq_sample_data = seq_res["traj_samples"][:, 0, 0]
par_sample_data = par_res["traj_samples"][:, 0, 0]
seq_acf_data = acf(seq_sample_data, nlags=100)
par_acf_data = acf(par_sample_data, nlags=100)

acf_df = pd.DataFrame(data=np.stack([seq_acf_data, par_acf_data], -1),
                      columns=["Sequential", "Parallel"])
acf_df.index.rename("lag", inplace=True)
acf_df.to_csv(f"./output/acf-x0.csv")

