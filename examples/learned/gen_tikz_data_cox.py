import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

learned_res = np.load("./output/cox-learned-gpu.npz")
learned_res = np.load("./output/cox-learned-gpu.npz")

# I created the indices poorly...
index = learned_res["indices"]

learned_runtime = learned_res["runtime_means"]

print(index["T"])
print(index["N"])

print("Runtime")

print(learned_runtime)

print("Learning time")
print(learned_res["learning_times"])


learned_scores = learned_res["batch_scores"]

print("Stats")

learned_stats = stats.describe(learned_scores, axis=-1)
print(learned_stats.mean)
print(learned_stats.variance ** 0.5)
