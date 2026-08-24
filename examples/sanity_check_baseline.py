from calibris.data import generate_dataset
from calibris.baseline import naive_gp_baseline
import numpy as np

d = generate_dataset(seed=0)
mean, std, cov = naive_gp_baseline(d["x"], d["observed"], sigma=d["sigma"])

within_1sigma = np.abs(d["true_signal"] - mean) < std
print(within_1sigma.mean())
