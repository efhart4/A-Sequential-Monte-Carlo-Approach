import numpy as np
import pandas as pd

T = 2 # number of time periods
N = 5 # number of particles

sample_paths_W = np.zeros((T, N))


TVP = np.array([1, 2])
TVP_series = pd.Series(TVP)


squared_error_W = (sample_paths_W - TVP_series.to_numpy().reshape(-1, 1)) ** 2
print(squared_error_W)
"""squared_error_W2 = (sample_paths_W - TVP_series.reshape(-1, 1)) ** 2
print(squared_error_W2)"""


# print the output and its name
print('squared_error_W')
print(squared_error_W)

MSE_W = np.mean(squared_error_W, axis=1)
# print the output and its name
print('MSE_W')
print(MSE_W)

mean_sample_path_iteration_W = np.mean(sample_paths_W, axis=1)
# print the output and its name
print('mean_sample_path_iteration_W')
print(mean_sample_path_iteration_W)
