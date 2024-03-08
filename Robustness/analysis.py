import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load the results
with open('/Users/modelt/Documents/Research/Extended Particle FIlter/Experiment 4/results.npy', 'rb') as f:
    result = pickle.load(f)

# Handle result
sample_mean_N = np.mean([result[0] for result in result], axis=0)
sample_mean_T = np.mean([result[1] for result in result], axis=0)
sample_std_N = np.mean([result[2] for result in result], axis=0)
sample_std_T = np.mean([result[3] for result in result], axis=0)
acceptance_mean_N = np.mean([result[4] for result in result], axis=0)
acceptance_mean_T = np.mean([result[5] for result in result], axis=0)
mean_deviation_T = np.mean([result[6] for result in result], axis=0)
mean_deviation_N = np.mean([result[7] for result in result], axis=0)

# Create Table 1
variable_names = ['$sigma_nu$ = 1', '$beta$ =1', '$lambda_0$ =1', '$sigma_omega$ =1', '$sigma_xi$ =1', '$rho$ = 0.7 ', 'm=1',   'Acceptance Rate']

# Create a DataFrame with the results for result07
df = pd.DataFrame({
    'N-distribution': [f"{sample_mean_N[i]:.3f} ({sample_std_N[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_N:.3f}"],
    'T-distribution': [f"{sample_mean_T[i]:.3f} ({sample_std_T[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_T:.3f}"],
}, index = variable_names)

# Convert df07 to LaTeX
latex = df.to_latex(caption="Metropolis-Hastings Sampler Results")
print("\n") 
print(latex)

# Create Figure
plt.figure()

# Plot the deviations
plt.plot(mean_deviation_N, label='N-distribution')
plt.plot(mean_deviation_T, label='T-distribution')
#plt.title(f'Correlation Coefficient: 0.7')
plt.legend(loc='upper left')

# Show the figure
plt.tight_layout()
plt.show()