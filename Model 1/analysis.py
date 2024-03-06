from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Load the results
with open('/experiment_I_data.npy', 'rb') as f:
    result07 = pickle.load(f)


with open('/experiment_II_data.npy', 'rb') as i:
    result0 = pickle.load(i)



# Handle result0
sample_mean_EPF_0 = np.mean([result0[0] for result0 in result0], axis=0)
sample_mean_PF_0 = np.mean([result0[1] for result0 in result0], axis=0)
sample_std_EPF_0 = np.mean([result0[2] for result0 in result0], axis=0)
sample_std_PF_0 = np.mean([result0[3] for result0 in result0], axis=0)
acceptance_mean_EPF_0 = np.mean([result0[4] for result0 in result0], axis=0)
acceptance_mean_PF_0 = np.mean([result0[5] for result0 in result0], axis=0)
mean_deviation_PF_0 = np.mean([result0[6] for result0 in result0], axis=0)
mean_deviation_EPF_0 = np.mean([result0[7] for result0 in result0], axis=0)



# Handle result07
sample_mean_EPF_07 = np.mean([result07[0] for result07 in result07], axis=0)
sample_mean_PF_07 = np.mean([result07[1] for result07 in result07], axis=0)
sample_std_EPF_07 = np.mean([result07[2] for result07 in result07], axis=0)
sample_std_PF_07 = np.mean([result07[3] for result07 in result07], axis=0)
acceptance_mean_EPF_07 = np.mean([result07[4] for result07 in result07], axis=0)
acceptance_mean_PF_07 = np.mean([result07[5] for result07 in result07], axis=0)
mean_deviation_PF_07 = np.mean([result07[6] for result07 in result07], axis=0)
mean_deviation_EPF_07 = np.mean([result07[7] for result07 in result07], axis=0)

# Create Table 1
variable_names = ['$sigma_nu$ = 1', '$beta$ =1', '$lambda_0$ =4', '$sigma_omega$ =0.1', '$sigma_xi$ =1', '$rho$ ', 'm=1',   'Acceptance Rate']

# Create a DataFrame with the results for result0
df0 = pd.DataFrame({
    'Extended Particle Filter': [f"{sample_mean_EPF_0[i]:.3f} ({sample_std_EPF_0[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_EPF_0:.3f}"],
    'Particle Filter': [f"{sample_mean_PF_0[i]:.3f} ({sample_std_PF_0[i]:.3f})" if i < len(sample_mean_PF_0) else '-' for i in range(7)] + [f"{acceptance_mean_PF_0:.3f}"]
}, index = variable_names)

# Create a DataFrame with the results for result07
df07 = pd.DataFrame({
    'Extended Particle Filter': [f"{sample_mean_EPF_07[i]:.3f} ({sample_std_EPF_07[i]:.3f})" for i in range(7)] + [f"{acceptance_mean_EPF_07:.3f}"],
    'Particle Filter': [f"{sample_mean_PF_07[i]:.3f} ({sample_std_PF_07[i]:.3f})" if i < len(sample_mean_PF_07) else '-' for i in range(7)] + [f"{acceptance_mean_PF_07:.3f}"]
}, index = variable_names)

# Convert df0 to LaTeX
latex0 = df0.to_latex(caption="Model I: Metropolis-Hastings Sampler Results for 0")
print(latex0)

# Convert df07 to LaTeX
latex07 = df07.to_latex(caption="Model I: Metropolis-Hastings Sampler Results for 07")
print(latex07)


# Create Figure 2
plt.figure()

# Create the first subplot for correlation coefficient of 0
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(mean_deviation_PF_0, label='Particle Filter')
plt.plot(mean_deviation_EPF_0, label='Extended Particle Filter')
plt.title(f'Correlation Coefficient: 0')
plt.legend(loc='upper left')



# Create the second subplot for correlation coefficient of 0.7
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.plot(mean_deviation_PF_07, label='Particle Filter')
plt.plot(mean_deviation_EPF_07, label='Extended Particle Filter')
plt.title(f'Correlation Coefficient: 0.7')
plt.legend(loc='upper left')


# Show the figure with both subplots
plt.tight_layout()
plt.show()