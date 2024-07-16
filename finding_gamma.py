import numpy as np
from scipy.optimize import minimize
import torch
import os

path = 'results/test_results_itnet_noisy_rest/'
# Define the parameters
sigma = 60 # Noise level
m = 0.6*(320**2) # undersampling
alpha = 0.1 # confidence level
l = 1372  #numer of rest experiments

#load restterm mean and variance and average over them
hat_sigma_R2_j = torch.load(os.path.join(path,"variance_restterm_matrix.pt"))
hat_S_j = torch.load(os.path.join(path,"restterm_expectation_matrix.pt"))
hat_sigma_R2_j = np.mean(np.array(hat_sigma_R2_j ))
hat_S_j  = np.mean(np.array(hat_S_j ))


# Define the function f(gamma) to be minimized
# minimize this function to find best gamma
def f(gamma):
    term1 = (sigma /np.sqrt(m)) * np.sqrt(np.log(1 / (gamma * alpha)))
    term2 = np.sqrt((l**2 - 1) / (l**2 * (1 - gamma) * alpha - l)) * np.sqrt(hat_sigma_R2_j)
    return term1 + term2 + hat_S_j

# Initial guess for gamma
gamma0 = 0.5  # Example initial guess, adjust as necessary

# Constraints: gamma should be between 0 and 1
bounds = [(1e-10, 1 - (1/(l*alpha)))]  # Avoid zero and one to prevent division by zero or log(0)

# Use minimize to find the gamma that minimizes f(gamma)
result = minimize(f, gamma0, bounds=bounds, method='L-BFGS-B')

# Display the result
if result.success:
    print(f'The value of gamma that minimizes the function is: {result.x[0]:.4f}')
    print(f'The minimum value of the function is: {result.fun:.4f}')
else:
    print('Minimization did not converge.')
