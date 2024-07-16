from matplotlib import pyplot as plt, patches
import numpy as np
from operators import im2vec, Fourier
import torch
import os

"""This script plots confidence intervals for one image from the test set for all 3 methods."""

#Load data
path = 'results/test_results_itnet_noisy/'
path_rest = 'results/test_results_itnet_noisy_rest/'
rec_test_all = torch.load(os.path.join(path, "rec_test_all.pt"))
mask_test_all = torch.load(os.path.join(path,"mask_test_all.pt"))
kspace_test_all = torch.load(os.path.join(path,"kspace_test_all.pt"))
tar_test_all = torch.load(os.path.join(path,"tar_test_all.pt"))
epsilon_test_all = torch.load(os.path.join(path,"epsilon_test_all.pt"))
hat_sigma_R2_j = torch.load(os.path.join(path_rest,"variance_restterm_matrix.pt"))
hat_S_j = torch.load(os.path.join(path_rest,"restterm_expectation_matrix.pt"))
variance_real = torch.load(os.path.join(path_rest,"variance_restterm_real.pt"))

#Choose image, gamma, sigma and alpha
i = 910
gamma = 0.3219
sigma = 60
alpha = 0.05
alpha_real = 1.96 #is 1.96 for alpha=0.05, is 1.64 for alpha=0.1
l = 1372

beta = rec_test_all[i:i+1, :, :, :]
mask = mask_test_all[i:i+1, :, :, :]
y = kspace_test_all[i:i+1, :, :, :]
gt = tar_test_all[i:i+1, :, :, :]
epsilon = epsilon_test_all[i:i+1,:,:,:]
mask_im = mask[0, 1, :, :]  # assume same mask per batch and channel
y_masked = im2vec(y)[:, :, im2vec(mask_im) > 0]
epsilon_masked = im2vec(epsilon)[:, :, im2vec(mask_im) > 0]
est_gt = beta-gt

# define operator, n=number of measurements and unbiased beta
Op = Fourier(mask)
n = torch.sum(mask[0][0]).cpu()
beta_u = beta + (320**2/n)*Op.adj(y_masked-Op.dot(beta))
image = np.array(gt.cpu())
estimator_u = beta_u[0][0]+1j*beta_u[0][1]
estimator_u = np.array(estimator_u.cpu())
estimator_u_abs = np.abs(estimator_u)
estimator_u_real = np.array(beta_u[0][0].cpu())

#Calculate different confidence radii and intervals
term1 = (np.sqrt(sigma**2 /n)) * np.sqrt(np.log(1 / (gamma * alpha)))
term2 = np.sqrt((l**2 - 1) / (l**2 * (1 - gamma) * alpha - l)) * np.sqrt(hat_sigma_R2_j)
delta_matrix = term1 + term2 + hat_S_j
delta_asymp = (np.sqrt(sigma**2 /n)) * np.sqrt(np.log(1 / alpha))
delta_asymp_real = (np.sqrt(((sigma**2)/2)/n)) * alpha_real  #is the real version of the asymptotic CI
image = image[0][0]
delta_gauss = (np.sqrt(((sigma**2)/2 + variance_real)/n)) * alpha_real

#Choose area in the inage for which to plot the confidence intervals.
a_1 = 240
a_2 = 165
b = 170
c = 250

#Plot and save CIs and image with rectangle
plt.figure(1)
plt.errorbar(range(50), estimator_u_abs[a_1:c, a_2:b].flatten(), yerr=delta_matrix[a_1:c, a_2:b].flatten(), capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(np.abs(image)[a_1:c, a_2:b].flatten(), 'r+')
plt.savefig(os.path.join(path, 'confidence_intervals_knee_new_60perc_005CI.png'))

plt.figure(2)
plt.errorbar(range(50), estimator_u_real[a_1:c, a_2:b].flatten(), yerr=delta_gauss, capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(np.real(image)[a_1:c, a_2:b].flatten(), 'r+')
plt.savefig(os.path.join(path, 'confidence_intervals_knee_gauss_60perc_005CI.png'))

plt.figure(3)
plt.errorbar(range(50), estimator_u_abs[a_1:c, a_2:b].flatten(), yerr=delta_asymp, capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(np.abs(image)[a_1:c, a_2:b].flatten(), 'r+')
plt.savefig(os.path.join(path, 'confidence_intervals_knee_asymp_60perc_005CI.png'))

plt.figure(4)
plt.errorbar(range(50), estimator_u_real[a_1:c, a_2:b].flatten(), yerr=delta_asymp_real, capsize=2, marker='o', drawstyle='steps', linestyle='', markerfacecolor='none')
plt.plot(np.real(image)[a_1:c, a_2:b].flatten(), 'r+')
plt.savefig(os.path.join(path, 'confidence_intervals_knee_asymp_real_60perc_005CI.png'))

fig, ax = plt.subplots(1)
ax.imshow(image, cmap="gray")
rect = patches.Rectangle((a_2,a_1), width=c-a_1, height=b-a_2, edgecolor='r', facecolor="none", linewidth=1)
ax.add_patch(rect)
plt.xticks([],visible=False)
plt.yticks([],visible=False)
plt.savefig(os.path.join(path, 'image_knee.pdf'))