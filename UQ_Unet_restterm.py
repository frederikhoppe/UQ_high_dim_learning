import torch
from operators import Fourier, im2vec
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

"""
This script estimates the variance and the standard deviation of the restterm on the estimation dataset.
"""
# read in data
path = 'results/test_results_itnet_60per_noisy_rest/'

rec_test_all = torch.load(os.path.join(path, "rec_test_all.pt"))
mask_test_all = torch.load(os.path.join(path,"mask_test_all.pt"))
kspace_test_all = torch.load(os.path.join(path,"kspace_test_all.pt"))
tar_test_all = torch.load(os.path.join(path,"tar_test_all.pt"))
epsilon_test_all = torch.load(os.path.join(path,"epsilon_test_all.pt"))
num_exp = 1372

restterm_infty = []
gaussterm_infty = []
restterm_all = []
restterm_l2 = []
gaussterm_l2 = []
error_infty =[]
deb_error_infty = []
error_l2 =[]
deb_error_l2 = []
variance_restterm_all = []
hitrates_support = []
hitrates_all = []
restterm_real = []

"""Loop iterates over experiments."""
for i in range(num_exp):
    print(i)
    beta = rec_test_all[i:i+1, :, :, :]
    mask = mask_test_all[i:i+1, :, :, :]
    y = kspace_test_all[i:i+1, :, :, :]
    gt = tar_test_all[i:i+1, :, :, :]
    epsilon = epsilon_test_all[i:i+1,:,:,:]
    mask_im = mask[0, 1, :, :]  # assume same mask per batch and channel
    y_masked = im2vec(y)[:, :, im2vec(mask_im) > 0]
    epsilon_masked = im2vec(epsilon)[:, :, im2vec(mask_im) > 0]
    est_gt = beta-gt

    # Print model error
    print("error", np.linalg.norm(est_gt.cpu()))
    error_l2.append(np.linalg.norm(est_gt.cpu()))
    error_infty.append(np.linalg.norm(est_gt.flatten().cpu(), ord=np.inf))


    """define operator, n=number of measurements and unbiased beta"""
    Op = Fourier(mask)
    n = torch.sum(mask[0][0]).cpu()
    beta_u = beta + (320**2/n)*Op.adj(y_masked-Op.dot(beta))
    error_deb = beta_u-gt
    print("error deb", np.linalg.norm(error_deb.cpu()))
    deb_error_l2.append(np.linalg.norm(error_deb.cpu()))
    deb_error_infty.append(np.linalg.norm(error_deb.flatten().cpu(), ord=np.inf))

    """Calculate restterm R"""
    step = Op.dot(est_gt)
    restterm = -(320**2/n)*Op.adj(step)+est_gt
    restterm_all.append(np.sqrt((restterm[0][0]**2+restterm[0][1]**2).cpu()))
    restterm_real.append(restterm[0][0].cpu())
    restterm_l2.append(np.linalg.norm(restterm.cpu()))
    print("Restterm",np.linalg.norm(restterm.cpu()))
    restterm_infty.append(np.linalg.norm(restterm.flatten().cpu(), ord=np.inf))

    """Calculate Gaussterm W"""
    gaussterm = (320**2/n)*Op.adj(epsilon_masked)
    gaussterm_l2.append(np.linalg.norm(gaussterm.cpu()))
    print("Gaussterm",np.linalg.norm(gaussterm.cpu()))
    gaussterm_infty.append(np.linalg.norm(gaussterm.flatten().cpu(), ord=np.inf))


"""Calculate Varaince matrix over all images in the estimation dataset,
 also calculate variance on the real part of the images."""
restterm_expectation_matrix = (1/num_exp)*sum(restterm_all)
restterm_expectation_real = (1/num_exp)*sum(restterm_real)
variance_restterm_matrix = 0
variance_restterm_real = 0
for j in range(num_exp):
    variance_restterm_matrix += (restterm_all[j]-restterm_expectation_matrix)**2
    variance_restterm_real += (restterm_real[j])**2
variance_restterm_matrix = (1/(num_exp-1))*variance_restterm_matrix
variance_restterm_real = (1/(num_exp-1))*variance_restterm_real
torch.save(variance_restterm_matrix,os.path.join(path,"variance_restterm_matrix.pt"))
torch.save(restterm_expectation_matrix, os.path.join(path,"restterm_expectation_matrix.pt"))
variance_real = n*(1/(320**2))*sum(variance_restterm_real.flatten())
torch.save(variance_real, os.path.join(path,"variance_restterm_real.pt"))


"""Plot histogram for real part of the restterm, to check if it is gaussian distributed with variance=variance_real"""
plt.figure(1)
plt.hist(restterm_real[900].flatten(), density=1, bins=1000)
plt.plot(np.arange(-0.5,0.5,0.001), norm.pdf(np.arange(-0.5,0.5,0.001),0,np.sqrt(float(variance_real)/n)))
plt.savefig(os.path.join(path,'restterm_real_historgram_900.pdf'))



"""Print the Average errors, normed R and Ws"""
print('########## Average results #######')
print('Number of Experiments:', num_exp )
print('L2-Norm Differenz mean:', sum(error_l2)/len(error_l2))
print('Loo-Norm Differenz mean:', sum(error_infty)/len(error_infty))
print('L2-Norm u-Differenz mean:',  sum(deb_error_l2)/(len(deb_error_l2)))
print('Loo-Norm u-Differenz mean:', sum(deb_error_infty)/len(deb_error_infty))
print()
print('L2-Norm Remainder term mean:', sum(restterm_l2)/len(restterm_l2))
print('Loo-Norm Remainder term mean:', sum(restterm_infty)/len(restterm_infty))
print('L2-Norm Gauss term mean:', sum(gaussterm_l2)/(len(gaussterm_l2)))
print('Loo-Norm Gauss term mean:', sum(gaussterm_infty)/len(gaussterm_infty))
print()
