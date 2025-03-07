import torch
from operators import Fourier, im2vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
This script computes the hit rates for images on the test dataset.
"""

# read in data
path = 'results/test_results_itnet_noisy/'
path_rest = 'results/test_results_itnet_noisy_rest/'
rec_test_all = torch.load(os.path.join(path, "rec_test_all.pt"))
mask_test_all = torch.load(os.path.join(path,"mask_test_all.pt"))
kspace_test_all = torch.load(os.path.join(path,"kspace_test_all.pt"))
tar_test_all = torch.load(os.path.join(path,"tar_test_all.pt"))
epsilon_test_all = torch.load(os.path.join(path,"epsilon_test_all.pt"))
hat_sigma_R2_j = torch.load(os.path.join(path_rest,"variance_restterm_real_matrix.pt"))
hat_S_j = torch.load(os.path.join(path_rest,"restterm_expectation_real_matrix.pt"))
variance_real = torch.load(os.path.join(path_rest,"variance_restterm_real.pt"))

"""Choose gamma calculated in finding_gamma.py, choose alpha and sigma."""
gamma = 0.2698
sigma = 60
alpha = 0.1
alpha_real = 1.64  #is 1.96 for alpha=0.05, is 1.64 for alpha=0.1
l = 1372
n = torch.sum(mask_test_all[0][0]).cpu()
a = np.random.normal(0, 1, 100000)

"""Calculate new confidence radius"""
term1 = (np.sqrt(((sigma**2)/2)/n)) *np.quantile(a,1-(gamma*alpha/2))
term2 = np.sqrt((l**2 - 1) / (l**2 * (1 - gamma) * alpha - l)) * np.sqrt(hat_sigma_R2_j)
delta_matrix = term1 + term2 + hat_S_j

restterm_infty = []
gaussterm_infty = []
restterm_all = []
restterm_l2 = []
gaussterm_l2 = []
error_infty =[]
deb_error_infty = []
error_l2 =[]
deb_error_l2 = []
hitrates_support_matrix = []
hitrates_all_matrix = []
hitrates_support_old = []
hitrates_all_old = []
hitrates_all_gauss = []
hitrates_support_gauss = []
num_exp = 100
print(rec_test_all.shape)
"""Loop iterates over experiments."""
for i in range(0,0+num_exp):
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
    print("error model", np.linalg.norm(est_gt.cpu()))
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
    restterm_l2.append(np.linalg.norm(restterm.cpu()))
    print("Restterm",np.linalg.norm(restterm.cpu()))
    restterm_infty.append(np.linalg.norm(restterm.flatten().cpu(), ord=np.inf))

    """Calculate Gaussterm W"""
    gaussterm = (320**2/n)*Op.adj(epsilon_masked)
    gaussterm_l2.append(np.linalg.norm(gaussterm.cpu()))
    print("Gaussterm",np.linalg.norm(gaussterm.cpu()))
    gaussterm_infty.append(np.linalg.norm(gaussterm.flatten().cpu(), ord=np.inf))


    """Calculate confidence intervals/radii according to the 3 methods."""
    delta_old = (np.sqrt(((sigma**2)/2)/n)) * alpha_real #(np.sqrt(sigma**2 /n)) * np.sqrt(np.log(1 / alpha))
    delta_matrix = delta_matrix.flatten().cpu()
    delta_gauss = (np.sqrt(((sigma**2)/2 + variance_real)/n)) * alpha_real

    gt = im2vec(gt)
    beta_u = im2vec(beta_u)
    diff = (gt-beta_u).cpu()

    # take nonzero elements or all elements and take difference between groundtruth  and unbiased beta
    sorted_array = torch.sort(gt[0, 0, :])[1].cpu()
    sorted_array = sorted_array[92160:]
    print(sorted_array.shape)
    print(gt.shape)
    gt_nonzero = gt[0, :, sorted_array]
    print(gt_nonzero)
    print(gt_nonzero.shape)
    beta_u_nonzero = beta_u[0, :, sorted_array]
    difference_nonzero = (gt_nonzero-beta_u_nonzero).cpu()
    print(difference_nonzero.shape)
    diff_nonzero = difference_nonzero #[:,:, 0].cpu()
    delta_matrix_nonzero = delta_matrix[sorted_array]
    print('index nonzero shape',sorted_array.shape)

    """Calculate new hit rates"""
    hitrate_all = 0
    hitrate_nonzero = 0
    for j in range(gt.size(2)):
        if bool(np.abs(diff[0, 0, j]) < delta_matrix[j]):
            hitrate_all += 1
    hitrates_all_matrix.append((hitrate_all/gt.size(2)))
    print('hitrates', (hitrate_all/gt.size(2)))
    print('shape of ground truth',gt_nonzero.shape)
    for k in range(gt_nonzero.size(1)):
        if bool(np.abs(diff_nonzero[0,k]) < delta_matrix_nonzero[k]):
            hitrate_nonzero += 1
    hitrates_support_matrix.append(hitrate_nonzero/gt_nonzero.size(1))
    print('hitrates', (hitrate_nonzero / gt_nonzero.size(1)))

    """Calculate gaussian adjusted hit rates """
    hitrate_all = 0
    hitrate_nonzero = 0
    for j in range(gt.size(2)):
        if bool(np.abs(diff[0, 0, j]) < delta_gauss):
            hitrate_all += 1
    hitrates_all_gauss.append((hitrate_all / gt.size(2)))
    print('hitrates', (hitrate_all / gt.size(2)))
    for k in range(gt_nonzero.size(1)):
        if bool(np.abs(diff_nonzero[0,k]) < delta_gauss):
            hitrate_nonzero += 1
    hitrates_support_gauss.append(hitrate_nonzero / gt_nonzero.size(1))
    print('hitrates', (hitrate_nonzero / gt_nonzero.size(1)))

    """Calculate asymptotic hit rates"""
    hitrate_all = 0
    hitrate_nonzero = 0
    for j in range(gt.size(2)):
        if bool(np.abs(diff[0, 0, j]) < delta_old):
            hitrate_all += 1
    hitrates_all_old.append((hitrate_all / gt.size(2)))
    print('hitrates', (hitrate_all / gt.size(2)))
    for k in range(gt_nonzero.size(1)):
        if bool(np.abs(diff_nonzero[0,k]) < delta_old):
            hitrate_nonzero += 1
    hitrates_support_old.append(hitrate_nonzero / gt_nonzero.size(1))
    print('hitrates', (hitrate_nonzero / gt_nonzero.size(1)))

"""Print the Average errors, normed R and Ws"""
print('########## Average results #######')
print('Number of Experiments:', num_exp )
print('L2-Norm Differenz mean:', sum(error_l2)/len(error_l2))
print('Loo-Norm Differenz mean:', sum(error_infty)/len(error_infty))
print('L2-Norm u-Differenz mean:',  sum(deb_error_l2)/(len(deb_error_l2)))
print('Loo-Norm u-Differenz mean:', sum(deb_error_infty)/len(deb_error_infty))
dataframe = pd.DataFrame({'hitrates all new': hitrates_all_matrix, 'hitrates support new': hitrates_support_matrix, 'hitrates all gauss': hitrates_all_gauss, 'hitrates support gauss': hitrates_support_gauss, 'hitrates all old': hitrates_all_old, 'hitrates support old': hitrates_support_old})
print()
print('L2-Norm Remainder term with M mean:', sum(restterm_l2)/len(restterm_l2))
print('Loo-Norm Remainder term with M mean:', sum(restterm_infty)/len(restterm_infty))
print('L2-Norm Gauss term with M mean:', sum(gaussterm_l2)/(len(gaussterm_l2)))
print('Loo-Norm Gauss term with M mean:', sum(gaussterm_infty)/len(gaussterm_infty))
print()
print('Hitrates, noise level and SSIM')
print('hitrate mean matrix:', sum(hitrates_all_matrix)/(len(hitrates_all_matrix)))
print('hitrate support mean matrix:', sum(hitrates_support_matrix)/(len(hitrates_support_matrix)))
print('hitrate mean gauss:', sum(hitrates_all_gauss)/(len(hitrates_all_gauss)))
print('hitrate support mean gauss:', sum(hitrates_support_gauss)/(len(hitrates_support_gauss)))
print('hitrate mean old:', sum(hitrates_all_old)/(len(hitrates_all_old)))
print('hitrate support mean old:', sum(hitrates_support_old)/(len(hitrates_support_old)))
print(dataframe)
dataframe.to_csv(os.path.join(path,'dataframe_CI_real_{}_test.dat'.format(alpha)))

