# `Non-asymptotic Uncertainty Quantification in High-Dimensional Learning`

Code to paper "Non-asymptotic Uncertainty Quantification in High-Dimensional Learning".
Image reconstruction from single-coil MRI images with radial line subsampling mask using the It-Net and U-Net.
Uncertainty Quantification using debiased estimator for different methods to calculate confidence intervals.
Uncertainty Quantification for sparse regression correcting the confidence intervals provided by the debiased LASSO.



## How to run the model-based regression experiments (Section 5.1 in paper)
Those experiments are implemented in Matlab.
1. Install the package `Tfocs`.
2. Run the file `experiments_model_based_regression.m`.


## How to run the MRI Reconstruction experiments (Section 5.2 in paper)

1. Check (and modify if necessary) the configuration file `config.py`. It specifies the directory paths for the data and results. By default, the data should be stored in the subdirectory `raw_data` and results and model weights are stored in the subdirectory `results`.
2. Download the [fastMRI Knee-MRI dataset](https://fastmri.org/dataset/) and place it in the data folder specified in the configuration. 
3. Prepare the data by running `data_management.py`.
4. Train networks using the scripts named `script_train_*.py`. 
5. Test the trained models on the test set and estimation set with the files `test_itnet_*.py`. 
6. Run the file `UQ_Unet_restterm.py` to calculate the expectation and variance of the restterm. (Algorithm 1 in the paper)
7. Run the file `finding_gamma.py` to calculate the correct gamma value. (Appendix Section A of the paper)
8. Run the file `UQ_Unet.py` to calculate hit rates on the test set for different methods to obtain confidence intervals (Algorithm 2 in the paper)
9. Run the file `boxplot.py` and `confidence_intervals.py` for visual representation.


## Requirements
The following packages were used for running the Python code, other version might work as well.

`matplotlib` *(v3.7.2)*  
`numpy` *(v1.24.3)*  
`pandas` *(v2.0.3)*  
`piq` *(v0.7.0)*  
`python` *(v3.8.19)*  
`torch` *(v1.9.0)*  
`scikit-image` *(v1.0.2)*  
`scipy` *(v1.9.1)*  
`torchvision` *(v0.10.0)*  
`tqdm` *(v4.65.0)*   
`h5py` *(v3.7.0)* 

## Acknowledgements 
The implementation of the U-Net is based on and adapt from https://github.com/mateuszbuda/brain-segmentation-pytorch/.
The implementation of the It-Net and other parts of the code are based on the project https://github.com/jmaces/robust-nets/tree/master/fastmri-radial by M. Genzel, J. Macdonald, and M. März.
Processing the fastMRI data is done with the code from https://github.com/facebookresearch/fastMRI. All code in fastmri_utils_new is from this Github.
The package `Tfocs`is available at https://cvxr.com/tfocs/download/. 

