import matplotlib as mpl
import torch
import torchvision

from data_management import (
    AlmostFixedMaskDataset,
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    filter_acquisition_no_fs,
)
from networks import IterativeNet, UNet
from operators import RadialMaskFunc
import os

"""
This script applies a trained network to the test data and saves the reconstruction using the given It-Net
"""

path = "results/test_results_unet_noisy/"
path_model = "results/radial_220_unet_noisy_train_phase_1/"

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:1")
torch.cuda.set_device(0)

# ----- network configuration -----
subnet_params = {
    "in_channels": 2,
    "drop_factor": 0.0,
    "base_features": 24,
    "out_channels": 2,
}
subnet = UNet

it_net_params = {
    "num_iter": 1,
    "lam": 1.0,
    "lam_learnable":False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "concat_mask": False,
    "multi_slice": False,
}
# ----- data configuration -----

mask_func = RadialMaskFunc((320, 320), 220)

test_data_params = {
    "mask_func": mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((320, 320)),
            Flatten(0, -3),
            Normalize(reduction="sum", use_target=True),
        ]),
    "sparse": False,
    "sparsity_level": 1,
}
test_data = AlmostFixedMaskDataset

# ------ loading the network ------

subnet_tmp = subnet(**subnet_params).to(device)
it_net_tmp = IterativeNet(subnet_tmp, **it_net_params).to(device)
it_net_tmp.load_state_dict(
    torch.load(os.path.join(path_model,
        "model_weights.pt"),
        map_location=torch.device(device),
    )
)
subnet = it_net_tmp.subnet
it_net = IterativeNet(subnet, **it_net_params).to(device)
it_net.freeze()
it_net.eval()

test_data = test_data("test_bigger", **test_data_params)
test_loader_params = {"shuffle": False}
test_loader_params = dict(test_loader_params)
if "sampler" in test_loader_params:
        test_loader_params["sampler"] = test_loader_params["sampler"](test_data)
data_load_test = torch.utils.data.DataLoader(
            test_data, 1, **test_loader_params
        )

# ----eval on test data ----

rec_test_all = []
kspace_test_all = []
tar_test_all = []
mask_test_all = []
epsilon_test_all = []

with torch.no_grad():
    for i, v_batch in enumerate(data_load_test):
        rec_test = torch.zeros_like(
            it_net((v_batch[0].to(device), v_batch[1].to(device)))
        )
        rec_test += it_net(
            (v_batch[0].to(device), v_batch[1].to(device))
            )
        rec_test_all.append(rec_test)
        kspace_test_all.append(v_batch[0].to(device))
        tar_test_all.append(v_batch[2].to(device))
        mask_test_all.append(v_batch[1].to(device))
        epsilon_test_all.append(v_batch[3].to(device))


rec_test_all = torch.cat(rec_test_all, dim=0)
kspace_test_all = torch.cat(kspace_test_all, dim=0)
tar_test_all = torch.cat(tar_test_all, dim=0)
mask_test_all = torch.cat(mask_test_all, dim=0)
epsilon_test_all = torch.cat(epsilon_test_all, dim=0)

torch.save(rec_test_all, os.path.join(path,"rec_test_all.pt"))
torch.save(kspace_test_all,os.path.join(path,"kspace_test_all.pt"))
torch.save(tar_test_all, os.path.join(path,"tar_test_all.pt"))
torch.save(mask_test_all, os.path.join(path,"mask_test_all.pt"))
torch.save(epsilon_test_all, os.path.join(path,"epsilon_test_all.pt"))
