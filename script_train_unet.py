import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchvision

from piq import MultiScaleSSIMLoss

from config import RESULTS_PATH
from data_management import (
    AlmostFixedMaskDataset,
    CropOrPadAndResimulate,
    Flatten,
    Normalize,
    filter_acquisition_no_fs,
)
from networks import IterativeNet, UNet
from operators import RadialMaskFunc, rotate_real

"""Execute this file for training the U-Net with you can specify if you want to use sparse data,
how much undersampling should be used and the type of the It-Net.
Changing the noise level can only be done in data_management.
"""

"""
Based on https://github.com/jmaces/robust-nets/tree/master/fastmri-radial 
and modified in agreement with their license. 

-----
 
MIT License

Copyright (c) 2020 Martin Genzel, Jan Macdonald, Maximilian März

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:1")

train_phases = 1

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
    "lam_learnable": False,
    "final_dc": False,
    "resnet_factor": 1.0,
    "concat_mask": False,
    "multi_slice": False,
}

# ----- training configuration -----
msssimloss = MultiScaleSSIMLoss(data_range=5e10)
maeloss = torch.nn.L1Loss(reduction="mean")

def msssim_l1_loss(pred, tar):
    return 0.7 * msssimloss(
        rotate_real(pred)[:, 0:1, ...], rotate_real(tar)[:, 0:1, ...],
    ) + 0.3 * maeloss(
        rotate_real(pred)[:, 0:1, ...], rotate_real(tar)[:, 0:1, ...],
    )


train_params = {
    "num_epochs": [20],
    "batch_size": [40],
    "loss_func": msssim_l1_loss,
    "save_path": [
        os.path.join(
            RESULTS_PATH,
            "radial_220_unet_with_noise_"
            "train_phase_{}".format((i + 1) % (train_phases + 1)),
        )
        for i in range(train_phases + 1)
    ],
    "save_epochs": 1,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [{"lr": 2e-4, "eps": 1e-4, "weight_decay": 1e-5}],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1],
    "train_transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((320, 320)),
            Flatten(0, -3),
            Normalize(reduction="sum", use_target=True),
        ]
    ),
    "val_transform": torchvision.transforms.Compose(
        [
            CropOrPadAndResimulate((320, 320)),
            Flatten(0, -3),
            Normalize(reduction="sum", use_target=True),
        ],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----
mask_func = RadialMaskFunc((320,320), 220)


train_data_params = {
    "mask_func": mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "sparse": False,
    "sparsity_level": 1,
}
train_data = AlmostFixedMaskDataset

val_data_params = {
    "mask_func": mask_func,
    "seed": 1,
    "filter": [filter_acquisition_no_fs],
    "num_sym_slices": 0,
    "multi_slice_gt": False,
    "keep_mask_as_func": True,
    "sparse": False,
    "sparsity_level": 1,
}
val_data = AlmostFixedMaskDataset


# ------ save hyperparameters -------
os.makedirs(train_params["save_path"][-1], exist_ok=True)
with open(
    os.path.join(train_params["save_path"][-1], "hyperparameters.txt"), "w"
) as file:
    for key, value in subnet_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in it_net_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in train_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    for key, value in val_data_params.items():
        file.write(key + ": " + str(value) + "\n")
    file.write("train_phases" + ": " + str(train_phases) + "\n")

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)
train_data = train_data("train_smaller", **train_data_params)
val_data = val_data("val_even_smaller", **val_data_params)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    it_net.train_on(train_data, val_data, **train_params_cur)


# ----- pick best weights and save them ----
ssim = []
psnr = []
nmse = []
for i in range(1, train_params["num_epochs"][-1] + 1):
    t = pandas.read_pickle(
        os.path.join(
            train_params["save_path"][-2], "val_metrics_epoch{}.pkl".format(i)
        )
    )
    ssim.append(t["ssim"].mean())
    psnr.append(t["psnr"].mean())
    nmse.append(t["nmse"].mean())

ssim = np.array(ssim)
ssim_max = np.argmax(ssim) + 1

psnr = np.array(psnr)
psnr_max = np.argmax(psnr) + 1

nmse = np.array(nmse)
nmse_min = np.argmin(nmse) + 1

ssim_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(ssim_max),
    )
)
torch.save(
    ssim_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_maxSSIM_{}.pt".format(ssim_max),
    ),
)

psnr_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(psnr_max),
    )
)
torch.save(
    psnr_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_maxPSNR_{}.pt".format(psnr_max),
    ),
)

nmse_w = torch.load(
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_epoch{}.pt".format(nmse_min),
    )
)
torch.save(
    nmse_w,
    os.path.join(
        train_params["save_path"][-2],
        "model_weights_minNMSE_{}.pt".format(nmse_min),
    ),
)

# plot overview
fig = plt.figure()
plt.plot(ssim, label="ssim")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_ssim.png"),
    bbox_inches="tight",
)

fig = plt.figure()
plt.plot(psnr, label="psnr")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_psnr.png"),
    bbox_inches="tight",
)

fig = plt.figure()
plt.plot(nmse, label="nmse")
plt.legend()
fig.savefig(
    os.path.join(train_params["save_path"][-2], "plot_nmse.png"),
    bbox_inches="tight",
)
