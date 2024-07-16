import itertools
import os
import pathlib
import random

from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import pandas as pd
import torch

import operators
from fastmri_utils_new.data import transforms
from fastmri_utils_new import fftc

from config import DATA_PATH, RESULTS_PATH
from operators import (
    prep_fft_channel,
    rotate_real,
    to_complex,
    unprep_fft_channel,
)


"""
Function for creating the dataset and managing it with the option to apply sparsity and noise to the image.
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


# ----- utilities -----


def filter_acquisition_fs(df):
    return df[
        (df["acquisition"] == "CORPDFS_FBK")
        | (df["acquisition"] == "CORPDFS_FBKREPEAT")
    ]


def filter_acquisition_no_fs(df):
    return df[df["acquisition"] == "CORPD_FBK"]


def explore_dataset(path, savepath):

    collected_data = []
    for fname in pathlib.Path(path).iterdir():
        f = h5py.File(fname, "r")
        kspace = f["kspace"]
        data_dict = {
            "fname": fname,
            "kspace_shape": kspace.shape,
        }
        data_dict.update(f.attrs)
        collected_data.append(data_dict)

    df = pd.DataFrame(collected_data)
    df.to_pickle(savepath)


# ----- sampling and shuffling -----


class RandomBlockSampler(torch.utils.data.Sampler):
    """ Samples elements randomly from consecutive blocks. """

    def __init__(self, data_source, block_size):
        self.data_source = data_source
        self.block_size = block_size

    @property
    def num_samples(self):
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        blocks = itertools.groupby(
            range(self.num_samples), lambda k: k // self.block_size
        )
        block_list = [list(group) for key, group in blocks]
        random.shuffle(block_list)
        return itertools.chain(*block_list)

    def __len__(self):
        return self.num_samples


# ----- data transforms -----
class JointRandomCrop(object):
    """ Joint random cropping transform for (input, target) image pairs. """

    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        iw, ih = imgs[0].shape[-2:]  # image width and height
        cw, ch = self.size  # crop width and height

        # sample random corner of cropping area
        w0 = random.randint(0, iw - cw) if iw > cw else 0
        h0 = random.randint(0, ih - ch) if ih > ch else 0
        return tuple(img[..., w0 : w0 + cw, h0 : h0 + ch] for img in imgs)


class MaskedFourierInversion(object):
    """ Inverse subsampled Fourier transform on (meas, target, mask) triples.

    Inverts zero filled meas to image domain and returns (inv, target) pair.

    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        kspace, mask, target = inputs
        inv = (fftc.ifft2c_new(prep_fft_channel(kspace)))
        return inv, target


class ComplexMagnitude(object):
    """ Removes a complex channel from (input, target) image pairs.

    Returns the magnitude of an image as a single channel. If the image has no
    complex channel, then it is passed on unaltered.

    """

    def __init__(self):
        pass

    def __call__(self, imgs):
        return tuple(
            [
                rotate_real(img)[..., 0:1, :, :]
                if img.shape[-3] == 2
                else torch.abs(img)
                for img in imgs
            ]
        )


class CropOrPadAndResimulate(object):
    """ Generates simulated measurements from (input, mask, target) triples.

    Returns (new_input, new_mask, target) triples.

    Uses target, e.g. obtained from RSS reconstructions, crops or pads it,
    recalculates the measurements of the cropped data, applies the cropped
    mask.

    Requires mask to be a mask generating function, not a precomputed mask.
    (See AbstractMRIDataset proprety `keep_mask_as_func`.)

    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, inputs):
        kspace, mask, target = inputs

        # pad if necessary
        p1 = max(0, self.shape[0] - target.shape[-2])
        p2 = max(0, self.shape[1] - target.shape[-1])
        target_padded = torch.nn.functional.pad(
            target, (p2 // 2, -(-p2 // 2), p1 // 2, -(-p1 // 2)),
        )

        # crop if necessary
        target_cropped = transforms.center_crop(target_padded, self.shape)

        # resimulate
        kspace_cropped = fftc.fft2c_new(prep_fft_channel(target_cropped))
        new_mask = mask(kspace_cropped.shape).expand_as(kspace_cropped)
        new_kspace = unprep_fft_channel(kspace_cropped * new_mask)
        new_mask = unprep_fft_channel(new_mask)

        tgs = target.shape[-3]
        if not tgs == 2:
            target_cropped = target_cropped[
                ..., ((tgs // 2) // 2) * 2 : ((tgs // 2) // 2) * 2 + 2, :, :
            ]

        return new_kspace, new_mask, target_cropped


class CenterCrop(object):
    """ Crops (input, target) image pairs to have matching size. """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, imgs):
        return tuple([transforms.center_crop(img, self.shape) for img in imgs])


class Flatten(object):
    """ Flattens selected dimensions of tensors. """

    def __init__(self, start_dim, end_dim):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, inputs):
        return tuple(
            [torch.flatten(x, self.start_dim, self.end_dim) for x in inputs]
        )


class Normalize(object):
    """ Normalizes (input, target) pairs with respect to target or input. """

    def __init__(self, p=2, reduction="sum", use_target=True):
        self.p = p
        self.reduction = reduction
        self.use_target = use_target

    def __call__(self, inputs):
        if self.use_target:
            tar = inputs[-1]
        else:
            tar = unprep_fft_channel(
                fftc.ifft2_new(prep_fft_channel(inputs[0]))
            )
        norm = torch.norm(tar, p=self.p)
        if self.reduction == "mean" and not self.p == "inf":
            norm /= np.prod(tar.shape) ** (1 / self.p)
        if len(inputs) == 2:
            return inputs[0] / norm, inputs[1] / norm
        else:
            return inputs[0] / norm, inputs[1], inputs[2] / norm




# ----- datasets -----
class AbstractMRIDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        subset,
        filter=None,
        num_sym_slices=0,
        multi_slice_gt=False,
        keep_mask_as_func=False,
        transform=None,
        sparse = False,
        sparsity_level = None,
    ):
        # load data set meta data
        if subset == "train":
            metafile = "singlecoil_train_metadata.pkl"
        elif subset == "train_larger":
            metafile = "singlecoil_train_larger_metadata.pkl"
        elif subset == "train_smaller":
            metafile = "singlecoil_train_smaller_metadata.pkl"
        elif subset == "val":
            metafile = "singlecoil_val_metadata.pkl"
        elif subset == "val_even_smaller":
            metafile = "singlecoil_val_even_smaller_metadata.pkl"
        elif subset == "test_bigger":
            metafile = "singlecoil_test_bigger_metadata.pkl"
        elif subset == "rest":
            metafile = "singlecoil_rest_term_analysis_data.pkl"
        df = pd.read_pickle(os.path.join(RESULTS_PATH, metafile))

        if isinstance(filter, (list, tuple)):
            for f in filter:
                df = f(df)
            self.df = df
        else:
            self.df = filter(df) if filter is not None else df
        self.num_sym_slices = num_sym_slices
        self.multi_slice_gt = multi_slice_gt
        self.keep_mask_as_func = keep_mask_as_func
        self.transform = transform
        self.sparse = sparse
        self.sparsity_level = sparsity_level

    def __len__(self):
        # get total number of slices in the data set volumes
        return self.df["kspace_shape"].apply(lambda s: s[0]).sum()

    @property
    def volumes(self):
        return len(self.df)

    def get_slices_in_volume(self, vol_idx):
        # cumulative number of slices in the data set volumes
        vols = self.df["kspace_shape"].apply(lambda s: s[0]).cumsum()

        # first slice of volume
        lo = vols.iloc[vol_idx - 1] if vol_idx > 0 else 0

        # last slice of volume
        hi = vols.iloc[vol_idx]
        return lo, hi

    def __getitem__(self, idx):

        # cumulative number of slices in the data set volumes
        vols = self.df["kspace_shape"].apply(lambda s: s[0]).cumsum() - 1

        # get index of volume and slice within volume
        vol_idx = vols.searchsorted(idx)
        sl_idx = idx if vol_idx == 0 else idx - (vols.iloc[vol_idx - 1] + 1)

        # select slices for multi-slice mode
        sl_from = sl_idx - self.num_sym_slices
        sl_to = sl_idx + self.num_sym_slices + 1

        # load data
        fname = self.df["fname"].iloc[vol_idx]
        data = h5py.File(fname, "r")

        # read out slices and pad if necessary
        sl_num = data["kspace"].shape[0]
        kspace_vol = transforms.to_tensor(
            np.asarray(
                data["kspace"][max(0, sl_from): min(sl_to, sl_num), ...]
            )
        )
        kspace_vol_padded = torch.nn.functional.pad(
            kspace_vol,
            (0, 0, 0, 0, 0, 0, max(0, -sl_from), max(0, sl_to - sl_num)),
        )

        gt = (
            transforms.to_tensor(
                np.asarray(
                    data["reconstruction_rss"][sl_idx: sl_idx + 1, ...]
                )
            )
            if "reconstruction_rss" in data
            else None
        )

        # sparsify the ground truth if sparse=True with sparsity level
        if self.sparse==True:
            gt_sparse = gt
            i = 0
            while float(torch.count_nonzero(gt_sparse)) > gt.shape[1]*gt.shape[2]*self.sparsity_level:
                threshold = torch.max(gt)*(0.2+0.01*i)
                gt_sparse = torch.nn.Threshold(threshold, 0)(gt_sparse)
                i = i + 1
            gt = gt_sparse

        # Data processing
        out = self._process_data(kspace_vol_padded, gt)

        # applies all the function from above which are part of transform
        transformed_output = self.transform(out) if self.transform is not None else out

        # calculate gaussian noise epsilon with mean=0 and chosen variance
        epsilon = torch.normal(0, 60/np.sqrt(2), size=(2, transformed_output[0].shape[1], transformed_output[0].shape[2]))
        epsilon_masked = epsilon*transformed_output[1]

        # add noise to y
        transformed_list = list(transformed_output)
        transformed_list[0] = transformed_list[0]+epsilon_masked
        # Append here epsilon_masked to transformed list if test_itnet/unet is used comment out otherwise
        transformed_list.append(epsilon_masked)
        transformed_output = tuple(transformed_list)
        return transformed_output

    @abstractmethod
    def _process_data(self, kspace_data, gt_data):
        """ Processing of raw data, e.g. masking. """
        pass


class RandomMaskDataset(AbstractMRIDataset):
    def __init__(self, subset, mask_func, **kwargs):
        super(RandomMaskDataset, self).__init__(subset, **kwargs)
        self.mask_func = mask_func

    def _process_data(self, kspace_data, gt_data):
        mask = self.mask_func(kspace_data.shape).expand_as(kspace_data)
        kspace_data = unprep_fft_channel(kspace_data.unsqueeze(0))
        mask = unprep_fft_channel(mask.unsqueeze(0))
        kspace_masked = kspace_data * mask
        gt_data = to_complex(gt_data)
        if self.keep_mask_as_func:
            mask = self.mask_func
        return kspace_masked, mask, gt_data


class AlmostFixedMaskDataset(AbstractMRIDataset):
    def __init__(self, subset, mask_func, seed, **kwargs):
        super(AlmostFixedMaskDataset, self).__init__(subset, **kwargs)
        self.mask_func = mask_func
        self.seed = seed

    def _process_data(self, kspace_data, gt_data):
        mask = self.mask_func(kspace_data.shape, seed=self.seed).expand_as(
            kspace_data
        )
        kspace_data = unprep_fft_channel(kspace_data.unsqueeze(0))
        mask = unprep_fft_channel(mask.unsqueeze(0))
        kspace_masked = kspace_data * mask
        gt_data = to_complex(gt_data)
        if self.keep_mask_as_func:

            def mask(shape):
                return self.mask_func(shape, seed=self.seed)

        return kspace_masked, mask, gt_data


# ---- run data exploration -----

if __name__ == "__main__":
    os.makedirs(RESULTS_PATH, exist_ok=True)
    explore_dataset(
        os.path.join(DATA_PATH, "singlecoil_train"),
        os.path.join(RESULTS_PATH, "singlecoil_train_metadata.pkl"),
    )
    explore_dataset(
        os.path.join(DATA_PATH, "singlecoil_val"),
        os.path.join(RESULTS_PATH, "singlecoil_val_metadata.pkl"),
    )

    # make additional metafiles with a larger train and smaller val set
    train_meta = pd.read_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_train_metadata.pkl")
    )
    val_meta = pd.read_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_val_metadata.pkl")
    )
    """Choose here how the dataset is distributed into the 4 sets train, test, validate and rest"""
    train_larger_meta = train_meta.append(val_meta.iloc[:50])
    train_smaller_meta = train_meta[:900]
    rest_term_analysis_data = train_meta[900:]
    test_small_meta = val_meta.iloc[:100]
    val_smaller_meta = val_meta.iloc[100:]
    test_small_meta.to_pickle(os.path.join(RESULTS_PATH, "singlecoil_test_bigger_metadata.pkl")
                              )
    train_larger_meta.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_train_larger_metadata.pkl")
    )
    train_smaller_meta.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_train_smaller_metadata.pkl")
    )
    rest_term_analysis_data.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_rest_term_analysis_data.pkl")
    )
    val_smaller_meta.to_pickle(
        os.path.join(RESULTS_PATH, "singlecoil_val_even_smaller_metadata.pkl")
    )
