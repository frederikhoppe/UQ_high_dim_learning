import os

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import torch

from fastmri_utils_new import evaluate
from fastmri_utils_new.data import transforms

from tqdm import tqdm

from operators import Fourier, LearnableInverter, im2vec, l2_error, rotate_real

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

"""U-Net and It-Net implementations can be found here."""

# ----- ----- Abstract Base Network ----- -----


class InvNet(torch.nn.Module, metaclass=ABCMeta):
    """ Abstract base class for networks solving linear inverse problems.

    The network is intended for the denoising of a direct inversion of a 2D
    signal from (noisy) linear measurements. The measurement model

        y = Ax + noise

    can be used to obtain an approximate reconstruction x_ from y using, e.g.,
    the pseudo-inverse of A. The task of the network is either to directly
    obtain x from y or denoise and improve this first inversion x_ towards x.

    """

    def __init__(self):
        super(InvNet, self).__init__()

    @abstractmethod
    def forward(self, z):
        """ Applies the network to a batch of inputs z. """
        pass

    def freeze(self):
        """ Freeze all model weights, i.e. prohibit further updates. """
        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _train_step(
        self, batch_idx, batch, loss_func, optimizer, batch_size, acc_steps
    ):
        if len(batch) == 2:
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        else:
            inp, aux, tar = batch
            inp = inp.to(self.device)
            aux = aux.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward((inp, aux))

        loss = loss_func(pred, tar) / acc_steps
        loss.backward()
        if (batch_idx // batch_size + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss * acc_steps, inp, tar, pred

    def _val_step(self, batch_idx, batch, loss_func):
        if len(batch) == 2:
            inp, tar = batch
            inp = inp.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward(inp)
        else:
            inp, aux, tar = batch
            inp = inp.to(self.device)
            aux = aux.to(self.device)
            tar = tar.to(self.device)
            pred = self.forward((inp, aux))
        loss = loss_func(pred, tar)
        return loss, inp, tar, pred

    def _on_epoch_end(
        self,
        epoch,
        save_epochs,
        save_path,
        logging,
        loss,
        inp,
        tar,
        pred,
        v_loss,
        v_inp,
        v_tar,
        v_pred,
        val_data,
    ):

        self._print_info()

        logging = logging.append(
            {
                "loss": loss.item(),
                "val_loss": v_loss.item(),
                "rel_l2_error": l2_error(
                    pred, tar, relative=True, squared=False
                )[0].item(),
                "val_rel_l2_error": l2_error(
                    v_pred, v_tar, relative=True, squared=False
                )[0].item(),
                "l2_error": l2_error(
                    pred, tar, relative=False, squared=False
                )[0].item(),
                "val_l2_error": l2_error(
                    v_pred, v_tar, relative=False, squared=False
                )[0].item(),
            },
            ignore_index=True,
            sort=False,
        )

        print(logging.tail(1))

        val_logging = self.evaluate_dataset(val_data, max_vols=None)

        if (epoch + 1) % save_epochs == 0:
            fig = self._create_figure(
                logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
            )

            os.makedirs(save_path, exist_ok=True)
            torch.save(
                self.state_dict(),
                os.path.join(
                    save_path, "model_weights_epoch{}.pt".format(epoch + 1)
                ),
            )
            logging.to_pickle(
                os.path.join(
                    save_path, "losses_epoch{}.pkl".format(epoch + 1)
                ),
            )
            val_logging.to_pickle(
                os.path.join(
                    save_path, "val_metrics_epoch{}.pkl".format(epoch + 1)
                ),
            )
            fig.savefig(
                os.path.join(save_path, "plot_epoch{}.png".format(epoch + 1)),
                bbox_inches="tight",
            )

        return logging, val_logging

    def _create_figure(
        self, logging, loss, inp, tar, pred, v_loss, v_inp, v_tar, v_pred
    ):
        def _implot(sub, im):
            if im.shape[-3] == 2:  # complex image
                p = sub.imshow(
                    torch.sqrt(im.pow(2).sum(-3))[0, :, :].detach().cpu()
                )
            else:  # real image
                p = sub.imshow(im[0, 0, :, :].detach().cpu())
            return p

        fig, subs = plt.subplots(2, 3, clear=True, num=1, figsize=(8, 6))

        # training and validation loss
        subs[0, 0].set_title("losses")
        subs[0, 0].semilogy(logging["loss"], label="train")
        subs[0, 0].semilogy(logging["val_loss"], label="val")
        subs[0, 0].legend()

        # validation input
        p10 = _implot(subs[1, 0], v_inp)
        subs[1, 0].set_title("val input")
        plt.colorbar(p10, ax=subs[1, 0])

        # validation output
        p01 = _implot(subs[0, 1], v_pred)
        subs[0, 1].set_title(
            "val output:\n ||x0-xrec||_2 / ||x0||_2 = "
            "{:1.2e}".format(logging["val_rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p01, ax=subs[0, 1])

        # validation target
        p11 = _implot(subs[1, 1], v_tar)
        subs[1, 1].set_title("val target")
        plt.colorbar(p11, ax=subs[1, 1])

        # validation difference
        p12 = _implot(subs[1, 2], v_pred - v_tar)
        subs[1, 2].set_title("val diff: x0 - x_pred")
        plt.colorbar(p12, ax=subs[1, 2])

        # training output
        p02 = _implot(subs[0, 2], pred)
        subs[0, 2].set_title(
            "train output:\n ||x0-xrec||_2 / ||x0||_2 = "
            "{:1.2e}".format(logging["rel_l2_error"].iloc[-1])
        )
        plt.colorbar(p02, ax=subs[0, 2])

        return fig

    def _add_to_progress_bar(self, dict):
        """ Can be overwritten by child classes to add to progress bar. """
        return dict

    def _on_train_end(self, save_path, logging, val_logging):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "model_weights.pt")
        )
        logging.to_pickle(os.path.join(save_path, "losses.pkl"),)
        val_logging.to_pickle(os.path.join(save_path, "val_metrics.pkl"),)

    def _print_info(self):
        """ Can be overwritten by child classes to print at epoch end. """
        pass

    def train_on(
        self,
        train_data,
        val_data,
        num_epochs,
        batch_size,
        loss_func,
        save_path,
        save_epochs=50,
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 2e-4, "eps": 1e-3},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 1.0},
        acc_steps=1,
        train_transform=None,
        val_transform=None,
        train_loader_params={"shuffle": True},
        val_loader_params={"shuffle": False},
    ):
        optimizer = optimizer(self.parameters(), **optimizer_params)
        scheduler = scheduler(optimizer, **scheduler_params)

        train_data.transform = train_transform
        val_data.transform = val_transform

        train_loader_params = dict(train_loader_params)
        val_loader_params = dict(val_loader_params)
        if "sampler" in train_loader_params:
            train_loader_params["sampler"] = train_loader_params["sampler"](
                train_data
            )
        if "sampler" in val_loader_params:
            val_loader_params["sampler"] = val_loader_params["sampler"](
                val_data
            )

        data_load_train = torch.utils.data.DataLoader(
            train_data, batch_size, **train_loader_params
        )
        data_load_val = torch.utils.data.DataLoader(
            val_data, batch_size, **val_loader_params
        )

        logging = pd.DataFrame(
            columns=["loss", "val_loss", "rel_l2_error", "val_rel_l2_error", "l2_error", "val_l2_error"]
        )

        for epoch in range(num_epochs):
            self.train()  # make sure we are in train mode
            t = tqdm(
                enumerate(data_load_train),
                desc="epoch {} / {}".format(epoch, num_epochs),
                total=-(-len(train_data) // batch_size),
            )
            optimizer.zero_grad()
            loss = 0.0
            for i, batch in t:
                loss_b, inp, tar, pred = self._train_step(
                    i, batch, loss_func, optimizer, batch_size, acc_steps
                )
                t.set_postfix(
                    **self._add_to_progress_bar({"loss": loss_b.item()})
                )
                loss += loss_b
            loss /= i + 1

            with torch.no_grad():
                self.eval()  # make sure we are in eval mode
                scheduler.step()
                v_loss = 0.0
                for i, v_batch in enumerate(data_load_val):
                    v_loss_b, v_inp, v_tar, v_pred = self._val_step(
                        i, v_batch, loss_func
                    )
                    v_loss += v_loss_b
                v_loss /= i + 1

                logging, val_logging = self._on_epoch_end(
                    epoch,
                    save_epochs,
                    save_path,
                    logging,
                    loss,
                    inp,
                    tar,
                    pred,
                    v_loss,
                    v_inp,
                    v_tar,
                    v_pred,
                    val_data,
                )

        self._on_train_end(save_path, logging, val_logging)
        return logging

    def evaluate_dataset(self, data, max_vols=None):
        """ Evaluates performance measures per volume. """
        logging = pd.DataFrame(
            index=range(data.volumes), columns=["nmse", "psnr", "ssim"],
        )
        num = (
            min(data.volumes, max_vols)
            if max_vols is not None
            else data.volumes
        )
        t = tqdm(range(num))
        for k in t:
            lo, hi = data.get_slices_in_volume(k)
            pred_list, tar_list = [], []
            for sl in range(lo, hi):
                slice_data = data[sl]
                if len(slice_data) == 3:
                    inp, aux, tar = slice_data
                    inp = inp.to(self.device).unsqueeze(0)
                    aux = aux.to(self.device).unsqueeze(0)
                    tar = tar.to(self.device).unsqueeze(0)
                    pred = self.forward((inp, aux))
                else:
                    inp, tar = slice_data
                    inp = inp.to(self.device).unsqueeze(0)
                    tar = tar.to(self.device).unsqueeze(0)
                    pred = self.forward(inp)
                # make complex signals real if necessary
                if tar.shape[-3] == 2:
                    tar = rotate_real(tar)[..., 0:1, :, :]
                if pred.shape[-3] == 2:
                    pred = rotate_real(pred)[..., 0:1, :, :]
                tar_list.append(tar.detach().cpu())
                pred_list.append(pred.detach().cpu())
            tar_np = torch.cat(tar_list, dim=0).squeeze(1).numpy()
            pred_np = torch.cat(pred_list, dim=0).squeeze(1).numpy()
            logging.iloc[k] = {
                "nmse": evaluate.nmse(tar_np, pred_np),
                "psnr": evaluate.psnr(tar_np, pred_np),
                "ssim": evaluate.ssim(tar_np, pred_np),
            }
        print(
            pd.DataFrame(
                {
                    "min": logging.min(),
                    "mean": logging.mean(),
                    "max": logging.max(),
                }
            )
        )
        return logging


# ----- ----- Iterative Networks ----- -----
class IterativeNet(InvNet):
    def __init__(
        self,
        subnet,
        num_iter,
        lam,
        lam_learnable,
        final_dc=True,
        resnet_factor=1.0,
        concat_mask=False,
        multi_slice=False,
        ee=False,
    ):
        super(IterativeNet, self).__init__()
        if isinstance(subnet, list):
            self.subnet = torch.nn.ModuleList(subnet)
        else:
            self.subnet = subnet
        self.num_iter = num_iter
        self.final_dc = final_dc
        self.resnet_factor = resnet_factor
        self.concat_mask = concat_mask
        self.multi_slice = multi_slice
        if not isinstance(lam, (list, tuple)):
            lam = [lam] * num_iter
        if not isinstance(lam_learnable, (list, tuple)):
            lam_learnable = [lam_learnable] * len(lam)

        self.lam = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(lam[it]), requires_grad=lam_learnable[it]
                )
                for it in range(len(lam))
            ]
        )
        # for fully learned nets
        if isinstance(ee, torch.Tensor):
            self.inverter = LearnableInverter(
                ee.shape[-2:], ee, learnable=True
            )
        self.ee = ee

    def forward(self, inp):
        y, mask = inp
        Op = Fourier(mask)
        mask_im = mask[0, 0, :, :]  # assume same mask per batch and channel
        y_masked = im2vec(y)[:, :, im2vec(mask_im) > 0]
        if isinstance(self.ee, torch.Tensor):  # fully learned
            xinv = self.inverter(y_masked)
        else:
            xinv = Op.inv(y_masked)
        for it in range(self.num_iter):
            # subnet step
            if self.multi_slice:
                xinvs = xinv.shape[-3]
                xinv_res = xinv[
                    ...,
                    ((xinvs // 2) // 2) * 2 : ((xinvs // 2) // 2) * 2 + 2,
                    :,
                    :,
                ]
            else:
                xinv_res = xinv

            if isinstance(self.subnet, torch.nn.ModuleList):
                xinv = self.resnet_factor * xinv_res + self.subnet[it](
                torch.cat([xinv, mask[:, 0:1, :, :]], dim=1)
                if self.concat_mask
                else xinv
            )
            else:
                xinv = self.resnet_factor * xinv_res + self.subnet(
                torch.cat([xinv, mask[:, 0:1, :, :]], dim=1)
                if self.concat_mask
                else xinv
            )

            # data consistency step
            if (self.final_dc) or (
                (not self.final_dc) and it < self.num_iter - 1
            ):
                if self.multi_slice:
                    xinv = xinv - self.lam[it] * Op.adj(
                        Op(xinv)
                        - y_masked[
                            ...,
                            ((xinvs // 2) // 2) * 2 : ((xinvs // 2) // 2) * 2
                            + 2,
                            :,
                        ]
                    )
                else:
                    xinv = xinv - self.lam[it] * Op.adj(Op(xinv) - y_masked)

        return transforms.center_crop(xinv, (320, 320))

    def _print_info(self):
        print("Current lambda(s):")
        print([self.lam[it].item() for it in range(len(self.lam))])
        print([self.lam[it].requires_grad for it in range(len(self.lam))])


# ----- ----- U-Net ----- -----


class UNet(InvNet):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self, in_channels=1, out_channels=1, base_features=32, drop_factor=0.0,
    ):
        # set properties of UNet
        super(UNet, self).__init__()

        self.encoder1 = UNet._conv_block(
            in_channels,
            base_features,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._conv_block(
            base_features,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._conv_block(
            base_features * 2,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._conv_block(
            base_features * 4,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._conv_block(
            base_features * 8,
            base_features * 16,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16, base_features * 8, kernel_size=2, stride=2,
        )
        self.decoder4 = UNet._conv_block(
            base_features * 16,
            base_features * 8,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8, base_features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._conv_block(
            base_features * 8,
            base_features * 4,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4, base_features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._conv_block(
            base_features * 4,
            base_features * 2,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._conv_block(
            base_features * 2,
            base_features,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    @staticmethod
    def _conv_block(in_channels, out_channels, drop_factor, block_name):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )



