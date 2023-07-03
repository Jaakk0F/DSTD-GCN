import pdb

import numpy as np
from torch.utils.data import Dataset

from . import utils


class CMUMocap(Dataset):

    def __init__(self,
                 data_path,
                 actions="all",
                 input_n=20,
                 output_n=10,
                 dct_used=15,
                 sample_rate=2,
                 scale=False,
                 scaler=None,
                 data_3d=True,
                 test_mode="all",
                 mirror=False,
                 padding=True):
        self.dct_used = dct_used

        acts = utils.define_actions(actions, "cmu")

        # 3D data or angular data
        if data_3d:
            all_seqs, dim_ignore, dim_used = utils.load_data_cmu_3d(data_path, acts, sample_rate, input_n, output_n,
                                                                    test_mode)
            if mirror:  # mirror augmentation, for now it only support 3D data
                all_seqs_m = self.get_mirror(all_seqs)
                all_seqs = np.concatenate((all_seqs, all_seqs_m), axis=0)
        else:
            # not implemented yet
            pass

        self.all_seqs = all_seqs  # (n, t, v * c)
        self.dim_used = dim_used

        # [0, 1, 7, 13] joints has no std
        # [0, 1, 2, 3, 4, 5, 21, 22, 23, 39, 40, 41]

        if padding:  # pad output with last input
            pad_idx = np.repeat([input_n - 1], output_n)
            i_idx = np.append(np.arange(0, input_n), pad_idx)
            pad_idx_inv = np.repeat([output_n], output_n)
            i_idx_inv = np.append(np.arange(output_n, output_n + input_n)[::-1], pad_idx_inv)
        else:  # output with ground truth
            i_idx = np.arange(0, input_n + output_n)
            i_idx_inv = i_idx[::-1]
        all_seqs = all_seqs[:, :, dim_used]
        self.input_seqs = all_seqs[:, i_idx, :].copy()
        self.input_seqs_inv = all_seqs[:, i_idx_inv, :].copy()
        self.output_seqs = all_seqs.copy()

        # NOTE: the time transformation only intialize, not for input and output, scale transformation are intialized for input and output, not for all seq
        if dct_used > 0:
            self.time_tsfm = utils.TimeTransform(input_n + output_n, dct_used)
            # self.input_seqs = self.time_tsfm.transform(self.input_seqs)
            # self.output_seqs = self.time_tsfm.transform(self.output_seqs)
        else:
            self.time_tsfm = None

        if scale:
            if scaler is not None:
                self.scale_tsfm = scaler
            else:
                # min-max norm
                # global_max = np.max(self.all_seqs)
                # global_min = np.min(self.all_seqs)
                # self.scale_tsfm = utils.MinMaxNorm(global_min, global_max)
                n, t, vc = all_seqs.shape
                all_seqs_scale = all_seqs.reshape(n * t, vc)
                global_mean = np.mean(all_seqs_scale, axis=0)
                global_std = np.std(all_seqs_scale, axis=0)
                self.scale_tsfm = utils.MeanStdNorm(global_mean, global_std)
            self.input_seqs = self.scale_tsfm.transform(self.input_seqs)
            self.input_seqs_inv = self.scale_tsfm.transform(self.input_seqs_inv)
            self.output_seqs = self.scale_tsfm.transform(self.output_seqs)
        else:
            self.scale_tsfm = None

        # calculate weight matrix of body
        n, t, vc = self.all_seqs.shape
        all_seqs_reshape = self.all_seqs.reshape(n, t, vc // 3, 3)
        all_seqs_reshape = np.abs(all_seqs_reshape[:, 1:, :, :] - all_seqs_reshape[:, :-1, :, :])
        joint_weight = np.mean(all_seqs_reshape, (0, 1, 3))
        self.joint_weight_all = (joint_weight - joint_weight.min()) / (joint_weight.max() - joint_weight.min())
        self.joint_weight_use = self.joint_weight_all[np.unique(dim_used // 3)]

    def get_mirror(self, all_seqs):
        all_seqs_m = all_seqs.copy()
        n, t, vc = all_seqs_m.shape
        all_seqs_m = all_seqs_m.reshape((n, t, vc // 3, 3))
        all_seqs = all_seqs.reshape((n, t, vc // 3, 3))
        right = [2, 3, 4, 5, 6, 21, 22, 23, 24, 27, 25, 26, 28]
        left = [8, 9, 10, 11, 12, 30, 31, 32, 33, 36, 24, 35, 37]
        all_seqs_m[:, :, right] = all_seqs[:, :, left]
        all_seqs_m[:, :, left] = all_seqs[:, :, right]
        all_seqs_m[..., 0] = -all_seqs_m[..., 0]
        all_seqs = all_seqs.reshape((n, t, vc))
        all_seqs_m = all_seqs_m.reshape((n, t, vc))
        return all_seqs_m

    def __len__(self):
        return np.shape(self.input_seqs)[0]

    def __getitem__(self, item):
        return (
            self.input_seqs[item],
            self.input_seqs_inv[item],
            self.output_seqs[item],
            self.all_seqs[item],
        )
