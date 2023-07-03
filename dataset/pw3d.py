import pdb
import pickle as pkl
from os import walk

import numpy as np
from torch.utils.data import Dataset

from . import utils


class PW3D(Dataset):

    def __init__(self,
                 data_path,
                 input_n=20,
                 output_n=10,
                 dct_used=15,
                 mode="train",
                 scale=False,
                 scaler=None,
                 mirror=False,
                 padding=True):
        self.dct_used = dct_used

        self.in_n = input_n
        self.out_n = output_n
        #self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        all_seqs = []
        files = []
        # load data
        for (dirpath, dirnames, filenames) in walk(data_path):
            files.extend(filenames)
        for f in files:
            with open(data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        # self.all_seqs = all_seqs[:, (their_input_n - input_n):, :]
        all_seqs *= 1000
        #all_seqs = all_seqs[:, (their_input_n - input_n):, 3:]
        # all_seqs = all_seqs[:, 0:, :]
        if mirror:  # mirror augmentation, for now it only support 3D data
            all_seqs_m = self.get_mirror(all_seqs)
            all_seqs = np.concatenate((all_seqs, all_seqs_m), axis=0)
        dim_used = np.array(range(3, all_seqs.shape[2]))
        self.all_seqs = all_seqs
        self.dim_used = dim_used

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
        right = [1, 4, 7, 10, 13, 16, 18, 20, 22]
        left = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        all_seqs_m[:, :, right] = all_seqs[:, :, left]
        all_seqs_m[:, :, left] = all_seqs[:, :, right]
        all_seqs_m[..., 0] = -all_seqs_m[..., 0]
        all_seqs = all_seqs.reshape((n, t, vc))
        all_seqs_m = all_seqs_m.reshape((n, t, vc))
        return all_seqs_m

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return (
            self.input_seqs[item],
            self.input_seqs_inv[item],
            self.output_seqs[item],
            self.all_seqs[item],
        )
