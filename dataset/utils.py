# -*- coding: utf-8 -*-
import os
import pdb

import numpy as np
import torch
from torch.autograd.variable import Variable


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2])

        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + dlta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + dlta

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3])
    return eul


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    return R


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_coordinate_space(channels, R0, T0):
    """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):
        R_diff = data_utils.expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))).reshape(-1)
        channels_rec[ii, :3] = T
        # T_prev = T
        # R_prev = R

    return channels_rec


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in range(poses_out.shape[0]):
        poses_out_list.append(unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename, with_key=False):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    if with_key:  # skip the first line
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def define_actions(action, dataset="h36m"):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    if dataset == "h36m":
        actions = [
            "walking",
            "eating",
            "smoking",
            "discussion",
            "directions",
            "greeting",
            "phoning",
            "posing",
            "purchases",
            "sitting",
            "sittingdown",
            "takingphoto",
            "waiting",
            "walkingdog",
            "walkingtogether",
        ]
    elif dataset == "cmu":
        actions = [
            "basketball",
            "basketball_signal",
            "directing_traffic",
            "jumping",
            "running",
            "soccer",
            "walking",
            "washwindow",
        ]
    elif dataset == "amass":
        pass
    elif dataset == "expi":
        # TODO: change into same format with other dataset
        if action == "pro3-train":
            actions = [
                "2/a-frame",
                "2/around-the-back",
                "2/coochie",
                "2/frog-classic",
                "2/noser",
                "2/toss-out",
                "2/cartwheel",
                "1/a-frame",
                "1/around-the-back",
                "1/coochie",
                "1/frog-classic",
                "1/noser",
                "1/toss-out",
                "1/cartwheel",
            ]
        elif action == "pro3-test":
            actions = [
                "2/crunch-toast",
                "2/frog-kick",
                "2/ninja-kick",
                "1/back-flip",
                "1/big-ben",
                "1/chandelle",
                "1/check-the-change",
                "1/frog-turn",
                "1/twisted-toss",
            ]
        elif action == "pro1-train":
            actions = [
                "2/a-frame",
                "2/around-the-back",
                "2/coochie",
                "2/frog-classic",
                "2/noser",
                "2/toss-out",
                "2/cartwheel",
            ]
        elif action == "pro1-test":
            actions = [
                "1/a-frame",
                "1/around-the-back",
                "1/coochie",
                "1/frog-classic",
                "1/noser",
                "1/toss-out",
                "1/cartwheel",
            ]
        else:
            actions = []
        return actions
    else:
        raise ValueError(f"No such dataset {dataset}")

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "debug":
        return actions[:1]

    # if action == "all_srnn":
    #     return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)


"""all methods above are borrowed from https://github.com/una-dinosauria/human-motion-prediction"""


def define_actions_cmu(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = [
        "basketball",
        "basketball_signal",
        "directing_traffic",
        "jumping",
        "running",
        "soccer",
        "walking",
        "washwindow",
    ]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    raise (ValueError, "Unrecognized action: %d" % action)


def load_data_cmu(path_to_dataset, actions, input_n, output_n, data_std=0, data_mean=0, is_test=False):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = "{}/{}".format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = "{}/{}/{}_{}.txt".format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            if not is_test:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :,]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    if not is_test:
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_cmu_3d(path_to_dataset, actions, sample_rate, input_n, output_n, mode="all"):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape

            exptmps = torch.from_numpy(action_sequence).float().cuda()

            xyz = expmap2xyz_torch_cmu(exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            action_sequence = xyz

            even_list = range(0, n, sample_rate)
            the_sequence = np.array(action_sequence[even_list, :])  # x, 114
            num_frames = len(the_sequence)
            if mode == "all":
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            # this is only for test mode
            elif mode == "8":
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[idx + (source_seq_len - input_n):(idx + source_seq_len +
                                                                             output_n), :]  # 35， 114
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    # data_std[dimensions_to_ignore] = 1.0
    # data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use


def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(
            R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
            R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]),
        )
        eul_remain[:, 2] = torch.atan2(
            R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
            R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]),
        )
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = Variable(torch.zeros(R.shape[0], 4)).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = (Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() +
         torch.mul(torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
             (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)),
             torch.matmul(r1, r1),
         ))
    return R


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = _some_variables_cmu()
    xyz = fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def load_data(
    path_to_dataset,
    subjects,
    actions,
    sample_rate,
    seq_len,
    input_n=10,
    data_mean=None,
    data_std=None,
):
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216

    :param path_to_dataset: path of dataset
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len: past frame length + future frame length
    :param is_norm: normalize the expmap or not
    :param data_std: standard deviation of the expmap
    :param data_mean: mean of the expmap
    :param input_n: past frame length
    :return:
    """

    sampled_seq = []
    complete_seq = []
    # actions_all = define_actions("all")
    # one_hot_all = np.eye(len(actions_all))
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    the_sequence = np.array(action_sequence[even_list, :])
                    num_frames = len(the_sequence)
                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence1 = np.array(action_sequence[even_list, :])
                num_frames1 = len(the_sequence1)

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence2 = np.array(action_sequence[even_list, :])
                num_frames2 = len(the_sequence2)

                fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len, input_n=input_n)
                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    # if is not testing or validation then get the data statistics
    if not (subj == 5 and subj == 11):
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_3d(path_to_dataset, subjects, actions, sample_rate, seq_len, test_mode="8"):
    """

    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    :param path_to_dataset:
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len:
    :return:
    """

    sampled_seq = []
    complete_seq = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(action_sequence[even_list, :])
                    the_seq = Variable(torch.from_numpy(the_sequence)).float().cuda()
                    # remove global rotation and translation
                    the_seq[:, 0:6] = 0
                    p3d = expmap2xyz_torch(the_seq)
                    the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()

                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        # the inputs are from all sequences
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()  # (num_frames - seq_len + 1, seq_len)
                    seq_sel = the_sequence[fs_sel, :]  # (num_frames - seq_len + 1, seq_len, 3)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames1 = len(even_list)
                the_sequence1 = np.array(action_sequence[even_list, :])
                the_seq1 = Variable(torch.from_numpy(the_sequence1)).float().cuda()
                the_seq1[:, 0:6] = 0
                p3d1 = expmap2xyz_torch(the_seq1)
                the_sequence1 = p3d1.view(num_frames1, -1).cpu().data.numpy()

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = "{0}/S{1}/{2}_{3}.txt".format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames2 = len(even_list)
                the_sequence2 = np.array(action_sequence[even_list, :])
                the_seq2 = Variable(torch.from_numpy(the_sequence2)).float().cuda()
                the_seq2[:, 0:6] = 0
                p3d2 = expmap2xyz_torch(the_seq2)
                the_sequence2 = p3d2.view(num_frames2, -1).cpu().data.numpy()

                # original example
                # # print("action:{}".format(action))
                # # print("subact1:{}".format(num_frames1))
                # # print("subact2:{}".format(num_frames2))
                if test_mode == "8":
                    fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len)
                elif test_mode == "256":
                    fs_sel1, fs_sel2 = find_indices_256(num_frames1, num_frames2, seq_len)
                elif test_mode == "all":
                    fs_sel1 = np.array([np.arange(i, i + seq_len) for i in range(num_frames1 - 100)])
                    fs_sel2 = np.array([np.arange(i, i + seq_len) for i in range(num_frames2 - 100)])
                else:
                    raise (ValueError(f"Invalid test_mode {test_mode}"))

                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel1), axis=0)
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence1, axis=0)
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

                # load all examples
                # fs = np.arange(0, num_frames1 - seq_len + 1)
                # fs_sel = fs
                # for i in np.arange(seq_len - 1):
                #     # the inputs are from all sequences
                #     fs_sel = np.vstack((fs_sel, fs + i + 1))
                # fs_sel = fs_sel.transpose()  # (num_frames - seq_len + 1, seq_len)
                # seq_sel = the_sequence1[
                #     fs_sel, :
                # ]  # (num_frames - seq_len + 1, seq_len, 3)
                # if len(sampled_seq) == 0:
                #     sampled_seq = seq_sel
                #     complete_seq = the_sequence1
                # else:
                #     sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                #     complete_seq = np.append(complete_seq, the_sequence1, axis=0)

    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])  # 测试的时候只忽略了[16, 20, 23, 24, 28, 31]
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)
    # sampled seq (N, T, V * C)
    #
    return sampled_seq, dimensions_to_ignore, dimensions_to_use


def get_dct_matrix(N):
    # N - sequence length
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def find_indices_256(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def find_indices_64(num_frames, seq_len):
    # not random choose. as the sequence is short and we want the test set to represent the seq better
    seed = 1234567890
    np.random.seed(seed)

    T = num_frames - seq_len + 1
    n = int(T / 64)
    list0 = np.arange(0, T)
    list1 = np.arange(0, T, (n + 1))
    t = 64 - len(list1)
    if t == 0:
        listf = list1
    else:
        list2 = np.setdiff1d(list0, list1)
        list2 = list2[:t]
        listf = np.concatenate((list1, list2))
    return listf


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


# from forward kinematics
def fkl(angles, parent, offset, rotInd, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.

    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/py#L14

    which originaly based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        # if not rotInd[i]:  # If the list is empty
        #     xangle, yangle, zangle = 0, 0, 0
        # else:
        #     xangle = angles[rotInd[i][0] - 1]
        #     yangle = angles[rotInd[i][1] - 1]
        #     zangle = angles[rotInd[i][2] - 1]
        if i == 0:
            xangle = angles[0]
            yangle = angles[1]
            zangle = angles[2]
            thisPosition = np.array([xangle, yangle, zangle])
        else:
            thisPosition = np.array([0, 0, 0])

        r = angles[expmapInd[i]]

        thisRotation = expmap2rotmat(r)

        if parent[i] == -1:  # Root node
            xyzStruct[i]["rotation"] = thisRotation
            xyzStruct[i]["xyz"] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]["xyz"] = (offset[i, :] + thisPosition).dot(
                xyzStruct[parent[i]]["rotation"]) + xyzStruct[parent[i]]["xyz"]
            xyzStruct[i]["rotation"] = thisRotation.dot(xyzStruct[parent[i]]["rotation"])

    xyz = [xyzStruct[i]["xyz"] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    # xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    return xyz


def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/py#L100

    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = (np.array([
        0,
        1,
        2,
        3,
        4,
        5,
        1,
        7,
        8,
        9,
        10,
        1,
        12,
        13,
        14,
        15,
        13,
        17,
        18,
        19,
        20,
        21,
        20,
        23,
        13,
        25,
        26,
        27,
        28,
        29,
        28,
        31,
    ]) - 1)

    offset = np.array([
        0.000000,
        0.000000,
        0.000000,
        -132.948591,
        0.000000,
        0.000000,
        0.000000,
        -442.894612,
        0.000000,
        0.000000,
        -454.206447,
        0.000000,
        0.000000,
        0.000000,
        162.767078,
        0.000000,
        0.000000,
        74.999437,
        132.948826,
        0.000000,
        0.000000,
        0.000000,
        -442.894413,
        0.000000,
        0.000000,
        -454.206590,
        0.000000,
        0.000000,
        0.000000,
        162.767426,
        0.000000,
        0.000000,
        74.999948,
        0.000000,
        0.100000,
        0.000000,
        0.000000,
        233.383263,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        121.134938,
        0.000000,
        0.000000,
        115.002227,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        151.034226,
        0.000000,
        0.000000,
        278.882773,
        0.000000,
        0.000000,
        251.733451,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        99.999627,
        0.000000,
        100.000188,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        151.031437,
        0.000000,
        0.000000,
        278.892924,
        0.000000,
        0.000000,
        251.728680,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        99.999888,
        0.000000,
        137.499922,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ])
    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def _some_variables_cmu():
    """
    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = (np.array([
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        8,
        9,
        10,
        11,
        12,
        1,
        14,
        15,
        16,
        17,
        18,
        19,
        16,
        21,
        22,
        23,
        24,
        25,
        26,
        24,
        28,
        16,
        30,
        31,
        32,
        33,
        34,
        35,
        33,
        37,
    ]) - 1)

    offset = 70 * np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        1.65674000000000,
        -1.80282000000000,
        0.624770000000000,
        2.59720000000000,
        -7.13576000000000,
        0,
        2.49236000000000,
        -6.84770000000000,
        0,
        0.197040000000000,
        -0.541360000000000,
        2.14581000000000,
        0,
        0,
        1.11249000000000,
        0,
        0,
        0,
        -1.61070000000000,
        -1.80282000000000,
        0.624760000000000,
        -2.59502000000000,
        -7.12977000000000,
        0,
        -2.46780000000000,
        -6.78024000000000,
        0,
        -0.230240000000000,
        -0.632580000000000,
        2.13368000000000,
        0,
        0,
        1.11569000000000,
        0,
        0,
        0,
        0.0196100000000000,
        2.05450000000000,
        -0.141120000000000,
        0.0102100000000000,
        2.06436000000000,
        -0.0592100000000000,
        0,
        0,
        0,
        0.00713000000000000,
        1.56711000000000,
        0.149680000000000,
        0.0342900000000000,
        1.56041000000000,
        -0.100060000000000,
        0.0130500000000000,
        1.62560000000000,
        -0.0526500000000000,
        0,
        0,
        0,
        3.54205000000000,
        0.904360000000000,
        -0.173640000000000,
        4.86513000000000,
        0,
        0,
        3.35554000000000,
        0,
        0,
        0,
        0,
        0,
        0.661170000000000,
        0,
        0,
        0.533060000000000,
        0,
        0,
        0,
        0,
        0,
        0.541200000000000,
        0,
        0.541200000000000,
        0,
        0,
        0,
        -3.49802000000000,
        0.759940000000000,
        -0.326160000000000,
        -5.02649000000000,
        0,
        0,
        -3.36431000000000,
        0,
        0,
        0,
        0,
        0,
        -0.730410000000000,
        0,
        0,
        -0.588870000000000,
        0,
        0,
        0,
        0,
        0,
        -0.597860000000000,
        0,
        0.597860000000000,
    ])
    offset = offset.reshape(-1, 3)

    rotInd = [
        [6, 5, 4],
        [9, 8, 7],
        [12, 11, 10],
        [15, 14, 13],
        [18, 17, 16],
        [21, 20, 19],
        [],
        [24, 23, 22],
        [27, 26, 25],
        [30, 29, 28],
        [33, 32, 31],
        [36, 35, 34],
        [],
        [39, 38, 37],
        [42, 41, 40],
        [45, 44, 43],
        [48, 47, 46],
        [51, 50, 49],
        [54, 53, 52],
        [],
        [57, 56, 55],
        [60, 59, 58],
        [63, 62, 61],
        [66, 65, 64],
        [69, 68, 67],
        [72, 71, 70],
        [],
        [75, 74, 73],
        [],
        [78, 77, 76],
        [81, 80, 79],
        [84, 83, 82],
        [87, 86, 85],
        [90, 89, 88],
        [93, 92, 91],
        [],
        [96, 95, 94],
        [],
    ]
    posInd = []
    for ii in np.arange(38):
        if ii == 0:
            posInd.append([1, 2, 3])
        else:
            posInd.append([])

    expmapInd = np.split(np.arange(4, 118) - 1, 38)

    return parent, offset, posInd, expmapInd


def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = angles.data.shape[0]
    j_n = offset.shape[0]
    p3d = Variable(torch.from_numpy(offset)).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = (torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :])
    return p3d


def main():
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    # numpy implementation
    # with h5py.File('samples.h5', 'r') as h5f:
    #     expmap_gt = h5f['expmap/gt/walking_0'][:]
    #     expmap_pred = h5f['expmap/preds/walking_0'][:]
    expmap_pred = np.array([
        0.0000000,
        0.0000000,
        0.0000000,
        -0.0000001,
        -0.0000000,
        -0.0000002,
        0.3978439,
        -0.4166636,
        0.1027215,
        -0.7767256,
        -0.0000000,
        -0.0000000,
        0.1704115,
        0.3078358,
        -0.1861640,
        0.3330379,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        0.0679339,
        0.2255526,
        0.2394881,
        -0.0989492,
        -0.0000000,
        -0.0000000,
        0.0677801,
        -0.3607298,
        0.0503249,
        0.1819232,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        0.3236777,
        -0.0476493,
        -0.0651256,
        -0.3150051,
        -0.0665669,
        0.3188994,
        -0.5980227,
        -0.1190833,
        -0.3017127,
        1.2270271,
        -0.1010960,
        0.2072986,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.2578378,
        -0.0125206,
        2.0266378,
        -0.3701521,
        0.0199115,
        0.5594162,
        -0.4625384,
        -0.0000000,
        -0.0000000,
        0.1653314,
        -0.3952765,
        -0.1731570,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        2.7825687,
        -1.4196042,
        -0.0936858,
        -1.0348599,
        -2.7419815,
        0.4518218,
        -0.3902033,
        -0.0000000,
        -0.0000000,
        0.0597317,
        0.0547002,
        0.0445105,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
    ])
    expmap_gt = np.array([
        0.2240568,
        -0.0276901,
        -0.7433901,
        0.0004407,
        -0.0020624,
        0.0002131,
        0.3974636,
        -0.4157083,
        0.1030248,
        -0.7762963,
        -0.0000000,
        -0.0000000,
        0.1697988,
        0.3087364,
        -0.1863863,
        0.3327336,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        0.0689423,
        0.2282812,
        0.2395958,
        -0.0998311,
        -0.0000000,
        -0.0000000,
        0.0672752,
        -0.3615943,
        0.0505299,
        0.1816492,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        0.3223563,
        -0.0481131,
        -0.0659720,
        -0.3145134,
        -0.0656419,
        0.3206626,
        -0.5979006,
        -0.1181534,
        -0.3033383,
        1.2269648,
        -0.1011873,
        0.2057794,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.2590978,
        -0.0141497,
        2.0271597,
        -0.3699318,
        0.0128547,
        0.5556172,
        -0.4714990,
        -0.0000000,
        -0.0000000,
        0.1603251,
        -0.4157299,
        -0.1667608,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        2.7811005,
        -1.4192915,
        -0.0932141,
        -1.0294687,
        -2.7323222,
        0.4542309,
        -0.4048152,
        -0.0000000,
        -0.0000000,
        0.0568960,
        0.0525994,
        0.0493068,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
        -0.0000000,
    ])
    xyz1 = fkl(expmap_pred, parent, offset, rotInd, expmapInd)
    xyz2 = fkl(expmap_gt, parent, offset, rotInd, expmapInd)

    exp1 = Variable(torch.from_numpy(np.vstack((expmap_pred, expmap_gt))).float()).cuda()
    xyz = fkl_torch(exp1, parent, offset, rotInd, expmapInd)
    xyz = xyz.cpu().data.numpy()
    print(xyz)


class GraphH36():

    def __init__(self, layout='h36m', strategy='uniform', max_hop=1, dilation=1):
        self.use_joint = {
            2: 0,
            3: 1,
            4: 2,
            5: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7,
            12: 8,
            13: 9,
            14: 10,
            15: 11,
            17: 12,
            18: 13,
            19: 14,
            21: 15,
            22: 16,
            25: 17,
            26: 18,
            27: 19,
            29: 20,
            30: 21,
        }
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A_j

    def get_edge(self, layout):
        if layout == 'h36m':
            self.num_node = 22
            self_link = [(i, i) for i in range(self.num_node)]
            # neighbor_link_ = [(1,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,9),
            #                   (1,10),(10,11),(11,12),(12,13),(11,14),(14,15),
            #                   (15,16),(16,17),(11,18),(18,19),(19,20),(20,21)]
            neighbor_link__ = [(5, 4), (10, 9), (4, 3), (9, 8), (3, 2), (8, 7), (13, 12), (14, 12), (21, 19), (22, 19),
                               (19, 18), (29, 27), (30, 27), (27, 26), (18, 17), (26, 25), (17, 13), (25, 13), (14, 13),
                               (15, 14)]

            neighbor_link_ = []
            for bp in neighbor_link__:
                neighbor_link_.append([self.use_joint[bp[0]], self.use_joint[bp[1]]])

            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link_]
            self.edge = self_link + neighbor_link
            self.center = 7

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A_j = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A_j = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A_j = A


class JointBoneTransformH36:

    def __init__(self):
        self.bone_pair = [
            (5, 4),
            (10, 9),
            (4, 3),
            (9, 8),
            (3, 2),
            (8, 7),
            (13, 12),
            (14, 12),
            (21, 19),
            (22, 19),
            (19, 18),
            (29, 27),
            (30, 27),
            (27, 26),
            (18, 17),
            (26, 25),
            (17, 13),
            (25, 13),
            (14, 13),
            (15, 14),
        ]

        self.use_joint = {
            2: 0,
            3: 1,
            4: 2,
            5: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7,
            12: 8,
            13: 9,
            14: 10,
            15: 11,
            17: 12,
            18: 13,
            19: 14,
            21: 15,
            22: 16,
            25: 17,
            26: 18,
            27: 19,
            29: 20,
            30: 21,
        }

        self.bone_pair_filter = []
        for bp in self.bone_pair:
            self.bone_pair_filter.append([self.use_joint[bp[0]], self.use_joint[bp[1]]])

    def get_joint_flatten_adjacent(self, dim=3):
        num_joint = len(self.use_joint)
        joint_adj = np.eye(num_joint * dim)
        for bp in self.bone_pair_filter:
            for i in range(dim):
                for j in range(dim):
                    # connection of different joint
                    joint_adj[bp[0] * dim + i, bp[1] * dim + j] = 1
                    joint_adj[bp[1] * dim + i, bp[0] * dim + j] = 1
                    joint_adj[bp[0] * dim + j, bp[1] * dim + i] = 1
                    joint_adj[bp[1] * dim + j, bp[0] * dim + i] = 1
                    # connection of same joint
                    joint_adj[bp[0] * dim + i, bp[0] * dim + j] = 1
                    joint_adj[bp[1] * dim + i, bp[1] * dim + j] = 1
        # return normalize_digraph(joint_adj)
        return joint_adj

    def get_bone_flattens_adjacent(self, dim=3):
        # get bone adjacency matrix from bone
        num_bone = len(self.bone_pair_filter)
        bone_adj = np.eye(num_bone * dim)
        for ib in range(num_bone):
            for jb in range(ib, num_bone):
                if (len(set(self.bone_pair_filter[ib]) & set(self.bone_pair_filter[jb])) > 0):
                    for i in range(dim):
                        for j in range(dim):
                            # connection of different joint
                            bone_adj[ib * dim + i, jb * dim + j] = 1
                            bone_adj[jb * dim + i, ib * dim + j] = 1
                            bone_adj[ib * dim + j, jb * dim + i] = 1
                            bone_adj[jb * dim + j, ib * dim + i] = 1
                            # connection of same joint
                            bone_adj[ib * dim + i, ib * dim + j] = 1
                            bone_adj[jb * dim + i, jb * dim + j] = 1
        # return normalize_digraph(bone_adj)
        return bone_adj

    def get_transition(self):
        # first calculate bone information using bone pair
        # then filter joints with use joints
        num_joint = 22
        num_bone = len(self.bone_pair_filter)
        bone_pair = np.array(self.bone_pair_filter)  # (num_bone, 2)
        data = np.zeros((num_joint * 3, num_bone * 3))
        for i in range(num_bone):
            data[bone_pair[i, 0] * 3, i * 3] = 1
            data[bone_pair[i, 1] * 3, i * 3] = 1
            data[bone_pair[i, 0] * 3 + 1, i * 3 + 1] = 1
            data[bone_pair[i, 1] * 3 + 1, i * 3 + 1] = 1
            data[bone_pair[i, 0] * 3 + 2, i * 3 + 2] = 1
            data[bone_pair[i, 1] * 3 + 2, i * 3 + 2] = 1
        return data

    def get_joint_adjacent(self):
        num_joint = len(self.use_joint)
        joint_adj = np.eye(num_joint)
        for bp in self.bone_pair_filter:
            # connection of different joint
            joint_adj[bp[0], bp[1]] = 1
            joint_adj[bp[1], bp[0]] = 1
        # return normalize_digraph(joint_adj)
        return joint_adj

    def get_bone_adjacent(self):
        # get bone adjacency matrix from bone
        num_bone = len(self.bone_pair_filter)
        bone_adj = np.eye(num_bone)
        for ib in range(num_bone):
            for jb in range(ib, num_bone):
                if (len(set(self.bone_pair_filter[ib]) & set(self.bone_pair_filter[jb])) > 0):
                    bone_adj[ib, jb] = 1
        # return normalize_digraph(bone_adj)
        return bone_adj


def create_transition_matrix_h36m():
    """
    create transition matrix from joint to bone
    """
    # TODO: modified into configuration files
    bone_pair = [
        (5, 4),
        (10, 9),
        (4, 3),
        (9, 8),
        (3, 2),
        (8, 7),
        (13, 12),
        (14, 12),
        (21, 19),
        (22, 19),
        (19, 18),
        (29, 27),
        (30, 27),
        (27, 26),
        (18, 17),
        (26, 25),
        (17, 13),
        (25, 13),
        (14, 13),
        (15, 14),
    ]
    use_joint = {
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        7: 4,
        8: 5,
        9: 6,
        10: 7,
        12: 8,
        13: 9,
        14: 10,
        15: 11,
        17: 12,
        18: 13,
        19: 14,
        21: 15,
        22: 16,
        25: 17,
        26: 18,
        27: 19,
        29: 20,
        30: 21,
    }
    # first calculate bone information using bone pair
    # then filter joints with use joints
    num_joint = 22
    num_bone = len(bone_pair)
    bone_pair = np.array(bone_pair)  # (num_bone, 2)
    bone_pair = np.concatenate((bone_pair, np.expand_dims(np.arange(num_bone), 1)), 1)  # (num_bone, 3)

    data = np.zeros((num_joint * 3, num_bone * 3))
    for i in range(num_bone):
        data[use_joint[bone_pair[i, 0]] * 3, bone_pair[i, -1] * 3] = 1
        data[use_joint[bone_pair[i, 1]] * 3, bone_pair[i, -1] * 3] = 1
        data[use_joint[bone_pair[i, 0]] * 3 + 1, bone_pair[i, -1] * 3 + 1] = 1
        data[use_joint[bone_pair[i, 1]] * 3 + 1, bone_pair[i, -1] * 3 + 1] = 1
        data[use_joint[bone_pair[i, 0]] * 3 + 2, bone_pair[i, -1] * 3 + 2] = 1
        data[use_joint[bone_pair[i, 1]] * 3 + 2, bone_pair[i, -1] * 3 + 2] = 1
    return data


def create_edge_adj(bone_pair):
    """
    create an edge adjacency matrix from vertex adjacency matrix
    """
    num_bone = len(bone_pair)
    edge_adj = np.zeros((num_bone, num_bone))
    for i in range(num_bone):
        for j in range(i, num_bone):
            if len(set(bone_pair[i]) & bone_pair[j]) > 0:
                edge_adj[i, j] = 1
            else:
                edge_adj[i, j] = 0
        edge_adj[i, i] = 1
    return edge_adj


class TimeTransform:

    def __init__(self, seq_len, dct_used):
        self.dct_m, self.idct_m = get_dct_matrix(seq_len)
        self.dct_used = dct_used
        self.seq_len = seq_len

    def transform(self, data):
        # in - (n, seq_len, dim)
        # out - (n, dct_n, dim)
        if isinstance(data, torch.Tensor):
            return self.transform_torch(data)
        elif isinstance(data, np.ndarray):
            return self.transform_np(data)
        else:
            raise NotImplementedError("Invalid data type for data")

    def inverse(self, data):
        if isinstance(data, torch.Tensor):
            return self.inverse_torch(data)
        elif isinstance(data, np.ndarray):
            return self.inverse_np(data)
        else:
            raise NotImplementedError("Invalid data type for data")

    def transform_torch(self, data):
        dct_m = Variable(torch.from_numpy(self.dct_m)).float().to(data.device)
        n_instance, n_seq, n_dim = data.shape
        assert n_seq == self.seq_len
        data = data.permute(0, 2, 1).contiguous()
        data = data.view(-1, n_seq)
        data = data.permute(1, 0).contiguous()
        data = torch.matmul(dct_m[0:self.dct_used, :], data)
        data = data.permute(1, 0).contiguous().reshape((n_instance, n_dim, self.dct_used))
        data = data.permute(0, 2, 1).contiguous()
        return data

    def transform_np(self, data):
        n_instance, n_seq, n_dim = data.shape
        assert n_seq == self.seq_len
        data = data.transpose(0, 2, 1)
        data = data.reshape(-1, n_seq)
        data = data.transpose()
        data = np.matmul(self.dct_m[0:self.dct_used, :], data)
        data = data.transpose().reshape((n_instance, n_dim, self.dct_used))
        data = data.transpose(0, 2, 1)
        return data

    def inverse_torch(self, data):
        # data - (n, n_dct, n_dim)
        n_instance = data.shape[0]
        idct_m = Variable(torch.from_numpy(self.idct_m)).float().to(data.device)
        # (n, n_dct, n_dim) -> (n, n_dim, n_dct)
        data = data.permute(0, 2, 1).contiguous()
        data = data.view(-1, self.dct_used)
        data = data.permute(1, 0).contiguous()
        data = torch.matmul(idct_m[:, 0:self.dct_used], data)
        data = data.permute(1, 0).contiguous()
        data = data.view(n_instance, -1, self.seq_len)
        data = data.permute(0, 2, 1).contiguous()
        return data

    def inverse_np(self, data):
        n_instance = data.shape[0]
        data = data.transpose(0, 2, 1).resshape(-1, self.dct_used)
        data = data.transpose()
        data = np.matmul(self.idct_m[:, 0:self.dct_used], data)
        data = data.transpose(0, 1)
        data = data.reshape(n_instance, -1, self.seq_len)
        data = data.transpose(0, 2, 1)
        return data


class MinMaxNorm:

    def __init__(self, v_min, v_max):
        self.set_range(v_min, v_max)

    def transform(self, data):
        data = (data - self._min) / self._gap
        data = data * 2 - 1
        return data

    def inverse(self, data):
        data = (data + 1) / 2
        data = data * self._gap + self._min
        return data

    def set_range(self, v_min, v_max):
        self._min = v_min
        self._max = v_max
        self._gap = v_max - v_min


class MeanStdNorm:

    def __init__(self, mean, std):
        self._mean = mean[np.newaxis, np.newaxis, :]
        self._std = std[np.newaxis, np.newaxis, :]

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            m = torch.Tensor(self._mean).to(data.dtype).to(data.device)
            s = torch.Tensor(self._std).to(data.dtype).to(data.device)
        else:
            m = self._mean
            s = self._std

        data = (data - m) / s
        return data

    def inverse(self, data):
        if isinstance(data, torch.Tensor):
            m = torch.Tensor(self._mean).to(data.dtype).to(data.device)
            s = torch.Tensor(self._std).to(data.dtype).to(data.device)
        else:
            m = self._mean
            s = self._std

        data = data * s + m
        return data


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
    Args
    completeData: nx99 matrix with data to normalize
    Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
    """
    n, t, vc = completeData.shape
    completeData = completeData.reshape(n * t, vc)
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0
    pdb.set_trace()
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


"""
utils functions for expi dataset
"""


def normExPI_xoz(img, P0, P1, P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: olane xoz

    X0 = P0
    X1 = (P1 - P0) / np.linalg.norm((P1 - P0)) + P0  #x
    X2 = (P2 - P0) / np.linalg.norm((P2 - P0)) + P0
    X3 = np.cross(X2 - P0, X1 - P0) + P0  #y
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1 - P0, X3 - P0) + P0  #z

    X = np.concatenate((np.array([X0, X1, X2, X3]).transpose(), np.array([[1, 1, 1, 1]])), axis=0)
    Q = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]).transpose()
    M = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp, np.array([1])), axis=0)
        img_norm[i] = M.dot(tmp)
    return img_norm


def normExPI_2p_by_frame(seq):
    nb, dim = seq.shape  # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1, 3))  #36
        P0 = (img[10] + img[11]) / 2  # left and right hip
        P1 = img[11]  # right hip
        P2 = img[3]  # back
        img_norm = normExPI_xoz(img, P0, P1, P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm


def unnorm_abs2Indep(seq):
    # in:  torch.size(bz, nb_frames, 36, 3)
    # out: torch.size(bz, nb_frames, 36, 3)
    seq = seq.detach().cpu().numpy()
    bz, frame, nb, dim = seq.shape
    seq_norm = seq
    for j in range(bz):
        for i in range(frame):
            img = seq[j][i]
            P0_m = (img[10] + img[11]) / 2
            P1_m = img[11]
            P2_m = img[3]
            if nb == 36:
                img_norm_m = normExPI_xoz(img[:int(nb / 2)], P0_m, P1_m, P2_m)
                P0_f = (img[18 + 10] + img[18 + 11]) / 2
                P1_f = img[18 + 11]
                P2_f = img[18 + 3]
                img_norm_f = normExPI_xoz(img[int(nb / 2):], P0_f, P1_f, P2_f)
                img_norm = np.concatenate((img_norm_m, img_norm_f))
            elif nb == 18:
                img_norm = normExPI_xoz(img, P0_m, P1_m, P2_m)
            seq_norm[j][i] = img_norm.reshape((nb, dim))
    seq = torch.from_numpy(seq_norm).cuda()
    return seq


def normNTURGBD_xoz(img, P0, P1, P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: olane xoz
    # 程序的含义是以两个hip的中点作为中心点，中心点到hip的距离作为x向量，然后求解xoy向量的坐标
    eps = 1e-10
    X0 = P0
    X1 = (P1 - P0) / (np.linalg.norm((P1 - P0)) + eps) + P0  #x
    X2 = (P2 - P0) / (np.linalg.norm((P2 - P0)) + eps) + P0
    X3 = np.cross(X2 - P0, X1 - P0) + P0  #y
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1 - P0, X3 - P0) + P0  #z

    X = np.concatenate((np.array([X0, X1, X2, X3]).transpose(), np.array([[1, 1, 1, 1]])), axis=0)
    Q = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]).transpose()
    try:
        M = Q.dot(np.linalg.pinv(X))
    except np.linalg.LinAlgError:
        pdb.set_trace()

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp, np.array([1])), axis=0)
        img_norm[i] = M.dot(tmp)
    return img_norm


def normNTURGBD_2p_by_frame(seq):
    # assert that there is not all zero frame in the input
    nb, dim = seq.shape  # nb_frames, dim=150
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1, 3))  # 50
        img = img - img[0:1]
        # P0 = (img[12] + img[16]) / 2  # left and right hip
        P0 = img[0]  # pelvis
        P1 = img[16]  # right hip
        P2 = img[20]  # back
        img_norm = normNTURGBD_xoz(img, P0, P1, P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm


def unnormNTURGBD_abs2Indep(seq):
    # in:  torch.size(bz, nb_frames, 36, 3)
    # out: torch.size(bz, nb_frames, 36, 3)
    seq = seq.detach().cpu().numpy()
    bz, frame, nb, dim = seq.shape
    seq_norm = seq
    for j in range(bz):
        for i in range(frame):
            img = seq[j][i]

            P0_m = img[0]  # pelvis
            # P0_m = (img[12] + img[16]) / 2
            P1_m = img[16]  # right hip
            P2_m = img[20]  # back
            if nb == 50:
                img_norm_m = normNTURGBD_xoz(img[:int(nb / 2)], P0_m, P1_m, P2_m)
                P0_f = (img[25 + 12] + img[25 + 16]) / 2
                # P0_f = img[25]
                P1_f = img[25 + 16]
                P2_f = img[25 + 20]
                img_norm_f = normNTURGBD_xoz(img[int(nb / 2):], P0_f, P1_f, P2_f)
                img_norm = np.concatenate((img_norm_m, img_norm_f))
            elif nb == 25:
                img_norm = normNTURGBD_xoz(img, P0_m, P1_m, P2_m)
            seq_norm[j][i] = img_norm.reshape((nb, dim))
    seq = torch.from_numpy(seq_norm).cuda()
    return seq


def normNTURGBD_pelvis(seq):
    nb, dim = seq.shape  # nb_frames, dim=150
    seq = seq.reshape((nb, dim // 3, 3))
    seq_norm = seq - seq[0:1, 0:1]
    return seq_norm


def filter_frames(seq):
    non_zero_ids = []
    nb = seq.shape[0]
    for i in range(nb):
        if not np.all(seq[i] == 0):
            non_zero_ids.append(i)
    return seq[non_zero_ids]


if __name__ == "__main__":
    r = np.random.rand(2, 3) * 10
    # r = np.array([[0.4, 1.5, -0.0], [0, 0, 1.4]])
    r1 = r[0]
    R1 = expmap2rotmat(r1)
    q1 = rotmat2quat(R1)
    # R1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    e1 = rotmat2euler(R1)

    r2 = r[1]
    R2 = expmap2rotmat(r2)
    q2 = rotmat2quat(R2)
    # R2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    e2 = rotmat2euler(R2)

    r = Variable(torch.from_numpy(r)).cuda().float()
    # q = expmap2quat_torch(r)
    R = expmap2rotmat_torch(r)
    q = rotmat2quat_torch(R)
    # R = Variable(torch.from_numpy(
    #     np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]]))).cuda().float()
    eul = rotmat2euler_torch(R)
    eul = eul.cpu().data.numpy()
    R = R.cpu().data.numpy()
    q = q.cpu().data.numpy()

    if np.max(np.abs(eul[0] - e1)) < 0.000001:
        print("e1 clear")
    else:
        print("e1 error {}".format(np.max(np.abs(eul[0] - e1))))
    if np.max(np.abs(eul[1] - e2)) < 0.000001:
        print("e2 clear")
    else:
        print("e2 error {}".format(np.max(np.abs(eul[1] - e2))))

    if np.max(np.abs(R[0] - R1)) < 0.000001:
        print("R1 clear")
    else:
        print("R1 error {}".format(np.max(np.abs(R[0] - R1))))

    if np.max(np.abs(R[1] - R2)) < 0.000001:
        print("R2 clear")
    else:
        print("R2 error {}".format(np.max(np.abs(R[1] - R2))))

    if np.max(np.abs(q[0] - q1)) < 0.000001:
        print("q1 clear")
    else:
        print("q1 error {}".format(np.max(np.abs(q[0] - q1))))
