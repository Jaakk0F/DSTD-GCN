import numpy as np
import torch

#  NOTE: all the loss function are implemented in pytorch


class AccumLoss(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_his = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val_his.append(val)
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def mae_error_3d(outputs, targets, joint_weights=None):
    if joint_weights is not None:
        body_weights = torch.from_numpy(joint_weights[np.newaxis, :,
                                                      np.newaxis])
    else:
        body_weights = torch.ones((1, outputs.shape[-1] // 3, 1))
    body_weights = body_weights.to(outputs.device)
    n, t, vc = outputs.shape
    pred_3d = outputs.contiguous().view(n * t, vc // 3, 3) * body_weights
    targ_3d = targets.contiguous().view(n * t, vc // 3, 3) * body_weights
    mean_3d_err = torch.mean(torch.abs(pred_3d - targ_3d))
    return mean_3d_err


def mse_error_3d(outputs, targets, joint_weights=None):
    if joint_weights is not None:
        body_weights = torch.from_numpy(joint_weights[np.newaxis, :,
                                                      np.newaxis])
    else:
        body_weights = torch.ones((1, outputs.shape[-1] // 3, 1))
    body_weights = body_weights.to(outputs.device)
    n, t, vc = outputs.shape
    pred_3d = outputs.contiguous().view(n * t, vc // 3, 3) * body_weights
    targ_3d = targets.contiguous().view(n * t, vc // 3, 3) * body_weights
    mean_3d_err = torch.mean(torch.sqrt((pred_3d - targ_3d)**2))
    return mean_3d_err


def mpjpe_error_3d(outputs, targets, joint_weights=None):
    if joint_weights is not None:
        body_weights = torch.from_numpy(joint_weights[np.newaxis, :,
                                                      np.newaxis])
    else:
        body_weights = torch.ones((1, outputs.shape[-1] // 3, 1))
    body_weights = body_weights.to(outputs.device)
    n, t, vc = outputs.shape
    pred_3d = outputs.contiguous().view(n * t, vc // 3, 3) * body_weights
    targ_3d = targets.contiguous().view(n * t, vc // 3, 3) * body_weights
    pred_3d, targ_3d = pred_3d.view(-1, 3), targ_3d.view(-1, 3)
    mean_3d_err = torch.mean(
        torch.norm(pred_3d - targ_3d, 2, 1) * body_weights)
    return mean_3d_err


def gram_matrix(outputs, targets, joint_weights=None):
    # ignore joint weights
    n, t, vc = outputs.shape
    pred = torch.cat((outputs[:, 1:], outputs[:, :-1]), dim=-1)
    targ = torch.cat((targets[:, 1:], targets[:, :-1]), dim=-1)
    pred = pred.div(n * 2 * t * vc)
    targ = targ.div(n * 2 * t * vc)
    pred = torch.matmul(pred, pred.transpose(1, 2).contiguous())
    targ = torch.matmul(pred, pred.transpose(1, 2).contiguous())
    mean_gram_loss = torch.sum((pred - targ)**2)
    return mean_gram_loss


def bone_error_3d(outputs, targets, weights=None):
    # calculate bone length
    pred_3d = calc_bone_length(outputs)
    targ_3d = calc_bone_length(targets)
    # L2 loss calculation
    mean_3d_err = torch.mean((pred_3d - targ_3d)**2)
    return mean_3d_err


# bone length calculation in h36 dataset
def calc_bone_length(data, dim=3):
    B, T, VC = data.shape
    assert VC % dim == 0
    data = data.view(B, T, VC // dim, dim)

    # inter_loss = 0.
    Ct = torch.cuda.FloatTensor(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
    Ct = Ct.to(data.device)

    data = data.transpose(2, 3)  # b, t, 3, 22
    data = torch.matmul(data, Ct)  # b, t, 3, 21
    bone_length = torch.norm(data, 2, 2)  # b, t, 21

    return bone_length


def transition_error_3d(outputs, targets, joint_weights=None):
    if joint_weights is not None:
        body_weights = torch.from_numpy(joint_weights[np.newaxis, :,
                                                      np.newaxis])
    else:
        body_weights = torch.ones((1, outputs.shape[-1] // 3, 1))
    body_weights = body_weights.to(outputs.device)
    pred_3d = outputs[:, 1:, :] - outputs[:, :-1, :]
    targ_3d = targets[:, 1:, :] - targets[:, :-1, :]
    n, t, vc = pred_3d.shape
    pred_3d = pred_3d.contiguous().view(n * t, vc // 3, 3) * body_weights
    targ_3d = targ_3d.contiguous().view(n * t, vc // 3, 3) * body_weights
    pred_3d, targ_3d = pred_3d.view(-1, 3), targ_3d.view(-1, 3)

    # L2 loss function
    mean_3d_err = torch.mean(
        torch.norm(pred_3d - targ_3d, 2, 1) * body_weights)
    return mean_3d_err
