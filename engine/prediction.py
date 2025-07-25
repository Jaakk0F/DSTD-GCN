# encoding: utf-8
"""
@author:  Jiajun Fu
@contact: Jaakk0F@foxmail.com
"""

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils.loss import (AccumLoss, bone_error_3d, gram_matrix, mae_error_3d, mpjpe_error_3d, mse_error_3d,
                         transition_error_3d)
from .utils.transform import (cst_inverse, cst_transform, st_inverse, st_transform, tsc_inverse, tsc_transform,
                              tscr_3dpw_inverse, tscr_3dpw_transform, tscr_cmu_inverse, tscr_cmu_transform,
                              tscr_h36m_inverse, tscr_h36m_transform)


# wrapper class for model
class ModelWrapper(nn.Module):

    def __init__(self, model, loss, n_out=1):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.n_out = n_out
        # init loss
        loss_dict = dict(jl2=mpjpe_error_3d,
                         bl2=bone_error_3d,
                         tl2=transition_error_3d,
                         cl1=mae_error_3d,
                         cl2=mse_error_3d,
                         gm2=gram_matrix)

        self.all_loss = set()

        if n_out > 1:
            self.loss_funcs = [{} for _ in range(n_out)]
            for ls in loss.keys():
                ls_list = loss[ls]
                self.loss_funcs[ls_list[-1]][ls] = (loss_dict[ls_list[0]], ls_list[1])
                if ls in self.all_loss:
                    raise ValueError("Redundant Error", ls)
                self.all_loss.add(ls)
        else:
            self.loss_funcs = {}
            for ls in loss.keys():
                self.loss_funcs[ls] = (loss_dict[loss[ls][0]], loss[ls][1])
                if ls in self.all_loss:
                    raise ValueError("Redundant Error", ls)
                self.all_loss.add(ls)

    def forward(self, inputs, training=True):
        outputs = self.model(inputs)
        if training:
            return outputs
        return outputs[-1] if isinstance(outputs, list) else outputs

    def calc_loss(self, pred, gt, loss_type="all", wgts=None):
        # return a list of loss if all, else single value
        # TODO: more sophisticated version of loss and output binding when there
        # TODO: are multiple losses and outputs
        if loss_type != "all" and loss_type != "sum" and loss_type not in self.all_loss:
            raise ValueError(f"Invalid loss type {loss_type}")
        if isinstance(pred, list):  # multiple inputs
            assert len(pred) == self.n_out
            if loss_type == "all":
                loss = {}
                for i in range(self.n_out):
                    for ls in self.loss_funcs[i]:
                        loss_func, weight = self.loss_funcs[i][ls]
                        if isinstance(gt, list):
                            loss[ls] = weight * loss_func(pred[i], gt[i], wgts)
                        else:
                            loss[ls] = weight * loss_func(pred[i], gt, wgts)
            elif loss_type == "sum":
                loss = 0
                for i in range(self.n_out):
                    for ls in self.loss_funcs[i]:
                        loss_func, weight = self.loss_funcs[i][ls]
                        if isinstance(gt, list):
                            loss += weight * loss_func(pred[i], gt[i], wgts)
                        else:
                            loss += weight * loss_func(pred[i], gt, wgts)
            else:  # multi out not support
                pass
        else:  # single outputs
            if loss_type == "all":
                loss = {}
                for ls in self.loss_funcs:
                    loss_func, weight = self.loss_funcs[ls]
                    loss[ls] = weight * loss_func(pred, gt, wgts)
            elif loss_type == "sum":
                loss = 0
                for ls in self.loss_funcs:
                    loss_func, weight = self.loss_funcs[ls]
                    loss += weight * loss_func(pred, gt, wgts)
            else:
                loss = self.loss_funcs[loss_type]
        return loss


class PredictionEngine:

    def __init__(self, config, model, logger):
        self.model = ModelWrapper(model, config["loss"], config["n_out"])
        self.config = config
        self.logger = logger
        self.reset()  # set parameters in the experiment

        # init transform
        transform_dict = dict(st=(st_transform, st_inverse),
                              tsc=(tsc_transform, tsc_inverse),
                              tscr_h36m=(tscr_h36m_transform, tscr_h36m_inverse),
                              tscr_cmu=(tscr_cmu_transform, tscr_cmu_inverse),
                              tscr_3dpw=(tscr_3dpw_transform, tscr_3dpw_inverse),
                              cst=(cst_transform, cst_inverse),
                              no=(None, None))

        logger.info("Trainable number of parameters of the network is: " +
                    str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        logger.info("Total number of parameters of the network is: " + str(sum(p.numel() for p in model.parameters())))

        self.transform_func, self.inverse_func = transform_dict[config["transform"]]

    def transform(self, x):
        if self.transform_func is None:
            return x
        # wrapper function
        if isinstance(x, list):
            x_ret = []
            for x_sub in x:
                x_ret.append(self.transform_func(x_sub))
            x_ret = list(x_ret)
        else:
            x_ret = self.transform_func(x)
        return x_ret

    def inverse(self, x):
        if self.inverse_func is None:
            return x
        # wrapper function
        if isinstance(x, list):
            x_ret = []
            for x_sub in x:
                x_ret.append(self.inverse_func(x_sub))
            x_ret = list(x_ret)
        else:
            x_ret = self.inverse_func(x)
        return x_ret

    def reset(self):
        self.lr = self.config["learn"]["lr"]
        self.best_err = float("inf")
        self.optimizer, self.scheduler = self._setup_learn(self.model.parameters(), self.config["learn"]["opt"])

    def recover(self, checkpoint_path, model_only=False):
        # recover from dict
        state = torch.load(checkpoint_path)
        err = state["err"]
        epoch = state["epoch"]
        if not model_only:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr = state["lr"]
        self.logger.info("load from lr {}, curr_avg {} from {}.".format(state["lr"], err, checkpoint_path))
        return epoch, err

    def save(self, checkpoint_path, err, epoch, is_best=False):
        state = {
            "lr": self.lr,
            "err": err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, checkpoint_path + "/last.pth")
        if is_best:
            torch.save(state, checkpoint_path + "/best.pth")

    def _setup_learn(self, params, opt_type="adam"):
        """ Setup optimizer and learning rate scheduler
        """
        if opt_type == "adam":
            optimizer = optim.Adam(
                params,
                lr=self.config["learn"]["lr"],
                weight_decay=self.config["learn"]["weight_decay"],
            )
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.config["learn"]["step_size"],
                                                  gamma=self.config["learn"]["gamma"])
            return optimizer, scheduler

    def train(
        self,
        train_loader,
        epoch,
        time_tsfm=None,
        scale_tsfm=None,
        weights=None,
        max_iter=-1,
    ):
        # if (epoch + 1) % self.config["lr_decay"] == 0:
        #     self.lr = lr_decay(self.optimizer, self.lr,
        #                        self.config["lr_gamma"])
        t_l = {key_loss: AccumLoss() for key_loss in self.config["loss"]}
        self.model.train()
        
        num_iter = len(train_loader) if max_iter == -1 else min(len(train_loader), max_iter)
        loop = tqdm(enumerate(train_loader), total=num_iter, leave=True)
        for i, (inputs, inputs_inv, targets, all_seqs) in loop:
            # cpu to gpu
            if isinstance(inputs, list):
                inputs = [input.float().cuda(non_blocking=True) for input in inputs]
                inputs_inv = [input_inv.float().cuda(non_blocking=True) for input_inv in inputs_inv]
                targets = [target.float().cuda(non_blocking=True) for target in targets]
                N = inputs[0].shape[0]
            else:
                inputs = inputs.float().cuda(non_blocking=True)
                inputs_inv = inputs_inv.float().cuda(non_blocking=True)
                targets = targets.float().cuda(non_blocking=True)
                N = inputs.shape[0]

            # feed network
            if time_tsfm is not None:
                inputs = time_tsfm.transform(inputs)
            inputs = self.transform(inputs)
            outputs = self.model(inputs, False)

            # count time
            # torch.cuda.synchronize()  #增加同步操作
            # start = time.time()
            # outputs = self.model(inputs, False)
            # torch.cuda.synchronize()  #增加同步操作
            # end = time.time()
            # print("Total Time", end - start)
            # pdb.set_trace()

            outputs = self.inverse(outputs)

            # result transformation
            if scale_tsfm is not None:
                outputs = scale_tsfm.inverse(outputs)
            if time_tsfm is not None:
                outputs = time_tsfm.inverse(outputs)

            # calculate loss and backward
            # only calculate final loss
            if outputs.shape[1] != targets.shape[1]:
                t = outputs.shape[1]
                targets_l = targets[:, -t:]
            else:  # calculate all loss
                targets_l = targets
            loss = self.model.calc_loss(outputs, targets_l, "all", weights)
            all_loss = 0

            # update loss
            for ls in loss:
                all_loss += loss[ls]
                t_l[ls].update(loss[ls].item() * N, N)

            # inverse input training
            if self.config["inverse"]:
                if time_tsfm is not None:
                    inputs_inv = time_tsfm.transform(inputs_inv)
                inputs_inv = self.transform(inputs_inv)
                outputs_inv = self.model(inputs_inv, True)
                outputs_inv = self.inverse(outputs_inv)
                inv_idx = torch.arange(targets.shape[1] - 1, -1, -1).long()
                targets_inv = targets[:, inv_idx,].contiguous()
                if outputs_inv.shape[1] != targets_inv.shape[1]:
                    t = outputs_inv.shape[1]
                    targets_il = targets_inv[:, -t:]
                else:
                    targets_il = targets_inv
                if time_tsfm is not None:
                    outputs_inv = time_tsfm.inverse(outputs_inv)
                if scale_tsfm is not None:
                    outputs_inv = scale_tsfm.inverse(outputs_inv)
                loss_inv = self.model.calc_loss(outputs_inv, targets_il, "all", weights)
                for ls in loss_inv:
                    all_loss += loss_inv[ls]
                all_loss /= 2

            # update parameters
            self.optimizer.zero_grad()
            all_loss.backward()
            if self.config.get("clip", -1) > 0:
                nn.utils.clip_grad_norm(self.model.model.parameters(), max_norm=self.config["clip"])
            self.optimizer.step()

            # show each loss
            desc = f"epoch: {epoch + 1}|[{i + 1}/{num_iter}]|train|"
            for ls in loss:
                desc += "{}:{:.2f}|".format(ls, t_l[ls].avg)
            loop.set_description(desc)
            
            # break
            if i >= num_iter - 1:
                break
            
        # update log with the last description
        self.logger.info(desc)

        # update lr (not use for the given input)
        self.scheduler.step()
        self.lr = self.scheduler.get_lr()[0]

        # calcalte average loss
        all_loss = 0
        for ls in t_l:
            all_loss += t_l[ls].avg
        return all_loss

    def test(
        self,
        test_loader,
        input_n=10,
        eval_frame=None,
        dim_used=None,
        joint_to_ignore=None,
        joint_equal=None,
        time_tsfm=None,
        scale_tsfm=None,
        action=None,
        save_path=None,
    ):
        N = 0
        t_l = AccumLoss()
        # write into the configuration file
        assert eval_frame is not None
        t_metric = np.zeros(len(eval_frame))

        save_results = dict() if save_path is not None else None

        self.model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
            for i, (inputs, _, _, all_seqs) in loop:
                # cpu to gpu
                if isinstance(inputs, list):
                    inputs = [input.float().cuda(non_blocking=True) for input in inputs]
                else:
                    inputs = inputs.float().cuda(non_blocking=True)
                all_seqs = all_seqs.float().cuda(non_blocking=True)

                inputs = self.transform(inputs)
                outputs = self.model(inputs, False)
                outputs = self.inverse(outputs)

                if isinstance(outputs, list):  # skip other result
                    outputs = outputs[0]
                    n, t = outputs.shape[:2]
                    outputs = outputs.view(n, t, -1)

                # all_seqs are alrealdy renormed
                if scale_tsfm is not None:
                    outputs = scale_tsfm.inverse(outputs)
                if time_tsfm is not None:
                    outputs = time_tsfm.inverse(outputs)

                n, seq_len, _ = all_seqs.shape

                outputs_3d = outputs
                pred_3d = all_seqs.clone()

                # calculate mpjpe
                # filter joint
                if not np.any(dim_used == None):
                    dim_used = np.array(dim_used)
                    if outputs.shape[1] != all_seqs.shape[1]:
                        pred_3d[:, input_n:, dim_used] = outputs_3d
                    else:
                        pred_3d[:, :, dim_used] = outputs_3d
                else:
                    if outputs.shape[1] != all_seqs.shape[1]:
                        pred_3d[:, input_n:] = outputs_3d
                    else:
                        pred_3d[:, :, :] = outputs_3d
                if not np.any(joint_to_ignore == None):
                    assert joint_to_ignore.shape == joint_equal.shape
                    index_to_ignore = np.concatenate(
                        (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
                    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
                    pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                pred_p3d = pred_3d.contiguous()
                pred_p3d = pred_p3d.view(n, seq_len, -1, 3)[:, input_n:, :, :]
                targ_p3d = all_seqs.contiguous()
                targ_p3d = targ_p3d.view(n, seq_len, -1, 3)[:, input_n:, :, :]
                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k]
                    metric_k = (torch.mean(
                        torch.norm(
                            targ_p3d[:, j, :, :].contiguous().view(-1, 3) -
                            pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                            2,
                            1,
                        )).item() * n)
                    t_metric[k] += metric_k
                    t_l.update(metric_k, n)
                N += n
                if action is None:
                    action = "NA"
                desc = f"action: {action}|[{i + 1}/{len(test_loader)}]|test|"
                desc += "loss:{:.2f}".format(t_l.avg)
                loop.set_description(desc)

                # save_result
                if save_results is not None:
                    pred_append = pred_p3d.cpu().numpy()
                    pred = save_results.get("result",
                                            np.array([], pred_append.dtype).reshape((0,) + pred_append.shape[1:]))
                    pred = np.concatenate((pred, pred_append), axis=0)
                    save_results["result"] = pred
                    targ_append = targ_p3d.cpu().numpy()
                    targ = save_results.get("target",
                                            np.array([], targ_append.dtype).reshape((0,) + targ_append.shape[1:]))
                    targ = np.concatenate((targ, targ_append), axis=0)
                    save_results["target"] = targ

            self.logger.info(desc)
            t_metric /= N

            if save_results is not None:
                np.savez(save_path + ".npz", target=save_results["target"], result=save_results["result"])
        return t_l.avg, t_metric
