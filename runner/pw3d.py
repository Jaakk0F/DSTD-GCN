import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from dataset import get_dataset
from torch.utils.data import DataLoader
from utils import Visualizer

from .base import BaseRunner


class PW3DRunner(BaseRunner):

    def __init__(self, config):
        super().__init__(config)

    def save_files(self):
        # make directories
        for path in self.config["save"]["path"]:
            if path != "base":
                update_path = os.path.join(self.config["save"]["path"]["base"], self.config["save"]["path"][path])
                self.config["save"]["path"][path] = update_path
                os.makedirs(update_path, exist_ok=True)

        # save file
        for file in self.config["save"]["files"]:
            shutil.copy(file, self.config["save"]["path"]["files"])

    def run_train(self):
        self.logger.info("Start training")
        # dataset
        # update config files
        train_data_config = self.config["dataset"]["train"]
        dataset_name = self.config["dataset"]["name"]
        train_dataset = get_dataset(dataset_name, **train_data_config)
        # print("train data shape {}".format(train_dataset.all_seqs.shape[0]))
        self.logger.info("train data shape {}".format(len(train_dataset)))
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.config.train_batch_size,
                                  shuffle=True,
                                  num_workers=self.config.num_workers,
                                  pin_memory=True)
        test_data_config = self.config["dataset"]["test"]
        # manually set the scaler in test dataset
        test_dataset = get_dataset(dataset_name, **test_data_config)
        self.logger.info("test data shape {}".format(len(test_dataset)))
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.config.test_batch_size,
                                 shuffle=False,
                                 num_workers=self.config.num_workers,
                                 pin_memory=True)

        # load model
        if self.config["model"]["load"]:
            start_epoch, err_best = self.engine.recover(self.config["model"]["ckpt"])
        else:
            start_epoch, err_best = 0, 1e10

        ret_log_best = None
        # start training
        for epoch in range(start_epoch, self.config["epoch"]):

            # print("==========================")
            # print(">>> epoch: {} | lr: {:.5f}".format(epoch + 1,
            #                                           self.engine.lr))
            self.logger.info("==========================")
            self.logger.info(">>> epoch: {} | lr: {:.5f}".format(epoch + 1, self.engine.lr))

            # writing result to csv
            ret_log_train = np.array([epoch + 1])
            head_train = np.array(["epoch"])

            # training for each epoch
            train_loss = self.engine.train(
                train_loader,
                epoch,
                train_dataset.time_tsfm,
                None,
                None,
                self.config["engine"]["max_iter"] 
            )
            ret_log_train = np.append(ret_log_train, [self.engine.lr, train_loss])
            head_train = np.append(head_train, ["lr", "train_loss"])

            # testing after each epoch
            test_err_avg = 0
            # from DSTDGCN
            # ret_log_test = np.array(np.zeros(11))  # all, per
            # head_test = np.array([
            #     "test_loss", '3d100', '3d200', '3d300', '3d400', '5003d', '3d600', '3d700', '3d800', '3d900', '3d1000'
            # ])
            # test_err_all = np.zeros(11)
            # from PGBIG
            ret_log_test = np.array(np.zeros(6))  # all, per
            head_test = np.array(["test_loss", '3d200', '3d400', '3d600', '3d800', '3d1000'])
            test_err_all = np.zeros(6)
            # using transform from train-set
            test_err_avg, test_err_all = self.engine.test(test_loader, self.config["setting"]["input_n"],
                                                          np.array(self.config["setting"]["eval_frame"]),
                                                          np.array(self.config["setting"]["dim_used"]),
                                                          np.array(self.config["setting"]["joint_to_ignore"]),
                                                          np.array(self.config["setting"]["joint_to_equal"]),
                                                          test_dataset.time_tsfm, None, "all")

            ret_log_test[0] = test_err_avg
            ret_log_test[1:len(test_err_all) + 1] = test_err_all

            # update log file and save checkpoint
            ret_log = np.append(ret_log_train, ret_log_test)
            head = np.append(head_train, head_test)
            df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
            if epoch == start_epoch:
                df.to_csv(self.config["save"]["path"]["base"] + "/training_loss.csv", header=head, index=False)
            else:
                with open(self.config["save"]["path"]["base"] + "/training_loss.csv", "a") as f:
                    df.to_csv(f, header=False, index=False)

            if not np.isnan(test_err_avg):
                is_best = test_err_avg < err_best
                err_best = min(test_err_avg, err_best)
            else:
                is_best = False
            self.engine.save(self.config["save"]["path"]["checkpoints"], test_err_avg, epoch, is_best)
            if is_best:
                ret_log_best = ret_log

            self.logger.info(">>> epoch: {} | loss: {:.5f} | best: {:.5f}".format(epoch + 1, test_err_avg, err_best))

            # update lr outside the runner
            # self.engine.scheduler.step(test_err_avg)
            # if self.engine.lr != self.engine.optimizer.param_groups[0]['lr']:
            #     self.engine.recover(
            #         self.config["save"]["path"]["checkpoints"] + "/best.pth",
            #         True)
            # self.engine.lr = self.engine.optimizer.param_groups[0]['lr']

            # best result for all
            # if ret_log_best is None:
            #     ret_log_best = ret_log
            # else:
            #     ret_log_best = np.min(np.stack((ret_log_best, ret_log),
            #                                    axis=0),
            #                           axis=0)

        # save the best result
        df = pd.DataFrame(np.expand_dims(ret_log_best, axis=0))
        with open(self.config["save"]["path"]["base"] + "/training_loss.csv", "a") as f:
            df.to_csv(f, header=False, index=False)

    def run_test(self):
        self.logger.info("Start testing")
        # dataset
        # get transform from train dataset
        dataset_name = self.config["dataset"]["name"]
        test_data_config = self.config["dataset"]["test"]

        # manually set the scaler in test dataset
        test_dataset = get_dataset(dataset_name, **test_data_config)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.test_batch_size,
            # batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True)
        self.logger.info("test data shape {}".format(test_dataset.all_seqs.shape[0]))

        # load model checkpoint
        if self.config["model"]["load"]:
            self.engine.recover(self.config["model"]["ckpt"])

        # testing after each epoch
        test_err_avg = 0
        # from DSTDGCN
        ret_log_test = np.array(np.zeros(11))  # all, per
        head_test = np.array(
            ["test_loss", '3d100', '3d200', '3d300', '3d400', '5003d', '3d600', '3d700', '3d800', '3d900', '3d1000'])
        test_err_all = np.zeros(11)
        # from PGBIG
        ret_log_test = np.array(np.zeros(6))  # all, per
        head_test = np.array(["test_loss", '3d200', '3d400', '3d600', '3d800', '3d1000'])
        test_err_all = np.zeros(6)

        # using transform from train-set
        test_err_avg, test_err_all = self.engine.test(
            test_loader, self.config["setting"]["input_n"], np.array(self.config["setting"]["eval_frame"]),
            np.array(self.config["setting"]["dim_used"]), np.array(self.config["setting"]["joint_to_ignore"]),
            np.array(self.config["setting"]["joint_to_equal"]), test_dataset.time_tsfm, None, "all",
            self.config["save"]["path"]["visualize"] + "all" if self.config["setting"]["save"] else None)
        # ret_log = np.append(ret_log, test_l)
        # test_err_avg = np.average(test_err_all)

        ret_log_test[0] = test_err_avg
        ret_log_test[1:len(test_err_all) + 1] = test_err_all

        self.logger.info("Loss: {:.5f}".format(test_err_avg))
        self.logger.info("Save result to " + self.config["save"]["path"]["base"] + "/testing_loss.csv")
        # save metric
        df = pd.DataFrame(np.expand_dims(ret_log_test, axis=0))
        df.to_csv(self.config["save"]["path"]["base"] + "/testing_loss.csv", header=head_test, index=False)

    def run_test_all(self):
        raise NotImplementedError()
    
    def run_visualize(self):
        max_seqs = 8
        # dataset
        # get transform from train dataset
        train_data_config = self.config["dataset"]["train"]
        dataset_name = self.config["dataset"]["name"]

        train_dataset = get_dataset(dataset_name, **train_data_config)
        # print("train data shape {}".format(train_dataset.all_seqs.shape[0]))
        self.logger.info("train data shape {}".format(
            train_dataset.all_seqs.shape[0]))

        test_data_config = self.config["dataset"]["test"]
        # manually set the scaler in test dataset
        test_data_config[dataset_name]["scaler"] = train_dataset.scale_tsfm

        vis_runner = Visualizer(self.dataset)

        # manually set the scaler in test dataset
        test_dataset = get_dataset(dataset_name, **test_data_config)
        self.logger.info("test data shape {}".format(len(test_dataset)))
        vis_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.config.num_workers,
                                 pin_memory=True)

        # visualize every sequence
        for idx, (_, _, _, out_seq) in enumerate(vis_loader):
            vis_seq = out_seq[0].cpu().numpy()
            figure_title = "S{}".format(idx + 1)
            vis_runner.plot_single(
                vis_seq, self.config["save"]["path"]["visualize"],
                figure_title, self.config["setting"]["input_n"])
            print("Now", figure_title)

    def run(self):
        if "train" in self.config["mode"]:
            self.run_train()
        elif "test" in self.config["mode"]:
            if "visualize" in self.config["mode"]:
                self.config["setting"]["save"] = True
            if "all" in self.config["mode"]:
                self.run_test_all()
            else:
                self.run_test()
        else:
            self.run_visualize()
