import os
import pdb
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset import define_actions, get_dataset
from engine import PredictionEngine
from model import get_model
from torch.utils.data import DataLoader
from utils import plot_predictions_single


class AMASSRunner:
    def __init__(self, config):
        # read config from path
        # update config
        self.config = config

        if "visualize" not in self.config["mode"]:
            model_opts = config["model"]
            model = get_model(config["model"]["name"], **model_opts)
            model = model.to(config["device"])
            self.logger = config["logger"]
            self.engine = PredictionEngine(config["engine"], model,
                                           self.logger)
        self.save_files()

        torch.backends.cudnn.benchmark = True

    def save_files(self):
        # make directories
        for path in self.config["save"]["path"]:
            if path != "base":
                update_path = os.path.join(self.config["save"]["path"]["base"],
                                           self.config["save"]["path"][path])
                self.config["save"]["path"][path] = update_path
                os.makedirs(update_path, exist_ok=True)

        # save files
        for file in self.config["save"]["files"]:
            shutil.copy(file, self.config["save"]["path"]["files"])

    def run_train(self):
        # dataset
        # update config files
        train_data_config = self.config["dataset"]["train"]
        dataset_name = self.config["dataset"]["name"]
        if "debug" in self.config["mode"]:
            test_acts = define_actions("debug", self.config["dataset"]["name"])
            train_data_config[dataset_name]["actions"] = "debug"
            train_data_config[dataset_name]["mode"] = "debug"
        else:
            test_acts = define_actions("all")
            train_data_config[dataset_name]["actions"] = "all"
            train_data_config[dataset_name]["mode"] = "train"
        train_dataset = get_dataset(dataset_name, **train_data_config)
        # print("train data shape {}".format(train_dataset.all_seqs.shape[0]))
        self.logger.info("train data shape {}".format(
            train_dataset.all_seqs.shape[0]))
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.config.train_batch_size,
                                  shuffle=True,
                                  num_workers=self.config.num_workers,
                                  pin_memory=True)

        test_data_config = self.config["dataset"]["test"]
        # manually set the scaler in test dataset
        test_data_config[dataset_name]["scaler"] = train_dataset.scale_tsfm
        test_loaders = dict()
        for act in test_acts:
            # manually update actions for test dataset
            test_data_config[dataset_name]["actions"] = act
            test_dataset = get_dataset(dataset_name, **test_data_config)
            test_loaders[act] = DataLoader(
                dataset=test_dataset,
                batch_size=self.config.test_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True)

        # load model
        if self.config["model"]["load"]:
            start_epoch, err_best = self.engine.recover(
                self.config["model"]["ckpt"])
        else:
            start_epoch, err_best = 0, 1e10

        ret_log_best = None
        # start training
        for epoch in range(start_epoch, self.config["epoch"]):

            # print("==========================")
            # print(">>> epoch: {} | lr: {:.5f}".format(epoch + 1,
            #                                           self.engine.lr))
            self.logger.info("==========================")
            self.logger.info(">>> epoch: {} | lr: {:.5f}".format(
                epoch + 1, self.engine.lr))

            # writing result to csv
            ret_log_train = np.array([epoch + 1])
            head_train = np.array(["epoch"])

            # training for each epoch
            train_loss = self.engine.train(
                train_loader,
                epoch,
                train_dataset.time_tsfm,
                train_dataset.scale_tsfm,
                train_dataset.joint_weight_use
                if self.config["engine"]["use_weight"] else None,
            )
            ret_log_train = np.append(ret_log_train,
                                      [self.engine.lr, train_loss])
            head_train = np.append(head_train, ["lr", "train_loss"])

            # testing after each epoch
            test_err_avg = 0
            if self.config["setting"]["output_n"] > 10:
                ret_log_test = np.array(np.zeros(9))  # all, per
                head_test = np.array([
                    "test_loss", "3d80", "3d160", "3d320", "3d400", "3d560",
                    "3d720", "3d880", "3d1000"
                ])
                test_err_all = np.zeros(8)
            else:
                ret_log_test = np.array(np.zeros(5))
                head_test = np.array(
                    ["test_loss", "3d80", "3d160", "3d320", "3d400"])
                test_err_all = np.zeros(4)
            for act in test_acts:
                # using transform from train-set
                act_test_err_avg, act_test_err_all = self.engine.test(
                    test_loaders[act], self.config["setting"]["input_n"],
                    np.array(self.config["setting"]["eval_frame"]),
                    np.array(self.config["setting"]["dim_used"]),
                    np.array(self.config["setting"]["joint_to_ignore"]),
                    np.array(self.config["setting"]["joint_to_equal"]),
                    train_dataset.time_tsfm, train_dataset.scale_tsfm, act)
                # ret_log = np.append(ret_log, test_l)
                # test_err_avg = np.average(test_err_all)
                test_err_avg += act_test_err_avg
                test_err_all += act_test_err_all
                ret_log_test = np.append(ret_log_test, act_test_err_all)
                head_test = np.append(head_test, [
                    act + "3d80", act + "3d160", act + "3d320", act + "3d400"
                ])
                if self.config["setting"]["output_n"] > 10:
                    head_test = np.append(head_test, [
                        act + "3d560", act + "3d720", act + "3d880",
                        act + "3d1000"
                    ])

            # update metric
            # average on all losses
            test_err_avg /= len(test_acts)
            test_err_all /= len(test_acts)

            ret_log_test[0] = test_err_avg
            ret_log_test[1:len(test_err_all) + 1] = test_err_all

            # update log file and save checkpoint
            ret_log = np.append(ret_log_train, ret_log_test)
            head = np.append(head_train, head_test)
            df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
            if epoch == start_epoch:
                df.to_csv(self.config["save"]["path"]["base"] +
                          "/training_loss.csv",
                          header=head,
                          index=False)
            else:
                with open(
                        self.config["save"]["path"]["base"] +
                        "/training_loss.csv", "a") as f:
                    df.to_csv(f, header=False, index=False)

            if not np.isnan(test_err_avg):
                is_best = test_err_avg < err_best
                err_best = min(test_err_avg, err_best)
            else:
                is_best = False
            self.engine.save(self.config["save"]["path"]["checkpoints"],
                             test_err_avg, epoch, is_best)

            # update lr outside the runner
            # self.engine.scheduler.step(test_err_avg)
            # if self.engine.lr != self.engine.optimizer.param_groups[0]['lr']:
            #     self.engine.recover(
            #         self.config["save"]["path"]["checkpoints"] + "/best.pth",
            #         True)
            # self.engine.lr = self.engine.optimizer.param_groups[0]['lr']

            if ret_log_best is None:
                ret_log_best = ret_log
            else:
                ret_log_best = np.min(np.stack((ret_log_best, ret_log),
                                               axis=0),
                                      axis=0)

        # save the best result
        df = pd.DataFrame(np.expand_dims(ret_log_best, axis=0))
        with open(self.config["save"]["path"]["base"] + "/training_loss.csv",
                  "a") as f:
            df.to_csv(f, header=False, index=False)

    def run_test(self):
        # dataset
        # get transform from train dataset
        train_data_config = self.config["dataset"]["train"]
        dataset_name = self.config["dataset"]["name"]
        train_dataset = get_dataset(dataset_name, **train_data_config)
        # print("train data shape {}".format(train_dataset.all_seqs.shape[0]))
        self.logger.info("train data shape {}".format(
            train_dataset.all_seqs.shape[0]))

        test_data_config = self.config["dataset"]["test"]
        if "debug" in self.config["mode"]:
            test_acts = define_actions("debug")
        else:
            test_acts = define_actions("all")
        # manually set the scaler in test dataset
        test_data_config[dataset_name]["scaler"] = train_dataset.scale_tsfm
        test_loaders = dict()
        for act in test_acts:
            # manually update actions for test dataset
            test_data_config[dataset_name]["actions"] = act
            test_dataset = get_dataset(dataset_name, **test_data_config)
            test_loaders[act] = DataLoader(
                dataset=test_dataset,
                batch_size=self.config.test_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True)

        # load model checkpoint
        if self.config["model"]["load"]:
            self.engine.recover(self.config["model"]["checkpoints"])

        # testing after each epoch
        test_err_avg = 0
        if self.config["setting"]["output_n"] > 10:
            ret_log_test = np.array(np.zeros(9))  # all, per
            head_test = np.array([
                "test_loss", "3d80", "3d160", "3d320", "3d400", "3d560",
                "3d720", "3d880", "3d1000"
            ])
            test_err_all = np.zeros(8)
        else:
            ret_log_test = np.array(np.zeros(5))
            head_test = np.array(
                ["test_loss", "3d80", "3d160", "3d320", "3d400"])
            test_err_all = np.zeros(4)

        for act in test_acts:
            # using transform from train-set
            act_test_err_avg, act_test_err_all = self.engine.test(
                test_loaders[act], self.config["setting"]["input_n"],
                np.array(self.config["setting"]["eval_frame"]),
                np.array(self.config["setting"]["dim_used"]),
                np.array(self.config["setting"]["joint_to_ignore"]),
                np.array(self.config["setting"]["joint_to_equal"]),
                train_dataset.time_tsfm, train_dataset.scale_tsfm, act)
            # ret_log = np.append(ret_log, test_l)
            # test_err_avg = np.average(test_err_all)
            test_err_avg += act_test_err_avg
            test_err_all += act_test_err_all
            ret_log_test = np.append(ret_log_test, act_test_err_all)
            head_test = np.append(
                head_test,
                [act + "3d80", act + "3d160", act + "3d320", act + "3d400"])
            if self.config["setting"]["output_n"] > 10:
                head_test = np.append(head_test, [
                    act + "3d560", act + "3d720", act + "3d880", act + "3d1000"
                ])

        # update metric
        test_err_avg /= len(test_acts)
        test_err_all /= len(test_acts)

        ret_log_test[0] = test_err_avg
        ret_log_test[1:len(test_err_all) + 1] = test_err_all

        self.logger.info("Save result to " +
                         self.config["save"]["path"]["base"] +
                         "/testing_loss.csv")

        df = pd.DataFrame(np.expand_dims(ret_log_test, axis=0))
        df.to_csv(self.config["save"]["path"]["base"] + "/testing_loss.csv",
                  header=head_test,
                  index=False)

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
        if "debug" in self.config["mode"]:
            test_acts = define_actions("debug")
        else:
            test_acts = define_actions("all")
        # manually set the scaler in test dataset
        test_data_config[dataset_name]["scaler"] = train_dataset.scale_tsfm

        fig = plt.figure()
        ax = plt.gca(projection="3d")

        for act in test_acts:
            # manually update actions for test dataset
            test_data_config[dataset_name]["actions"] = act
            test_dataset = get_dataset(dataset_name, **test_data_config)
            act_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=self.config.num_workers,
                                    pin_memory=True)
            seq_count = 0
            for (_, _, _, seq) in act_loader:
                seq_expmap = seq[0].cpu().numpy()
                plt.cla()
                figure_title = "A{}_S{}".format(act, (seq_count + 1))
                plot_predictions_single(seq_expmap, fig, ax, figure_title)
                plt.savefig(self.config["save"]["path"]["visualize"] +
                            figure_title + ".png")
                seq_count += 1
                if seq_count >= max_seqs:
                    break

    def run(self):
        if "train" in self.config["mode"]:
            self.run_train()
        elif "test" in self.config["mode"]:
            self.run_test()
        else:
            self.run_visualize()
