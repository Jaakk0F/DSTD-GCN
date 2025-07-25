import os
import random
import shutil

import numpy as np
import torch
from dataset import define_actions, get_dataset
from engine import PredictionEngine
from model import get_model
from torch.utils.data import DataLoader
from utils import Visualizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BaseRunner:

    def __init__(self, config):
        # read config from path
        # update config
        self.config = config

        self.logger = config["logger"]
        # if train, test, load model
        # else visualize
        if "t" in self.config["mode"]:
            model_opts = config["model"]
            model = get_model(config["model"]["name"], **model_opts)
            model = model.to(config["device"])
            self.engine = PredictionEngine(config["engine"], model,
                                           self.logger)
        self.dataset = config["dataset"]["name"]
        self.save_files()
        setup_seed(777)

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
        raise NotImplementedError()

    def run_test(self):
        raise NotImplementedError()

    def run_test_all(self):
       raise NotImplementedError()

    def run_visualize(self):
        max_seqs = 8
        # dataset
        # get transform from train dataset
        train_data_config = self.config["dataset"]["train"]
        dataset_name = self.config["dataset"]["name"]
        if "debug" in self.config["mode"]:
            test_acts = define_actions("debug", self.config["dataset"]["name"])
            train_data_config[dataset_name]["actions"] = "debug"
        else:
            test_acts = define_actions("all", self.config["dataset"]["name"])
            train_data_config[dataset_name]["actions"] = "all"
        train_dataset = get_dataset(dataset_name, **train_data_config)
        # print("train data shape {}".format(train_dataset.all_seqs.shape[0]))
        self.logger.info("train data shape {}".format(
            train_dataset.all_seqs.shape[0]))

        test_data_config = self.config["dataset"]["test"]
        # manually set the scaler in test dataset
        test_data_config[dataset_name]["scaler"] = train_dataset.scale_tsfm

        vis_runner = Visualizer(self.dataset)

        # 8 sequence mode
        for act in test_acts:
            # manually update actions for test dataset
            test_data_config[dataset_name]["actions"] = act
            test_dataset = get_dataset(dataset_name, **test_data_config)
            act_loader = DataLoader(dataset=test_dataset,
                                    batch_size=8,
                                    shuffle=False,
                                    num_workers=self.config.num_workers,
                                    pin_memory=True)
            # max sequence mode
            # seq_count = 0
            # for (_, _, _, seq) in act_loader:
            #     seq = seq[0].cpu().numpy()
            #     figure_title = "A{}_S{}".format(act, (seq_count + 1))
            #     vis_runner.plot_single(
            #         seq, self.config["save"]["path"]["visualize"],
            #         figure_title)
            #     seq_count += 1
            #     if seq_count >= max_seqs:
            #         break
            # all sequence mode, for 8 test sequence
            for (_, _, _, out_seq) in act_loader:
                n_seq = out_seq.shape[0]
                for i in range(n_seq):
                    vis_seq = out_seq[i].cpu().numpy()
                    figure_title = "A{}_S{}".format(act, (i + 1))
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
