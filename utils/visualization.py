"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
    For now, it supports h36m, cmu and 3dpw
"""
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

plt.switch_backend("agg")


class Visualizer:

    def __init__(self, dataset="h36m", pose_size=(140, 200), center=(320, 260)):
        # image shape (640, 480)
        self.pose_size = pose_size  # single pose image size
        self.center = center
        self.dataset = dataset
        # Start and endpoints of our representation
        if dataset == "h36m":
            self.num_joints = 32
            self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
            self.J = (np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1)
            # Left / right indicator
            self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
            vals = np.zeros((32, 3))
        elif dataset == "cmu":
            self.num_joints = 38
            self.I = np.array([
                1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16,
                30, 31, 32, 33, 34, 35, 33, 37
            ]) - 1
            self.J = np.array([
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38
            ]) - 1
            # Left / right indicator
            self.LR = np.array([
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0
            ],
                               dtype=bool)
        elif dataset == "3dpw":
            self.num_joints = 24
            self.I = np.array([0, 0, 0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
            self.J = np.array([1, 2, 3, 4, 7, 10, 5, 8, 11, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
            # Left / right indicator
            self.LR = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        else:
            raise ValueError("No such dataset")

    def plot_single(self, data, save_path, file_name, input_n=10, save_image=False):
        num_frames = data.shape[0]
        data = data.reshape(num_frames, self.num_joints, 3)[:, :, (0, 2, 1)]
        save_names = []
        # visualize the point
        # ax.scatter(x, y, z, c='b')

        # (250, 40, 40) #FA2828 红
        # (245, 125, 125) #F57D7D 粉
        # (11, 11, 11) #0B0B0B 黑色
        # (180, 180, 180) #B4B4B4 灰色
        # #3498db 蓝色
        # #e74c3c 红色
        # #FFFF00 黄色
        # #       绿色
        # Make connection matrix
        for idf in range(num_frames):
            plt.clf()
            plt.figure()
            ax = plt.gca(projection='3d')
            # 关闭坐标轴
            ax._axis3don = False
            ax.grid(False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim3d([-1000, 1000])
            ax.set_ylim3d([-1000, 1000])
            ax.set_zlim3d([-1000, 1000])
            # 透明化背景以及刻度
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            data_frame = data[idf]
            for i in np.arange(len(self.I)):
                x, y, z = [np.array([data_frame[self.I[i], j], data_frame[self.J[i], j]]) for j in range(3)]

                if self.dataset == "3dpw":
                    x, y, z = y, x, z
                else:
                    y = -y  # 改变面朝向
                # 使用单种颜色，只区分输入和输出
                if idf < input_n:
                    ax.plot(x, y, z, lw=2, c='#0B0B0B')
                else:
                    ax.plot(x, y, z, lw=2, c='#3498db')
                # 使用不同的颜色标注两边
                # if self.LR[i] == 0:
                #     ax.plot(x, y, z, lw=2, c='#FA2828')
                # elif self.LR[i] == 1:
                #     ax.plot(x, y, z, lw=2, c='#F57D7D')
                # else:
                #     ax.plot(x, y, z, lw=2, c='#FFFF00')

            save_name = save_path + f"{file_name}_{idf}.png"
            plt.axis("off")
            plt.savefig(save_name)
            plt.close()
            save_names.append(save_name)

        # read and make gif
        images = []
        imgs_smy = []
        for save_name in save_names:
            image = imageio.imread(save_name)
            images.append(image)
            img_smy = Image.open(save_name)
            imgs_smy.append(img_smy)
        imageio.mimsave(save_path + file_name + '.gif', images, 'GIF', duration=0.05)

        # delete files
        if not save_image:
            for save_name in save_names:
                os.remove(save_name)

        # save summary
        # single image size (100, 150)
        w, h = self.pose_size
        w_c, h_c = self.center
        image_save = Image.new('RGB', (w * (len(imgs_smy) + 1) // 2, h))
        for i, image in enumerate(imgs_smy):
            if i % 2 == 1:
                continue
            image = image.crop((w_c - w // 2, h_c - h // 2, w_c + w // 2, h_c + h // 2))
            image_save.paste(image, ((i // 2) * w, 0))
        image_save.save(save_path + file_name + '.png')

    def plot_multi(self, data_pred, data_gt, save_path, file_name, save_image=False):
        assert data_pred.shape == data_gt.shape
        num_frames = data_pred.shape[0]
        data_pred = data_pred.reshape(num_frames, self.num_joints, 3)[:, :, (0, 2, 1)]
        data_gt = data_gt.reshape(num_frames, self.num_joints, 3)[:, :, (0, 2, 1)]
        save_names = []
        # visualize the point
        # ax.scatter(x, y, z, c='b')

        # (250, 40, 40) #FA2828 红
        # (245, 125, 125) #F57D7D 粉
        # (11, 11, 11) #0B0B0B 黑色
        # (180, 180, 180) #B4B4B4 灰色
        # #3498db 蓝色
        # #e74c3c 红色
        # #FFFF00 黄色
        # Make connection matrix
        for idf in range(num_frames):
            plt.clf()
            plt.figure()
            ax = plt.gca(projection='3d')
            ax._axis3don = False
            ax.grid(False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim3d([-1000, 1000])
            ax.set_ylim3d([-1000, 1000])
            ax.set_zlim3d([-1000, 1000])
            # 透明化背景以及刻度
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            image_pred = data_pred[idf]
            image_gt = data_gt[idf]
            for i in np.arange(len(self.I)):
                # plot prediction
                x, y, z = [np.array([image_pred[self.I[i], j], image_pred[self.J[i], j]]) for j in range(3)]
                if self.dataset == "3dpw":
                    x, y, z = y, x, z
                else:
                    y = -y  # 改变面朝向
                ax.plot(x, y, z, lw=2, color='#FA2828' if self.LR[i] else '#FA2828')
                # plot ground truth
                x, y, z = [np.array([image_gt[self.I[i], j], image_gt[self.J[i], j]]) for j in range(3)]
                if self.dataset == "3dpw":
                    x, y, z = y, x, z
                else:
                    y = -y  # 改变面朝向
                ax.plot(x, y, z, lw=1, color='#3498db' if self.LR[i] else '#3498db')
            save_name = save_path + f"{file_name}_{idf}.png"
            plt.savefig(save_name)
            plt.close()
            save_names.append(save_name)

        # read and make gif
        images = []
        imgs_smy = []
        for save_name in save_names:
            image = imageio.imread(save_name)
            images.append(image)
            img_smy = Image.open(save_name)
            imgs_smy.append(img_smy)
        imageio.mimsave(save_path + file_name + '.gif', images, 'GIF', duration=0.05)

        # delete files
        if not save_image:
            for save_name in save_names:
                os.remove(save_name)

        # save summary
        # single image size (100, 150)
        w, h = self.pose_size
        w_c, h_c = self.center
        image_save = Image.new('RGB', (w * (len(imgs_smy) + 1) // 2, h))
        for i, image in enumerate(imgs_smy):
            if i % 2 == 1:
            # if i % 2 == 0:
                continue
            image = image.crop((w_c - w // 2, h_c - h // 2, w_c + w // 2, h_c + h // 2))
            image_save.paste(image, ((i // 2) * w, 0))
        image_save.save(save_path + file_name + '.png')


def plot_predictions_single(xyz_show, full_path):
    nframes_pred = xyz_show.shape[0]
    # # Compute 3d points for each frame
    # xyz_show = np.zeros((nframes_pred, 96))
    # for i in range(nframes_pred):
    #     xyz_show[i, :] = fkl(expmap[i, :], parent, offset, rotInd,
    #                          expmapInd).reshape([96])

    # # ****************************
    # # 调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    mydata = xyz_show.reshape(38, 3)[:, [0, 2, 1]]
    # # ****************************

    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 2]

    I = np.array([
        1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
        32, 33, 34, 35, 33, 37
    ]) - 1
    J = np.array([
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38
    ]) - 1
    # Left / right indicator
    LR = np.array(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    # visualize the point
    # ax.scatter(x, y, z, c='b')

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色
    # #3498db 蓝色
    # #e74c3c 红色
    # #FFFF00 黄色
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        if LR[i] == 0:
            ax.plot(x, y, z, lw=2, c='#3498db')
        elif LR[i] == 1:
            ax.plot(x, y, z, lw=2, c='#3498db')
        else:
            ax.plot(x, y, z, lw=2, c='#FFFF00')

    # initialize view
    plt.savefig(full_path)
    plt.close()


def draw_pic_single(mydata, full_path):
    # 22, 3
    # I
    # J
    # LR

    # # ****************************
    # # 调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    mydata = mydata.reshape(38, 3)[:, [0, 2, 1]]
    # # ****************************

    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 2]

    I = np.array([
        1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
        32, 33, 34, 35, 33, 37
    ]) - 1
    J = np.array([
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38
    ]) - 1
    # Left / right indicator
    LR = np.array(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    # visualize the point
    # ax.scatter(x, y, z, c='b')

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色
    # #3498db 蓝色
    # #e74c3c 红色
    # #FFFF00 黄色
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        if LR[i] == 0:
            ax.plot(x, y, z, lw=2, c='#3498db')
        elif LR[i] == 1:
            ax.plot(x, y, z, lw=2, c='#3498db')
        else:
            ax.plot(x, y, z, lw=2, c='#FFFF00')

    # initialize view
    plt.savefig(full_path)
    plt.close()


# plot ground truth and prediction
class Ax3DPoseMulti(object):

    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=["GT", "Pred"]):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = (np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1)
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(
                        x,
                        z,
                        y,
                        lw=2,
                        linestyle="--",
                        c=rcolor if self.LR[i] else lcolor,
                        label=label[0],
                    ))
            else:
                self.plots.append(self.ax.plot(
                    x,
                    y,
                    z,
                    lw=2,
                    linestyle="--",
                    c=rcolor if self.LR[i] else lcolor,
                ))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(
                    x,
                    y,
                    z,
                    lw=2,
                    c=rcolor if self.LR[i] else lcolor,
                    label=label[1],
                ))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc="lower left")
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 96, ("channels should have 96 entries, it has %d instead" % gt_channels.size)
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 96, ("channels should have 96 entries, it has %d instead" % pred_channels.size)
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        # r = 750
        r = 1000
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect("equal")


def plot_predictions_multi(expmap_gt, expmap_pred, fig, ax, f_title):
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    nframes_pred = expmap_pred.shape[0]

    # Compute 3d points for each frame
    xyz_gt = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_gt[i, :] = fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd).reshape([96])
    xyz_pred = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_pred[i, :] = fkl(expmap_pred[i, :], parent, offset, rotInd, expmapInd).reshape([96])

    # === Plot and animate ===
    ob = Ax3DPoseMulti(ax)
    # Plot the prediction
    for i in range(nframes_pred):
        ob.update(xyz_gt[i, :], xyz_pred[i, :])
        ax.set_title(f_title + " frame:{:d}".format(i + 1), loc="left")
        plt.show(block=False)

        fig.canvas.draw()
        plt.pause(0.05)
