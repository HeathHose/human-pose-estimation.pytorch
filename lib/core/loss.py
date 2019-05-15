# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        """
        如果在创建MSELoss实例的时候在构造函数中传入size_average=False，那么求出来的平方和将不会除以n
        """
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    """
    def forward()函数式编程的部分
    """
    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        """
        tensor.reshape((::-1))保留前两维维数
        tensor.split(split_size=1,dim=1) 在第一维进行切割成多份維度为1的tensor
        tensor.squeeze()将输入张量形状中的 1 去除
        
        output 维度变化
        reshape:32,17,64,48改变为 32,17,3072
        split:  32,17,3072改变为 17个（32,1,3072）
        squeeze：32.1.3072改变为32,3072
        方便循环处理num_joints为0-16的pred
        
        """
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            """
            loss*0.5 约定俗成，进行梯度下降时，与平方导数得到的2抵消
            tensor.mul() tensor 与tensor之间,具体维度可以不同，总元素必须相同（4x4 与2x8）
            """
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        """
        这里代码出现小bug：
        vis是定义为的可见性标志。v = 0表示未标记，v = 1表示标记但不可见，v = 2表示标记且可见???
        num_joints的数量等于 样本中in_vis=2 的数量,不一定等于17
        """
        return loss / num_joints
