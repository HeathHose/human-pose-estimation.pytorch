# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        """
        pixel_std 设为200???
        """
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):

        db_rec = copy.deepcopy(self.db[idx])
        """
        db_rec: 坐标为整张图片左上角为原点,目标检测框和关键点的坐标信息
        image
        filename
        imgnum
        jionts_3d
        joints_3d_vis
        center
        score
        scale
        deepcopy 子list的改变,不会影响db_rec
        """
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        """
        cv2: 
        cv2.IMREAD_COLOR：读入一副彩色图像。图像的透明度会被忽略
        cv.IMREAD_IGNORE_ORIENTATION    
            1.不因EXIF 方向标志(orientation)而转化图像的坐标:
            2.EXIF（Exchangeable Image File）是“可交换图像文件”的缩写
                当中包含了专门为数码相机的照片而定制的元数据
                可以记录数码照片的拍摄参数、缩略图及其他属性信息
            3.Orientation参数，记录图片拍摄的相机的旋转信息，浏览器根据orientation参数中的val信息自动旋转图片到正确的方向
        """
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        """
        random.random()生成[0,1]之间的浮点数
        np.random.randn() 生成标准正态分布的一个或一组数据 
        scale_factor :0.3
        rotation_factor :40
        
        仿射变换（对应着五种变换，平移，缩放，旋转，翻转，错切）是一种二维坐标到二维坐标的x线性变换
        保持了二维图形的平直性（直线变换后仍然是直线）和平行性（二维图形之间相对位置关系保持不变）
        
        s:仿射变换的旋转参数
        随机缩放sf+1,缩放幅度分布:[0.7,1.3] ???为什么缩放这么多
        r:仿射变换的参数
        0.6(旋转)：0.4（不旋转） 旋转幅度随机分布到：[-pi*40/180,pi*40/180] ???为什么旋转40
        """
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
        """
        对目标检测框进行仿射变换，调整成256X192：
        1.之前在coco.py 提取目标检测框，构造rec时,已将图片中的框 宽高比例调成 192/256
        2.现在 进行仿射变换将目标检测框 旋转做数据增强 scale使图片变为256X192
        """
        trans = get_affine_transform(c, s, r, self.image_size)
        """
        ??? 区别不明
        cv2.INTER_LINEAR: 设置插值方式，默认方式为线性插值
        INTER_NEAREST	最近邻插值
        INTER_LINEAR	双线性插值（默认设置）
        INTER_AREA	使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
        INTER_CUBIC	4x4像素邻域的双三次插值
        INTER_LANCZOS4	8x8像素邻域的Lanczos插值
        
        在缩小时推荐cv2.INTER_ARER,扩大是推荐cv2.INTER_CUBIC和cv2.INTER_LINEAR
        """
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        """
        dataloader封装好的数据预处理
        """
        if self.transform:
            input = self.transform(input)
        """
        对关键点进行仿射变换
        affine_transform:
        1.将关键点的[x,y,1]转置变成列向量
        2.左乘变换矩阵,进行仿射变换
        """
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
    """
    ???
    """
    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            """
            所以这个pixel_std到底是拿来干啥的
            scale = [w/pixel_std,h/pixel_std]
            area = scale[0]*scale[1]*pixel_std**2
            """
            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):###???
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        """
        target_weight(1: visible, 0: invisible),
        """
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            """
            ul: upper left 左上角
            br: bottom right 右下角
            feat_stride: 检测框(图片)大小与 热力图大小的比
            int(x+0.5) 四舍五入
            
            热力图
            mu_x: 转换成热力图之后，关节点的坐标x
            mu_y: 转换成热力图之后，关节点的坐标y
            去掉了小数点，为保持tmp_szie**2的大小，补偿br（int(mu_x + tmp_size + 1)）
            距离热力图64x48中心点 tmp_size(3sigma)距离表示： 距离关键点sigma个像素
            离sigma距离,正太分布占68.4
            离2sigma距离，正太分布占95.4
            离3sigma距离，正太分布占99.8 
            """
            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]#？？？？ 保证3*sigma的范围都在热力图内
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                """
                x[:, np.newaxis]  在这一位置增加一个一维
                
                from __future__ import division : " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法
                """
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
