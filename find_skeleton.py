import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import scipy.io
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as transforms_F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json
from pycocotools.coco import COCO
from os import path
import math
import torch.nn.functional as F
from scipy import ndimage
from numpy import matlib
from torch.optim import lr_scheduler
from apex import amp

from torch.nn.modules import loss
from skimage.feature import peak_local_max
from tensorboardX import SummaryWriter
import random
import time
from torchstat import stat

matplotlib.use('TkAgg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nModules = 2
nFeats = 256
nStack = 3
nKeypoint_COCO = 17
nSkeleton_COCO = 19
nKeypoint_MPII = 16
nSkeleton_MPII = 15
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton_MPII + 1
nOutChannels_2 = nKeypoint_MPII + 1
epochs = 300
batch_size = 48
keypoints = 17
skeleton = 20
inputsize = 256
learning_rate = 1e-4

threshold = 1

mode = 'test'
load_model_name = 'params_1_add_cross_entropy_and_bootstrapped_together_fine_tune'
save_model_name = 'params_1_add_cross_entropy_and_bootstrapped_together_fine_tune'
# load_mask_name = 'params_1_mask'
# save_mask_name = 'params_1_mask'

train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
train_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
val_set_coco = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_val2017.json'
train_image_dir_coco = '/data/COCO/COCO2017/train2017/'
val_image_dir_coco = '/data/COCO/COCO2017/val2017'

loss_img = save_model_name[:-4] + 'loss.png'
accuracy_img = save_model_name[:-4] + 'accuracy.png'

rootdir = '/data/lsp_dataset/images/'
retrain = False
train_mask = False
usemask = False
write = True
fine_tune = False
dataset = 'mpii'

sks = [[0, 1],
       [1, 2],
       [2, 6],
       [6, 3],
       [3, 4],
       [4, 5],
       [6, 7],
       [7, 8],
       [8, 9],
       [10, 11],
       [11, 12],
       [12, 8],
       [8, 13],
       [13, 14],
       [14, 15]
       ]


class ResidualBlock(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.numIn = numIn
        self.numOut = numOut
        self.bn1 = nn.BatchNorm2d(numIn)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(numIn, int(numOut / 2), 1, 1)
        self.bn2 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(int(numOut / 2), int(numOut / 2), 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(int(numOut / 2))
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(int(numOut / 2), numOut, 1, 1)
        self.bn4 = nn.BatchNorm2d(numOut)
        self.downsaple = nn.Sequential(
            nn.Conv2d(numIn, numOut, 1, stride=stride, bias=False),
            nn.BatchNorm2d(numOut)
        )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        out = self.bn4(x)
        if self.stride != 1 | self.numIn != self.numOut:
            residual = self.downsaple(residual)
        out += residual
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP_Block(nn.Module):
    def __init__(self):
        super(ASPP_Block, self).__init__()
        inplanes = 256
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        out = self.conv1(x)
        return out


class hourglass(nn.Module):
    def __init__(self, f):
        super(hourglass, self).__init__()
        self.f = f

        self.downsample1 = ResidualBlock(f, f, stride=2)
        self.downsample2 = ResidualBlock(f, f, stride=2)
        self.downsample3 = ResidualBlock(f, f, stride=2)
        self.downsample4 = ResidualBlock(f, f, stride=2)

        self.residual1 = ResidualBlock(f, int(f / 2))
        self.residual2 = ResidualBlock(f, int(f / 2))
        self.residual3 = ResidualBlock(f, int(f / 2))
        self.residual4 = ResidualBlock(f, int(f / 2))

        self.upsample1 = ResidualBlock(f, int(f / 2))
        self.upsample2 = ResidualBlock(f, int(f / 2))
        self.upsample3 = ResidualBlock(f, int(f / 2))
        self.upsample4 = ResidualBlock(f, int(f / 2))

        self.aspp = ASPP_Block()

    def forward(self, x):
        up1 = self.residual1(x)
        down1 = self.downsample1(x)
        up2 = self.residual2(down1)
        down2 = self.downsample2(down1)
        up3 = self.residual3(down2)
        down3 = self.downsample3(down2)
        up4 = self.residual4(down3)
        down4 = self.downsample4(down3)
        out = self.aspp(down4)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample4(out)
        out = torch.cat([out, up4], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample3(out)
        out = torch.cat([out, up3], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample2(out)
        out = torch.cat([out, up2], dim=1)
        out = F.interpolate(out, scale_factor=2)
        out = self.upsample1(out)
        out = torch.cat([out, up1], dim=1)
        return out


class creatModel(nn.Module):
    def __init__(self):
        super(creatModel, self).__init__()
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, nFeats)
        )

        self.stage1 = hourglass(nFeats)
        self.stage1_out = nn.Conv2d(nFeats, nOutChannels_0, 1, 1, 0, bias=False)
        self.stage1_return = nn.Conv2d(nOutChannels_0, int(nFeats / 2), 1, 1, 0, bias=False)
        self.stage1_retuen_2 = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)
        self.stage1_down_feature = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)

        self.stage2 = hourglass(nFeats)
        self.stage2_out = nn.Conv2d(nFeats, nOutChannels_1, 1, 1, 0, bias=False)
        self.stage2_return = nn.Conv2d(nOutChannels_1, int(nFeats / 2), 1, 1, 0, bias=False)
        self.stage2_retuen_2 = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)
        self.stage2_down_feature = nn.Conv2d(nFeats, int(nFeats / 4), 1, 1, 0, bias=False)

        self.stage3 = hourglass(nFeats)
        self.stage3_out = nn.Conv2d(nFeats, nOutChannels_2, 1, 1, 0, bias=False)

    def forward(self, x):
        inter = self.preprocess1(x)
        out = []

        i = 0

        ll = self.stage1(inter)
        tmpOut = self.stage1_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage1_return(tmpOut)
        ll_ = self.stage1_retuen_2(ll)
        inter = self.stage1_down_feature(inter)
        inter = torch.cat([tmpOut, ll_, inter], dim=1)

        i = 1

        ll = self.stage2(inter)
        tmpOut = self.stage2_out(ll)
        out.insert(i, tmpOut)
        tmpOut = self.stage2_return(tmpOut)
        ll_ = self.stage2_retuen_2(ll)
        inter = self.stage2_down_feature(inter)
        inter = torch.cat([tmpOut, ll_, inter], dim=1)

        i = 2

        ll = self.stage3(inter)
        tmpOut = self.stage3_out(ll)
        out.insert(i, tmpOut)

        return out


class myImageDataset(data.Dataset):
    def __init__(self, image_dir):
        'Initialization'
        self.image_dir = image_dir
        self.list = os.listdir(image_dir)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        image_name = self.list[index]
        image = Image.open(path.join(self.image_dir, image_name)).resize([256, 256])
        return transforms.ToTensor()(image), image_name


class PCKh(nn.Module):
    def __init__(self):
        super(PCKh, self).__init__()

    def forward(self, x, target, rect):
        accuracy = np.zeros([x.shape[0], 11])
        predicts = []
        labels = []
        for i in range(x.shape[0]):
            correct = np.zeros([11])
            total = np.zeros([11])
            predict = np.zeros([x.shape[1], 2])
            label = np.zeros([x.shape[1], 2])
            standard = np.sqrt((rect[i][0] - rect[i][2]) ** 2 + (rect[i][1] - rect[i][3]) ** 2) * 0.6
            for j in range(x.shape[1]):
                try:
                    label_ys, label_xs = torch.nonzero(target[i] == (j + 1))[0]
                except:
                    continue
                predict_ys, predict_xs = torch.nonzero(x[i, j + 1, :, :] >= torch.max(x[i, j + 1, :, :]))[0]
                distance = torch.sqrt(
                        (torch.pow(label_ys - predict_ys, 2) + torch.pow(label_xs - predict_xs,
                                                                         2)).float()) / standard
                for step, k in enumerate(np.arange(0, 0.55, 0.05)):
                    if distance < k:
                        correct[step] += 1
                    total[step] += 1
                predict[j] = [predict_xs, predict_ys]
                label[j] = [label_xs, label_ys]
            accuracy[i] = (correct / total)
            predicts.append(predict)
            labels.append(label)
        return accuracy, predicts, labels


def main():
    save_dir = '/data/one punch/how people walk skeleton'
    # generatemask = generateMask().cuda().half().eval()
    model = creatModel().cuda().eval().half()
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # state = torch.load(load_mask_name)
    # generatemask.load_state_dict(state['state_dict'])
    state = torch.load(load_model_name)
    model.load_state_dict(state['state_dict'])

    # loss_background = Costomer_CrossEntropyLoss().cuda()
    dataLoader = data.DataLoader(myImageDataset('/data/one punch/how people walk'), batch_size=1, num_workers=1)
    for step, [x_, name] in enumerate(dataLoader):
        bx_ = x_.cuda().half()
        result = model(bx_)
        skeleton = result[1]
        cm = ScalarMappable(Normalize(0, nSkeleton_MPII - 1))
        for i in range(skeleton.shape[0]):
            skeleton_inner = skeleton[i]
            skeleton_inner = torch.argmax(skeleton_inner, dim=0)
            if skeleton_inner.max() != 0:
                skeleton_inner = cm.to_rgba(skeleton_inner.cpu(), bytes=True)[:, :, :3]
                skeleton_image = Image.fromarray(skeleton_inner)
                skeleton_image.save(path.join(save_dir, name[i]))
    print('yyy')


if __name__ == '__main__':
    main()
