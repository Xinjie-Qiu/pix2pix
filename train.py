import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from os import path
from numpy import matlib
from torch.optim import lr_scheduler
from apex import amp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
matplotlib.use('TkAgg')
# from torch.nn.modules import loss
import random
from torchvision.transforms import functional as transforms_F
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
import torch.utils.data as data
from torch import optim
import os
from torch.nn.utils import spectral_norm
import scipy
from scipy import io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


nModules = 2
nFeats = 256
nStack = 3
nKeypoint = 17
nSkeleton = 19
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton + 1
nOutChannels_2 = nKeypoint
epochs = 3000
batch_size = 30
keypoints = 17
skeleton = 20
low_level_chennel = 48

inputsize = 256

threshold = 0.8

save_model_name = 'params_3_one_punch.pkl'
load_model_name = 'params_3_one_punch.pkl'

mode = 'train'
write = True

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

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']

        w, h = image.size[:2]

        new_w, new_h = self.output_size, self.output_size

        new_w, new_h = int(new_w), int(new_h)

        img = image.resize([new_h, new_w])

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        for i in range(len(segment)):
            segment[i][0::2] = np.multiply(segment[i][0::2], new_w / w / 4)
            segment[i][1::2] = np.multiply(segment[i][1::2], new_h / h / 4)
            keypoints[i][0::3] = np.multiply(keypoints[i][0::3], new_w / w / 4)
            keypoints[i][1::3] = np.multiply(keypoints[i][1::3], new_h / h / 4)

        return {'image': img, 'segment': segment, 'keypoints': keypoints}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']
        if random.random() < self.p:
            w, h = image.size[:2]
            image = transforms_F.hflip(image)
            for i in range(len(segment)):
                segment[i][0::2] = np.abs(np.subtract(segment[i][0::2], w / 4))
                # segment[i][1::2] = np.abs(np.subtract(segment[i][1::2], h))
                keypoints[i][0::3] = np.abs(np.subtract(keypoints[i][0::3], w / 4))
                # keypoints[i][1::3] = np.abs(np.subtract(keypoints[i][0::3], w))
        return {'image': image, 'segment': segment, 'keypoints': keypoints}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segment, keypoints = sample['image'], sample['segment'], sample['keypoints']

        w, h = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = Image.fromarray(np.array(image)[top: top + new_h, left: left + new_w])

        for i in range(len(segment)):
            segment[i][0::2] = np.maximum(np.subtract(segment[i][0::2], left / 4), 0)
            segment[i][1::2] = np.maximum(np.subtract(segment[i][1::2], top / 4), 0)
            keypoints[i][0::3] = np.maximum(np.subtract(keypoints[i][0::3], left / 4), 0)
            keypoints[i][1::3] = np.maximum(np.subtract(keypoints[i][1::3], top / 4), 0)

        return {'image': img, 'segment': segment, 'keypoints': keypoints}


class myImageDataset_COCO(data.Dataset):
    def __init__(self, anno, image_dir, transform=None):
        'Initialization'
        self.anno = COCO(anno)
        self.image_dir = image_dir
        self.lists = self.anno.getImgIds(catIds=self.anno.getCatIds())
        self.transform = transform

    def __len__(self):
        return len(self.lists)
        # return 1000

    def __getitem__(self, index):
        list = self.lists[index]
        sample = {}
        image_name = self.anno.loadImgs(list)[0]['file_name']
        image_path = path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        sample['image'] = image
        # plt.imshow(image)
        # plt.show()
        label_id = self.anno.getAnnIds(list)
        labels = self.anno.loadAnns(label_id)

        segment_array = []
        keypoints_array = []
        draw = ImageDraw.Draw(image)
        for label in labels:
            try:
                segment = label['segmentation'][0]
                segment_array.append(segment)
                # seg_x = segment[0::2]
                # seg_y = segment[1::2]
                # draw.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
                # plt.imshow(image)
                # plt.show()
                sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
                kp = np.array(label['keypoints'])
                keypoints_array.append(kp)
            except KeyError:
                pass

        sample['keypoints'] = keypoints_array
        sample['segment'] = segment_array
        sample = Rescale(320)(sample)
        sample = RandomCrop(inputsize)(sample)
        sample = RandomHorizontalFlip()(sample)

        # Label_map_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        # Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        # Label_map_background = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        # Label_map_background = Image.fromarray(Label_map_background, 'L')
        # draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        # draw_background = ImageDraw.Draw(Label_map_background)
        #
        # draw = ImageDraw.Draw(sample['image'])
        # for i in range(len(sample['segment'])):
        #     segment = sample['segment'][i]
        #     seg_x = np.array(segment[0::2]).astype(np.int)
        #     seg_y = np.array(segment[1::2]).astype(np.int)
        #     draw_background.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
        #     x = np.array(sample['keypoints'][i][0::3]).astype(np.int)
        #     y = np.array(sample['keypoints'][i][1::3]).astype(np.int)
        #     v = sample['keypoints'][i][2::3]
        #     for k in range(keypoints):
        #         if v[k] > 0:
        #             draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(sample['image'])
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(Label_map_background)
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(Label_map_keypoints)
        #     plt.show()
        #     print('esf')
        Label_map_skeleton = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_skeleton = Image.fromarray(Label_map_skeleton, 'L')
        Label_map_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_keypoints = Image.fromarray(Label_map_keypoints, 'L')
        Label_map_background = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_map_background = Image.fromarray(Label_map_background, 'L')
        draw_skeleton = ImageDraw.Draw(Label_map_skeleton)
        draw_keypoints = ImageDraw.Draw(Label_map_keypoints)
        draw_background = ImageDraw.Draw(Label_map_background)
        Gauss_map = np.zeros([17, int(inputsize / 4), int(inputsize / 4)])

        for i in range(len(sample['segment'])):
            segment = sample['segment'][i]
            seg_x = np.array(segment[0::2]).astype(np.int)
            seg_y = np.array(segment[1::2]).astype(np.int)
            draw_background.polygon(np.stack([seg_x, seg_y], axis=1).reshape([-1]).tolist(), fill='#010101')
            x = np.array(sample['keypoints'][i][0::3]).astype(np.int)
            y = np.array(sample['keypoints'][i][1::3]).astype(np.int)
            v = sample['keypoints'][i][2::3]
            sks = np.array(self.anno.loadCats(label['category_id'])[0]['skeleton']) - 1
            kp = np.array(label['keypoints'])
            for k in range(keypoints):
                if v[k] > 0:
                    sigma = 1
                    mask_x = np.matlib.repmat(x[k], int(inputsize / 4), int(inputsize / 4))
                    mask_y = np.matlib.repmat(y[k], int(inputsize / 4), int(inputsize / 4))

                    x1 = np.arange(int(inputsize / 4))
                    x_map = np.matlib.repmat(x1, int(inputsize / 4), 1)

                    y1 = np.arange(int(inputsize / 4))
                    y_map = np.matlib.repmat(y1, int(inputsize / 4), 1)
                    y_map = np.transpose(y_map)

                    temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                    Gauss_map[k, :, :] += np.exp(-temp)
                    draw_keypoints.point(np.array([x[k], y[k]]).tolist(), 'rgb({}, {}, {})'.format(k + 1, k + 1, k + 1))
            for i, sk in enumerate(sks):
                if np.all(v[sk] > 0):
                    draw_skeleton.line(np.stack([x[sk], y[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        del draw_skeleton, draw_background
        # plt.subplot(1, 4, 1)
        # plt.imshow(sample['image'])
        # plt.subplot(1, 4, 2)
        # plt.imshow(np.array(Label_map_background))
        # plt.subplot(1, 4, 3)
        # plt.imshow(np.array(Label_map_skeleton.resize([256, 256])))
        # plt.subplot(1, 4, 4)
        # plt.imshow(np.array(Label_map_keypoints))
        # plt.show()
        #
        # print('esf')
        cm = ScalarMappable()

        image_after = self.transform(np.array(sample['image']) / 255)

        cm.set_array(np.array(Label_map_skeleton))
        result = cm.to_rgba(np.array(Label_map_skeleton.resize([256, 256])))[:, :, :3].swapaxes(0, 2)
        return image_after.float(), torch.Tensor(result).float()


class myImageDataset(data.Dataset):
    def __init__(self, imagedir, matdir):
        'Initialization'
        T = scipy.io.loadmat(matdir, squeeze_me=True, struct_as_record=False)
        M = T['RELEASE']
        self.M = M
        self.annots = M.annolist
        is_train = M.img_train
        lists = np.nonzero(is_train)
        single_person = np.zeros_like(self.annots)
        for i in lists[0]:
            anno = self.annots[i]
            rect = anno.annorect
            if isinstance(rect, scipy.io.matlab.mio5_params.mat_struct):
                if 'annopoints' in rect._fieldnames:
                    single_person[i] = 1

        self.list = np.nonzero(single_person)[0]
        self.imagedir = imagedir

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        anno = self.annots[self.list[index]]
        image_name = anno.image.name

        image = Image.open(path.join(self.imagedir, image_name)).convert('RGB')
        w, h = image.size
        image = image.resize([inputsize, inputsize])

        rect = anno.annorect

        points = rect.annopoints.point
        points_rect = np.zeros([keypoints, 3])
        for point in points:
            if point.is_visible == 0:
                is_visible = 0
            else:
                is_visible = 1
            points_rect[point.id] = [point.x, point.y, is_visible]


        Label_keypoints = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_keypoints = Image.fromarray(Label_keypoints, 'L')
        draw_keypoints = ImageDraw.Draw(Label_keypoints)

        Label_skeleton = np.zeros([int(inputsize / 4), int(inputsize / 4)])
        Label_skeleton = Image.fromarray(Label_skeleton, 'L')
        draw_skeleton = ImageDraw.Draw(Label_skeleton)

        xs = points_rect[:, 0] * inputsize / w / 4
        ys = points_rect[:, 1] * inputsize / h / 4
        v = points_rect[:, 2]

        for i in range(keypoints):
            if v[i] > 0:
                draw_keypoints.point([xs[i], ys[i]], fill='rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        for i, sk in enumerate(sks):
            if np.all(v[sk]) > 0:
                draw_skeleton.line(np.stack([xs[sk], ys[sk]], axis=1).reshape([-1]).tolist(),
                                       'rgb({}, {}, {})'.format(i + 1, i + 1, i + 1))
        cm = ScalarMappable(norm=Normalize(0, 15))
        cm.set_array(np.array(Label_skeleton))
        result = cm.to_rgba(np.array(Label_skeleton.resize([256, 256])))[:, :, :3]
        # plt.subplot(1, 3, 1)
        # plt.imshow(image)
        # plt.subplot(1, 3, 2)
        # plt.imshow(np.array(Label_keypoints))
        # plt.subplot(1, 3 ,3)
        # plt.imshow(result)
        # plt.show()
        return transforms.ToTensor()(image).float(), transforms.ToTensor()(result).float()


class myImageDataset_one_punch(data.Dataset):
    def __init__(self, image_dir, skeleton_dir):
        'Initialization'
        self.image_dir = image_dir
        self.skeleton_dir = skeleton_dir
        self.list = os.listdir(skeleton_dir)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_name = self.list[index]
        image = Image.open(path.join(self.image_dir, image_name)).resize([256, 256])
        skeleton_image = Image.open(path.join(self.skeleton_dir, image_name)).resize([256, 256])
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return mytransform(image), mytransform(skeleton_image)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, numIn, numOut, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.numIn = numIn
        self.numOut = numOut
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = spectral_norm(nn.Conv2d(numIn, int(numOut / 2), 1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = spectral_norm(nn.Conv2d(int(numOut / 2), int(numOut / 2), 3, stride, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.conv3 = spectral_norm(nn.Conv2d(int(numOut / 2), numOut, 1, 1))
        self.downsaple = nn.Sequential(
            spectral_norm(nn.Conv2d(numIn, numOut, 1, stride=stride, bias=False))
        )

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.conv3(x)
        if self.stride != 1 | self.numIn != self.numOut:
            residual = self.downsaple(residual)
        out += residual
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = spectral_norm(nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                                   stride=1, padding=padding, dilation=dilation, bias=False))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.atrous_conv(x)

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
                                             spectral_norm(nn.Conv2d(inplanes, 256, 1, stride=1, bias=False)),
                                             nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1280, 256, 1, bias=False)),
            nn.LeakyReLU(0.2)
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


class Generator(nn.Module):
    """Generator."""

    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.inner = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer10 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256 * 2, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer11 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256 * 2, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer12 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256 * 2, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer13 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128 * 2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64 * 2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        down1 = self.layer1(x)                              #64,128,128
        down2 = self.layer2(down1)                          #128,64,64
        down3 = self.layer3(down2)                          #256,32,32
        # down3 = self.attn1(down3)                           #256,32,32
        down4 = self.layer4(down3)                          #256,16,16
        down5 = self.layer5(down4)                          #256,8,8
        out = self.inner(down5)                             #256,8,8
        # out = self.aspp(down5)                              #256,8,8
        out = self.layer10(torch.cat([out, down5], 1))      #256,16,16
        out = self.layer11(torch.cat([out, down4], 1))      #256,32,32
        # out = self.attn2(out)                               #256,32,32
        out = self.layer12(torch.cat([out, down3], 1))      #128,64,64
        out = self.layer13(torch.cat([out, down2], 1))      #64,128,128
        out = self.output(torch.cat([out, down1], 1))         #3,256,256
        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(6, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, 3, 2, 1)),
            nn.LeakyReLU(0.2)
        )
        # self.attn1 = Self_Attn(256, nn.LeakyReLU(0.2))
        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, 3, 2, 1)),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.attn1(x)
        x = self.layer5(x)
        out = self.output(x)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        ngf = output_nc * 2
        unet_block = UnetSkipConnectionBlock(ngf * 512, ngf * 1024, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(ngf * 128, ngf * 256, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 32, ngf * 64, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, 2 * inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc * 2)

        if outermost:
            upconv = nn.PixelShuffle(2)
            down = [downconv]
            up = [upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.PixelShuffle(2)
            down = [downrelu, downconv]
            up = [upconv]
            model = down + up
        else:
            upconv = nn.PixelShuffle(2)
            down = [downrelu, downconv, downnorm]
            up = [upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == '__main__':
    if mode == 'train':
        if write == True:
            writer = SummaryWriter('run/' + save_model_name)
        # anno = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
        # image_dir = '/data/COCO/COCO2017/train2017'
        # mytransform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        loss = nn.BCEWithLogitsLoss().cuda()
        loss_L1 = nn.L1Loss().cuda()
        netG = Generator().cuda()
        netD = Discriminator().cuda()
        optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0, 0.9))
        optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0, 0.9))
        # netG, optG = amp.initialize(netG, optG, opt_level="O1")
        # netD, optD = amp.initialize(netD, optD, opt_level='O1')
        netG.train()
        netD.train()
        train_image_dataloader = data.DataLoader(myImageDataset_one_punch('/data/one punch/oniginal_image/', '/data/one punch/skeleton/'), batch_size, shuffle=True, num_workers=16)
        if not os.path.isfile(load_model_name):
            epoch = 0
        else:
            state = torch.load(load_model_name)
            netG.load_state_dict(state['state_dict_G'])
            netD.load_state_dict(state['state_dict_D'])
            # optG.load_state_dict(state['optimizer_G'])
            # optD.load_state_dict(state['optimizer_D'])
            epoch = state['epoch']
        while epoch <= epochs:
            for i, [x_, y_skeleton] in enumerate(train_image_dataloader, 0):
                real_B, real_A = x_.cuda(), y_skeleton.cuda()
                fake_B = netG(real_A)

                optD.zero_grad()
                for p in netD.parameters():
                    p.requires_grad = True
                fake_AB = torch.cat([real_A, fake_B], dim=1)
                pred_fake = netD(fake_AB)
                fake_label = torch.Tensor([0]).cuda()
                fake_label = fake_label.expand_as(pred_fake)

                real_AB = torch.cat([real_A, real_B], dim=1)
                pred_real = netD(real_AB)
                real_label = torch.Tensor([1]).cuda()
                real_label = real_label.expand_as(pred_fake)

                ra_pred_real = pred_real - pred_fake.mean()
                ra_pred_fake = pred_fake - pred_real.mean()

                loss_D = (torch.mean((ra_pred_real - real_label) ** 2) + torch.mean(
                    (ra_pred_fake + real_label) ** 2)) / 2

                # loss_D = (loss_D_fake + loss_D_real)
                # with amp.scale_loss(loss_D, optD) as scaled_loss:
                #     scaled_loss.backward(retain_graph=True)
                loss_D.backward(retain_graph=True)
                optD.step()

                optG.zero_grad()
                for p in netD.parameters():
                    p.requires_grad = False
                fake_AB = torch.cat([real_A, fake_B], dim=1)
                pred_fake = netD(fake_AB)
                real_label = torch.Tensor([1]).cuda()
                real_label = real_label.expand_as(pred_fake)

                ra_pred_fake = pred_fake - pred_real.mean()
                ra_pred_real = pred_real - pred_fake.mean()


                # Generator loss (You may want to resample again from real and fake data)
                loss_G_GAN = (torch.mean((ra_pred_real + real_label) ** 2) + torch.mean(
                    (ra_pred_fake - real_label) ** 2)) / 2


                # loss_G_GAN = loss(ra_pred_fake, real_label)
                loss_G_L1 = loss_L1(fake_B, real_B)
                loss_G = loss_G_GAN + loss_G_L1
                # with amp.scale_loss(loss_G, optG) as scaled_loss:
                #     scaled_loss.backward()
                loss_G.backward()
                optG.step()
                if i % 50 == 0:
                    steps = i + len(train_image_dataloader) * epoch
                    if write == True:
                        writer.add_scalar('loss_G', loss_G, steps)
                        writer.add_scalar('loss_D', loss_D, steps)
                    print('[{}/{}][{}/{}] LossG: {} Loss_D: {}'.format(
                        epoch, epochs, i, len(train_image_dataloader), loss_G, loss_D))
                if i % 100 == 0:
                    if write == True:
                        steps = i + len(train_image_dataloader) * epoch
                        image_fake_B = torchvision.utils.make_grid(fake_B, normalize=True, range=(0, 1))
                        writer.add_image('fake_B', image_fake_B, steps)
                        image_real_B = torchvision.utils.make_grid(real_B, normalize=True, range=(0, 1))
                        writer.add_image('real_B,', image_real_B, steps)
            epoch += 1
            state = {
                'epoch': epoch,
                'state_dict_G': netG.state_dict(),
                'state_dict_D': netD.state_dict(),
                'optimizer_G': optG.state_dict(),
                'optimizer_D': optD.state_dict(),
            }
            torch.save(state, save_model_name)
    elif mode == 'test':
        netG = UnetGenerator(3, 3, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=False).eval().half().cuda()
        anno = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
        image_dir = '/data/COCO/COCO2017/train2017'
        mytransform = transforms.Compose([
            transforms.ToTensor()
        ])
        state = torch.load(load_model_name)
        netG.load_state_dict(state['state_dict_G'])
        train_image_dataloader = data.DataLoader(myImageDataset_COCO(anno, image_dir, transform=mytransform),
                                                 1, shuffle=True, num_workers=1)
        for i, [x_, y_skeleton] in enumerate(train_image_dataloader, 0):
            real_B, real_A = x_.cuda().half(), y_skeleton.cuda().half()
            fake_B = netG(real_A)
            plt.imshow(transforms.ToPILImage()(fake_B.cpu().float()[0]))
            plt.show()
            print('efs')
