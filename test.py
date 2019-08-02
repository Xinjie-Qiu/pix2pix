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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


nModules = 2
nFeats = 256
nStack = 3
nKeypoint = 17
nSkeleton = 19
nOutChannels_0 = 2
nOutChannels_1 = nSkeleton + 1
nOutChannels_2 = nKeypoint
epochs = 2000
batch_size = 40
keypoints = 17
skeleton = 20
low_level_chennel = 48

inputsize = 256

threshold = 0.8

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
        self.list = self.list.sort()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_name = self.list[index]
        image = Image.open(path.join(self.image_dir, image_name)).resize([256, 256])
        skeleton_image = Image.open(path.join(self.skeleton_dir, image_name)).resize([256, 256])
        return transforms.ToTensor()(image).float(), transforms.ToTensor()(skeleton_image).float()


class myImageDataset_real_image(data.Dataset):
    def __init__(self, skeleton_dir):
        'Initialization'
        self.skeleton_dir = skeleton_dir
        self.list = os.listdir(skeleton_dir)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_name = self.list[index]
        skeleton_image = Image.open(path.join(self.skeleton_dir, image_name)).resize([256, 256])
        mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return mytransform(skeleton_image).float()


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


if __name__ == '__main__':

    netG = Generator().cuda()
    state = torch.load(load_model_name)
    netG.load_state_dict(state['state_dict_G'])
    netG.eval()
    test_image_dataloader = data.DataLoader(
        myImageDataset_real_image('/data/one punch/how people walk skeleton'), 1,
        shuffle=False, num_workers=1)
    index = 0
    for step, skeleton in enumerate(test_image_dataloader):
        b_skeleton = skeleton.cuda()
        image = netG(b_skeleton)
        for i in range(image.shape[0]):
            image_inner = image[i]
            image_inner = torchvision.utils.make_grid(image_inner, normalize=True, range=(0, 1))
            plt.imshow(transforms.ToPILImage()(image_inner.cpu()))
            plt.show()
            print('esfes')
            # image_inner.save('/data/one punch/fake_image/{}.png'.format(index))
            # index += 1
    print('esfe')
