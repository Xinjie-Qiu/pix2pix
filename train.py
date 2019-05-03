import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import json
from pycocotools.coco import COCO
from os import path
import math
import torch.nn.functional as F
from scipy import ndimage
from numpy import matlib
from torch.optim import lr_scheduler
from apex import amp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
matplotlib.use('TkAgg')
from torch.nn.modules import loss
import random
from torchvision.transforms import functional as transforms_F
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
import torch.utils.data as data
from torch import optim
import os

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
epochs = 51
batch_size = 32
keypoints = 17
skeleton = 20
low_level_chennel = 48

inputsize = 256

threshold = 0.8

save_model_name = 'params_1.pkl'
load_model_name = 'params_1.pkl'

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

        image_after = self.transform(sample['image'])

        cm.set_array(np.array(Label_map_skeleton))
        result = cm.to_rgba(np.array(Label_map_skeleton.resize([256, 256])))[:, :, :3].swapaxes(0, 2)
        return image_after, torch.Tensor(result)


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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
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
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

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
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == '__main__':
    writer = SummaryWriter('run/' + save_model_name)
    anno = '/data/COCO/COCO2017/annotations_trainval2017/annotations/person_keypoints_train2017.json'
    image_dir = '/data/COCO/COCO2017/train2017'
    mytransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    loss = nn.MSELoss().cuda()
    loss_L1 = nn.L1Loss().cuda()
    netG = UnetGenerator(3, 3, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=False).cuda()
    netD = NLayerDiscriminator(6, 64, n_layers=3, norm_layer=nn.BatchNorm2d).cuda()
    optD = optim.Adam(netD.parameters(), lr=1e-4)
    optG = optim.Adam(netG.parameters(), lr=1e-4)
    train_image_dataloader = data.DataLoader(myImageDataset_COCO(anno, image_dir, transform=mytransform), batch_size, shuffle=True, num_workers=8)
    if not os.path.isfile(load_model_name):
        epoch = 0
    else:
        state = torch.load(load_model_name)
        netG.load_state_dict(state['state_dict_G'])
        netD.load_state_dict(state['state_dict_D'])
        optG.load_state_dict(state['optimizer_G'])
        optD.load_state_dict(state['optimizer_D'])
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
            loss_D_fake = loss(pred_fake, fake_label)

            real_AB = torch.cat([real_A, real_B], dim=1)
            pred_real = netD(real_AB)
            real_label = torch.Tensor([1]).cuda()
            real_label = real_label.expand_as(pred_fake)
            loss_D_real = loss(pred_real, real_label)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward(retain_graph=True)
            optD.step()

            optG.zero_grad()
            for p in netD.parameters():
                p.requires_grad = False
            fake_AB = torch.cat([real_A, fake_B], dim=1)
            pred_fake = netD(fake_AB)
            real_label = torch.Tensor([1]).cuda()
            real_label = real_label.expand_as(pred_fake)
            loss_G_GAN = loss(pred_fake, real_label)
            loss_G_L1 = loss_L1(fake_B, real_B)
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()
            if i % 50 == 0:
                steps = i + len(train_image_dataloader) * epoch
                writer.add_scalar('loss_G', loss_G, steps)
                writer.add_scalar('loss_D', loss_D, steps)
                print('[{}/{}][{}/{}] LossG: {} Loss_D: {}'.format(
                    epoch, epochs, i, len(train_image_dataloader), loss_G, loss_D))
        epoch += 1
        state = {
            'epoch': epoch,
            'state_dict_G': netG.state_dict(),
            'state_dict_D': netD.state_dict(),
            'optimizer_G': optG.state_dict(),
            'optimizer_D': optD.state_dict(),
        }
        torch.save(state, save_model_name)

    x = torch.rand([4, 3, 256, 256]).cuda()

    fake = netG.forward(x)
    result = torch.cat([x, fake], dim=1)
    result = netD(result)

    print('efsdf')