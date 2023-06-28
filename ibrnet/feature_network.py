# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import pdb


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 coarse_out_ch=32,
                 fine_out_ch=32,
                 norm_layer=None,
                 coarse_only=False
                 ):

        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.coarse_only = coarse_only
        if self.coarse_only:
            fine_out_ch = 0
        self.coarse_out_ch = coarse_out_ch
        self.fine_out_ch = fine_out_ch
        out_ch = coarse_out_ch + fine_out_ch

        # original
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='reflect')
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # pdb.set_trace() # x : [10, 3, 378, 504]
        x = self.relu(self.bn1(self.conv1(x))) # x : [10, 64, 189, 252]

        x1 = self.layer1(x) # x1 : [10, 64, 95, 126]
        x2 = self.layer2(x1) # x2 : [10, 128, 48, 63]
        x3 = self.layer3(x2) # [10, 256, 24, 32]

        x = self.upconv3(x3) # [10, 128, 48, 64]
        x = self.skipconnect(x2, x) # [10, 256, 48, 64]
        x = self.iconv3(x) # [10, 128, 48, 64]

        x = self.upconv2(x) # [10, 64, 96, 128]
        x = self.skipconnect(x1, x) # [10, 128, 96, 128]
        x = self.iconv2(x) # [10, 64, 96, 128]

        x_out = self.out_conv(x) # [10, 64, 96, 128]

        if self.coarse_only:
            x_coarse = x_out
            x_fine = None
        else:
            x_coarse = x_out[:, :self.coarse_out_ch, :] # [10, 32, 96, 128]
            x_fine = x_out[:, -self.fine_out_ch:, :] # [10, 32, 96, 128]
        
        return x_coarse, x_fine


# ------Fast_Fourier_Convolution_(FFC)--------------------------------------------

from ibrnet.ffc import FFC_BN_ACT, FFCResnetBlock, LearnableSpatialTransformWrapper, ConcatTupleLayer, get_activation
class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_upsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        head_model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        down_model = []
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            down_model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        body_model = []
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            body_model += [cur_resblock]

        body_model += [ConcatTupleLayer()]

        ### upsample
        up_model = []
        for i in range(n_upsampling):
            mult = 2 ** (n_downsampling - i) #0: 8, 1: 4, 2: 2
            up_model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        tail_model = []
        # pdb.set_trace()
        if out_ffc:
            tail_model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        tail_model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf*4, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            tail_model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.head_model = nn.Sequential(*head_model)
        self.down_model = nn.Sequential(*down_model)
        self.body_model = nn.Sequential(*body_model)
        self.up_model = nn.Sequential(*up_model)
        self.tail_model = nn.Sequential(*tail_model)

    def forward(self, input):
        
        n, c, h, w = input.shape
        h, w = int(math.ceil(h/16)*16), int(math.ceil(w/16)*16)
        out = F.interpolate(input, (h, w))
        
        out = self.head_model(out)      #[8, 64, 378, 504], 0
        out = self.down_model(out)      #[8, 154, 48, 63]， [8, 358, 48, 63]
        out = self.body_model(out)      #[512, 48, 63]， [512, 48, 63]
        out = self.up_model(out)        #[8, 256, 96, 126]
        out = self.tail_model(out)      #[8, 64, 96, 126]
        return out[:, :32, ...], out[:, 32:, ...]






# ------Flow_Guided_Feature--------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
import torchvision
# import flow_vis_torch

from mmcv.runner import load_checkpoint

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN, make_layer, flow_warp)
from mmedit.models.backbones.sr_backbones.basicvsr_net import (ResidualBlocksWithInputConv, SPyNet)

from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class MSBasicVSRPlusPlus(nn.Module):
    """MSBasicVSRPlusPlus network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 cpu_cache_length=100,
                 scale_factor=4,
                 joint_forward_backward=False,
                #  shift_flow=1
                 ): # TODO
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.scale_factor = scale_factor
        shift_flow = 1
        self.shift_flow = shift_flow
        self.joint_forward_backward = joint_forward_backward

        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self._norm_layer = nn.InstanceNorm2d
        self.down1 = nn.Conv2d(3, mid_channels, 3, 2, 1, bias=True)
        self.layer1 = ResidualBlocksWithInputConv(mid_channels, mid_channels, 5) #self._make_layer(BasicBlock, 64, 3, stride=2)  #
        self.down2 = nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True)
        self.layer2 = ResidualBlocksWithInputConv(mid_channels, mid_channels, 5)

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels * (shift_flow**2),
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            # if not self.joint_forward_backward:
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)
            # else:
            #     if i >= 2:
            #         self.backbone[module] = ResidualBlocksWithInputConv(
            #         (2 + i - 2) * mid_channels, mid_channels, num_blocks)
            #     else:
            #         self.backbone[module] = ResidualBlocksWithInputConv(
            #         (2 + i) * mid_channels, mid_channels, num_blocks)
        
        # upsampling module
        # if not self.joint_forward_backward:
        self.fusion = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        # else:
        if self.joint_forward_backward:
            self.fusion1 = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels//2, 5)
            # self.fusion2 = ResidualBlocksWithInputConv(3 * mid_channels, mid_channels, 5)

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                'Deformable alignment module is not added. '
                'Probably your CUDA is not configured correctly. DCN can only '
                'be used with CUDA enabled. Alignment is skipped now.')
            
        self.tail = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(64, 64, 3, 1, 1)
                                            )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def multi_propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, m, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, m, _, h, w = flows.size()
        flows = flows.view(n, t, -1, h, w)
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:

                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1[:, :2, ...].permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    for mi in range(m//2):
                        flow_n2[:,mi*2:(mi+1)*2,...] = flow_n1[:,mi*2:(mi+1)*2,...] + flow_warp(flow_n2[:,mi*2:(mi+1)*2,...], flow_n1[:,mi*2:(mi+1)*2,...].permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2[:, :2, ...].permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                # if i > 1:
                #     pdb.set_trace()
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                # gray_feat = torch.mean(cond_n1.squeeze(0), 0)
                # plt.imshow(gray_feat.cpu())
                # plt.savefig(str('denoise_dir/test_flow/conf_n1e.png'))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    # gray_feat = torch.mean(cond_n2.squeeze(0), 0)
                    # plt.imshow(gray_feat.cpu())
                    # plt.savefig(str('denoise_dir/test_flow/conf_n2e.png'))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)
            # pdb.set_trace()
            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]
            # pdb.set_trace()
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def propagate_2nd(self, feats, flows, flows_2nd, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                # gray_feat = torch.mean(cond_n1.squeeze(0), 0)
                # plt.imshow(gray_feat.cpu())
                # plt.savefig(str('denoise_dir/test_flow/conf_n1e.png'))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    if flow_idx[i] < flows_2nd.shape[1]:
                        flow_n2 = flows_2nd[:, flow_idx[i], :, :, :]
                    else:
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        if self.cpu_cache:
                            flow_n2 = flow_n2.cuda()
                        flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))

                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    # gray_feat = torch.mean(cond_n2.squeeze(0), 0)
                    # plt.imshow(gray_feat.cpu())
                    # plt.savefig(str('denoise_dir/test_flow/conf_n2e.png'))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def reconstruction(self, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, h, w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])
        
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, num_outputs):
            # pdb.set_trace()
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.fusion(hr)
            outputs.append(hr)

        return torch.stack(outputs, dim=1) #TODO + torch.stack(feats['spatial'], dim=1) 

    def joint_reconstruction(self, feats, stage):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, h, w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])
        
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        if ('backward_1' in feats) and ('forward_1' in feats):
            feats_backward_1, feats_forward_1 = feats['backward_1'].copy(), feats['forward_1'].copy() 
        if ('backward_2' in feats) and ('forward_2' in feats):
            feats_backward_2, feats_forward_2 = feats['backward_2'].copy(), feats['forward_2'].copy() 

        for i in range(0, num_outputs):
            hr = [feats[k].pop(0) for k in feats if ((k != 'spatial') and (k[-1] == stage))]
            # print(feats.keys(), stage, len(hr))
            # for k in feats:
            #     print(k, k[-1], stage, k[-1]== stage)

            hr_sp = feats['spatial'][mapping_idx[i]]
            hr.insert(0, hr_sp)
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()
            if stage == '1':
                hr = self.fusion1(hr) + hr_sp
            else:
                hr = self.fusion2(hr) + hr_sp
            outputs.append(hr)
        # pdb.set_trace()
        # features = torch.stack(outputs, dim=1) + torch.stack(feats['spatial'], dim=1)
        feats['spatial'] = outputs #[features[i:i+1] for i in range(0, num_outputs)]
        if ('backward_1' in feats) and ('forward_1' in feats):
            feats['backward_1'], feats['forward_1'] = feats_backward_1, feats_forward_1
        if ('backward_2' in feats) and ('forward_2' in feats):
            feats['backward_2'], feats['forward_2'] = feats_backward_2, feats_forward_2

        # print(feats.keys(), len(feats['forward_1']), len(feats['spatial']))
        return feats

    def forward(self, features, flows, flows2=None):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = features.size()
        # pdb.set_trace()
        base_features = self.layer2(self.down2(self.layer1(self.down1(features.view(-1, c, h, w)))))
        features = self.layer2(self.down2(self.layer1(self.down1(features.view(-1, c, h, w))))).view(n, t, -1, h//4, w//4)
        n, t, _, h, w = features.size()
        flows_forward, flows_backward = flows

        if flows2 is not None:
            flows_forward_2nd, flows_backward_2nd = flows2

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = features[:, i, :, :, :].cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats['spatial'] = [features[:, i, :, :, :] for i in range(0, t)]

        if not self.joint_forward_backward:
            # feature propgation
            for iter_ in [1, 2]:
                for direction in ['backward', 'forward']:
                    module = f'{direction}_{iter_}'

                    feats[module] = []
                    
                    if direction == 'backward':
                        flows = flows_backward
                        if flows2 is not None:
                            flows_2nd = flows_backward_2nd
                    elif flows_forward is not None:
                        flows = flows_forward
                        if flows2 is not None:
                            flows_2nd = flows_forward_2nd
                    else:
                        flows = flows_backward.flip(1)
                        if flows2 is not None:
                            flows_2nd = flows_backward_2nd.flip(1)

                    if flows2 is not None:
                        feats = self.propagate_2nd(feats, flows, flows_2nd, module)
                    else:
                        if flows.ndim==6:
                            feats = self.multi_propagate(feats, flows, module)
                        else:
                            feats = self.propagate(feats, flows, module)

                    if self.cpu_cache:
                        del flows
                        if flows2 is not None:
                            del flows2
                        torch.cuda.empty_cache()

            out_features = self.reconstruction(feats).view(n*t, -1, h, w)

        else:
            for iter_ in [1, 2]:
                for direction in ['backward', 'forward']:
                    module = f'{direction}_{iter_}'

                    feats[module] = []
                    
                    if direction == 'backward':
                        flows = flows_backward
                        if flows2 is not None:
                            flows_2nd = flows_backward_2nd
                    elif flows_forward is not None:
                        flows = flows_forward
                        if flows2 is not None:
                            flows_2nd = flows_forward_2nd
                    else:
                        flows = flows_backward.flip(1)
                        if flows2 is not None:
                            flows_2nd = flows_backward_2nd.flip(1)
                    
                    if flows2 is not None:
                        feats = self.propagate_2nd(feats, flows, flows_2nd, module)
                    else:
                        if flows.ndim==6:
                            feats = self.multi_propagate(feats, flows, module)
                        else:
                            feats = self.propagate(feats, flows, module)

                    if self.cpu_cache:
                        del flows
                        if flows2 is not None:
                            del flows2
                        torch.cuda.empty_cache()
                
                if iter_ == 1: 
                    feats = self.joint_reconstruction(feats, stage=str(iter_))  #.view(n*t, -1, h, w)
                
                # else:
                #     # pdb.set_trace()
                #     feats['spatial'] = feats['forward_1']
                
                # feats['spatial'] = [out_features[i:i+1] for i in range(0, t)]
                # if ('backward_1' in feats) and ('forward_1' in feats):
                #     del feats['backward_1'] 
                #     del feats['forward_1']
            out_features = self.reconstruction(feats).view(n*t, -1, h, w)
            out_features = self.tail(out_features)
            
        return out_features[:, :32, ...], out_features[:, 32:, ...] #TODO


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        if flow_1.shape[1] > 2:
            extra_feat = torch.cat([extra_feat, flow_1[:,:2,...], flow_2[:,:2,...]], dim=1)
        else:
            extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # if flow_2.max() > 0:
        #     pdb.set_trace()

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // flow_1.shape[1], 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // flow_1.shape[1], 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # flow_rgb = flow_vis_torch.flow_to_color(offset_2[:,:2])
        # torchvision.utils.save_image(flow_rgb, 'denoise_dir/test_flow/offset_2.png', normalize=True)

        # mask
        mask = torch.sigmoid(mask)
        
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    # feature_net = ResUNet(coarse_out_ch=32, fine_out_ch=32, coarse_only=False).cuda()
    feature_net = MSBasicVSRPlusPlus(mid_channels=64, num_blocks=7, max_residue_magnitude=10,
                            is_low_res_input=True, cpu_cache_length=100, scale_factor=2).cuda()
    
    print(feature_net(x))

# parser.add_argument('--coarse_feat_dim', type=int, default=32, help="2D feature dimension for coarse level")
# parser.add_argument('--fine_feat_dim', type=int, default=32, help="2D feature dimension for fine level")
# parser.add_argument('--num_source_views', type=int, default=10,
#                     help='the number of input source views for each target view')