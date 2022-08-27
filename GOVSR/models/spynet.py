import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# from util import load_checkpoint
from models.common import flow_warp
from mmcv.cnn import ConvModule
# from torch.nn import Conv2d as ConvModule
from mmcv.runner import load_checkpoint

class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained, requires_grad=False):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        params=list(self.parameters())
        pnum=0
        for p in params:
            l=1
            for j in p.shape:
                l*=j
            pnum+=l
        print(f'Number of parameters for SPyNet: {pnum}')

        for param in self.parameters():
            param.requires_grad = requires_grad
        print(f'SPyNet requires_grad = {requires_grad}')

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow

class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)



# class SPyNetBasicModule(nn.Module):
#     """Basic Module for SPyNet.
#     Paper:
#         Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
#     """

#     def __init__(self):
#         super().__init__()

#         self.basic_module = nn.Sequential(
#             ConvModule(
#                 in_channels=8,
#                 out_channels=32,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3),
#             nn.LeakyReLU(0.2,True),
#             ConvModule(
#                 in_channels=32,
#                 out_channels=64,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3),
#             nn.LeakyReLU(0.2,True),
#             ConvModule(
#                 in_channels=64,
#                 out_channels=32,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3),
#             nn.LeakyReLU(0.2,True),
#             ConvModule(
#                 in_channels=32,
#                 out_channels=16,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3),
#             nn.LeakyReLU(0.2,True),
#             ConvModule(
#                 in_channels=16,
#                 out_channels=2,
#                 kernel_size=7,
#                 stride=1,
#                 padding=3),
#             nn.LeakyReLU(0.2,True))

#     def forward(self, tensor_input):
#         """
#         Args:
#             tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
#                 8 channels contain:
#                 [reference image (3), neighbor image (3), initial flow (2)].
#         Returns:
#             Tensor: Refined flow with shape (b, 2, h, w)
#         """
#         return self.basic_module(tensor_input)