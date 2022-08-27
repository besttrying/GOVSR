import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        # nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.constant_(m.bias, 0.0)

class cha_loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(cha_loss, self).__init__()
        self.eps=eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target)**2+self.eps**2
        out = torch.sqrt(diff)#.sum(dim=3,keepdim=True)
        loss=torch.mean(out)

        return loss

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, :, index]

    return it

def compute_flow(flownet, lrs, step = 1):
    """Compute optical flow using SPyNet for feature warping.
    Note that if the input is an mirror-extended sequence, 'flows_forward'
    is not needed, since it is equal to 'flows_backward.flip(1)'.
    Args:
        lrs (tensor): Input LR images with shape (n, t, c, h, w)
    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = lrs.size()
    lrs_1 = lrs[:, :-step, :, :, :].reshape(-1, c, h, w)
    lrs_2 = lrs[:, step:, :, :, :].reshape(-1, c, h, w)

    flows_backward = flownet(lrs_1, lrs_2).view(n, t - step, 2, h, w)

    flows_forward = flownet(lrs_2, lrs_1).view(n, t - step, 2, h, w)

    return flows_forward, flows_backward

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output