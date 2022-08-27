import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from models.common import generate_it, compute_flow, flow_warp
from models.spynet import SPyNet
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmedit.models.common import PixelShufflePack, ResidualBlockNoBN, default_init_weights, make_layer
from mmedit.models.backbones.sr_backbones.basicvsr_net import ResidualBlocksWithInputConv

class UPSCALE(nn.Module): 
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.1,True)):
        super(UPSCALE, self).__init__()
        # level = int(math.log(scale) / math.log(2))
        body = []
        body.append(nn.Conv2d(basic_feature, 128, 3, 1, 3//2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(32, 128, 3, 1, 3//2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(32, 32, 3, 1, 3//2))
        body.append(act)
        body.append(nn.Conv2d(32, 3, 3, 1, 3//2))
        # for i in range(level):
        #     body.append(nn.Conv2d(basic_feature, 3 * 4 ** (level-i), 3, 1, 3//2))
        #     if i != level - 1:
        #         body.append(act)
        #     body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)

class SimpleUpscale(nn.Module): 
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.1,True)):
        super(SimpleUpscale, self).__init__()
        self.bf = basic_feature
        self.scale = scale
        self.act = act
        self.upscale = UPSCALE(basic_feature, scale, act)
        self.img_upsample = nn.Upsample(
            scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, lqs, feats, start, end):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        # for k in feats.keys():
        #     print(k, len(feats[k]), feats[k][0].size())

        keys = list(feats.keys())
        for i in range(0, lqs.size(1)):
            # hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            # hr.insert(0, feats['spatial'][mapping_idx[i]])
            # hr = [feats[k].pop(0) for k in feats] # tested same
            # hr = [feats[k][i] for k in feats]     # tested same
            
            # hr = torch.cat(hr, dim=1)
            hr = feats[keys[-1]][i]

            hr = self.upscale(hr)
            hr += self.img_upsample(lqs[:, i, :, :, :])
            outputs.append(hr)

        return torch.stack(outputs, dim=2)

class BASICVSR_UPSCALE(nn.Module): 
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.1,True)):
        super(BASICVSR_UPSCALE, self).__init__()
        self.bf = basic_feature
        self.scale = scale
        self.act = act
        self.reconstruction = ResidualBlocksWithInputConv(
            basic_feature, basic_feature, 1)
        self.upsample1 = PixelShufflePack(
            basic_feature, basic_feature, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            basic_feature, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, lqs, feats, start, end):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        # for k in feats.keys():
        #     print(k, len(feats[k]), feats[k][0].size())

        keys = list(feats.keys())
        for i in range(0, lqs.size(1)):
            # hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            # hr.insert(0, feats['spatial'][mapping_idx[i]])
            # hr = torch.cat(hr, dim=1)
            hr = feats[keys[-1]][i]
            # if self.cpu_cache:
            #     hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.act(self.upsample1(hr))
            hr = self.act(self.upsample2(hr))
            hr = self.act(self.conv_hr(hr))
            hr = self.conv_last(hr)
            # if self.is_low_res_input:
            #     hr += self.img_upsample(lqs[:, i, :, :, :])
            # else:
            #     hr += lqs[:, i, :, :, :]
            hr += self.img_upsample(lqs[:, i, :, :, :])

            # if self.cpu_cache:
            #     hr = hr.cpu()
            #     torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=2)

class PFRBsWithInOut(nn.Module):
    def __init__(self, in_c=64, out_c=64, basic_feature=64, num_channel=7, num_blocks=1, act=torch.nn.LeakyReLU(0.2,True), mode = 'group'):
        super(PFRBsWithInOut, self).__init__()
        self.nc = num_channel
        self.act = act
        self.mode = mode
        self.conv_in = nn.Conv2d(in_c, basic_feature, 1, 1, 1//2)
        # if 'group' in mode:
        #     self.pfrbs = nn.Sequential(*[PFRBByGroupConv(basic_feature, num_channel, act) for _ in range(num_blocks)])
        # else:
        #     self.pfrbs = nn.Sequential(*[PFRB(basic_feature // num_channel, num_channel, act) for _ in range(num_blocks)])
        self.pfrbs = nn.Sequential(*[PFRB(basic_feature // num_channel, num_channel, act) for _ in range(num_blocks)])
        # self.pfrbs = nn.Sequential(*[PFRBByGroupConv(basic_feature, num_channel, act) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(basic_feature, out_c, 1, 1, 1//2)

        for m in self.modules():
            default_init_weights(m, 0.1)

    def forward(self, x):
        x0 = self.act(self.conv_in(x))
        # if 'group' not in self.mode:
        #     x0 = torch.chunk(x0, self.nc, dim = 1)
        x0 = torch.chunk(x0, self.nc, dim = 1)
        x0 = self.pfrbs(x0)
        x0 = torch.cat(x0, dim = 1)
        # if 'group' not in self.mode:
        #     x0 = torch.cat(x0, dim = 1)
        x0 = self.act(self.conv_out(x0))
        return x0

class PFRBByGroupConv(nn.Module): 
    def __init__(self, basic_feature=64, num_channel=3, act=torch.nn.LeakyReLU(0.2,True)):
        super(PFRBByGroupConv, self).__init__()
        self.bf = basic_feature
        self.nc = num_channel
        self.act = act
        self.conv0 = nn.Conv2d(self.bf, self.bf, 3, 1, 3//2, groups = self.nc)
        self.conv1 = nn.Conv2d(self.bf, self.bf // self.nc, 1, 1, 1 // 2, groups = 1)
        self.conv2 = nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3 // 2, groups = self.nc)
    
    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x0_split = torch.chunk(x0, self.nc, dim = 1)
        x1 = self.act(self.conv1(x0))
        x2 = [[_, x1] for _ in x0_split]
        x2 = torch.cat(sum(x2, []), dim=1)
        x2 = self.act(self.conv2(x2))

        return x + x2

class PFRB(nn.Module): 
    def __init__(self, basic_feature=64, num_channel=7, act=torch.nn.LeakyReLU(0.2,True)):
        super(PFRB, self).__init__()
        self.bf = basic_feature
        self.nc = num_channel
        self.act = act
        self.conv0=nn.Sequential(*[nn.Conv2d(self.bf, self.bf, 3, 1, 3//2, dilation=1) for _ in range(num_channel)])
        self.conv1=nn.Conv2d(self.bf * num_channel, self.bf, 1, 1, 1//2)
        self.conv2=nn.Sequential(*[nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3//2, dilation=1) for _ in range(num_channel)])
    
    def forward(self, x):
        x1=[self.act(self.conv0[i](x[i])) for i in range(self.nc)]
        merge=torch.cat(x1,1)
        base=self.act(self.conv1(merge))
        x2=[torch.cat([base,i],1) for i in x1]
        x2=[self.act(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i],x2[i]) for i in range(self.nc)]

class ResBlock(nn.Module):
    def __init__(self, nb = 1, bf=64, act = nn.LeakyReLU(0.2,True)):
        super(ResBlock, self).__init__()
        self.nb = nb
        self.bf = bf
        self.act = act
        self.conv = nn.Sequential(*[nn.Conv2d(bf, bf, 1, 1, 1//2, groups=1) for _ in range(nb)])

    def forward(self, x):
        for i in range(self.nb):
            x = x + self.act(self.conv[i](x))
        return x

class LOCAL_NL(nn.Module):
    def __init__(self, in_c=64, out_c=64, mid_c=96, bf=16, act = nn.LeakyReLU(0.2,True)):
        super(LOCAL_NL, self).__init__()
        self.bf = bf
        self.act = act
        self.conv_in = PFRBsWithInOut(in_c, mid_c, mid_c, 3, 1, act)#nn.Conv2d(in_c, bf, 3, 1, 3//2)
        self.g = nn.Conv2d(mid_c, bf, 1, 1, 1//2)
        # self.theta = nn.Conv2d(in_c, bf, 1, 1, 1//2)
        self.phi = nn.Conv2d(mid_c, bf, 1, 1, 1//2)
        self.w = nn.Conv2d(bf, mid_c, 1, 1, 1//2)
        self.conv_out = PFRBsWithInOut(mid_c, out_c, mid_c, 3, 1, act)#nn.Conv2d(bf, out_c, 3, 1, 3//2)

        constant_init(self.conv_out, val=0, bias=0)
        

    def forward(self, x, k=5, pos=0):
        # n = len(x.shape)
        # if n == 3:
        #     x = x.unsqueeze(0) # convert to (B, C, H, W)
        # elif n == 4:
        #     # B, C, H, W = x.shape
        #     pass
        # if n == 5:
        #     B, C, T, H, W = x.shape
        #     x = x.view(B, -1, H, W)
        #     C = C * T
        x = self.conv_in(x)
        g_x = self.g(x)
        # theta_x = self.theta(x)
        phi_x = self.phi(x)

        B, C, H, W = phi_x.shape

        att = F.unfold(phi_x, k, padding = k // 2)
        # print(phi_x.shape, att.shape)
        att = att.view(B, C, -1, H * W)

        att = torch.sum(att, 1, keepdims= False)                        # (B, k*k, HW)
        att = F.softmax(att, dim = 1)

        if pos > 0:
            att, idx1 = torch.sort(att, dim=1, descending=True)#, stable=True)
            att = att[:, :pos]

        att_g = F.unfold(g_x, k, padding = k // 2)
        att_g = att_g.view(B, C, -1, H * W)
        att_g = torch.mean(att_g, -1) # (B, C, k*k)

        if pos > 0:
            att_g, idx1 = torch.sort(att_g, dim=1, descending=True)#, stable=True)
            att_g = att_g[:, :, :pos]

        mat = torch.matmul(att_g, att).view(B, C, H, W)

        return self.conv_out(self.w(mat) + x) #self.tail(self.w(mat) + x)

class UNIT(nn.Module): 
    def __init__(self, kind='successor', basic_feature=64, mid_c = 114, num_channel=3, num_b=5, alignment_filter = 144, scale=4, act=nn.LeakyReLU(0.1,True), lnl=None, head=None, merge=None, upscale = None):
        super(UNIT, self).__init__()
        self.bf = basic_feature
        self.nf = num_channel
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind

        # inter_f = 3 * self.nf + self.bf if kind != 'successor' else self.nf * (3 + self.bf)
        self.head = head
        # self.conv1 = nn.Conv2d(self.nf * self.bf, self.nf * self.bf, 1, 1, 1//2)
        self.lnl = lnl # LOCAL_NL(self.nf * self.bf, 16)

        # self.blocks = nn.Sequential(*[PFRB(self.bf, 3, act) for i in range(num_b)])
        # if self.kind != 'successor':
        #     self.conv0 = nn.Conv2d(3, self.bf, 3, 1, 3//2)
        # self.conv_feat = nn.Conv2d(self.bf * 3, self.bf * 2, 3, 1, 3//2)
        self.ei_align = ExplicitImplicitAlignment(2 * basic_feature, basic_feature, 3, padding=1, deform_groups=16, max_residue_magnitude = 10, act = act, alignment_filter = alignment_filter)
        self.backbone = PFRBsWithInOut(basic_feature * 3 + 9, basic_feature, mid_c, num_channel, num_b, act, mode = 'list') # ResidualBlocksWithInputConv(self.bf * 3, self.bf, 4)##
        # nn.Sequential(*[nn.Conv2d(self.bf * (num_b + 2), self.bf , 3, 1, 3//2), 
        #                                 self.act, 
        #                                 nn.Conv2d(self.bf, self.bf, 3, 1, 3//2), 
        #                                 self.act,
        #                                 nn.Conv2d(self.bf, self.bf, 3, 1, 3//2), 
        #                                 self.act])
        self.merge = merge #nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3//2)
        self.upscale = upscale
        
        print(kind, num_b)
        # if kind != 'successor':
        #     for param in self.parameters():
        #         param.requires_grad = False
        
    
    def forward(self, lqs, feats, flows_forward, flows_backward, mode):
        B, T, _, H, W = flows_forward.size()

        T = len(feats['spatial'])

        frame_idx = list(range(T))

        if 'backward' in mode:
            frame_idx = frame_idx[::-1]

        ht_past = flows_forward.new_zeros(B, self.bf, H, W)
        keys = list(feats.keys())
        # print(keys)

        for idx, t in enumerate(frame_idx):
            lq_past = lqs[:, t-1] if t > 0 else lqs[:, 0]
            lq_current = lqs[:, t]
            lq_future = lqs[:, t+1] if t < T - 1 else lqs[:, -1]
            ht_current = feats[keys[-2]][t] #feats['spatial'][t]
            if 'backward' in mode:
                ht_future = feats[keys[-2]][t - 1] if t > 0 else flows_forward.new_zeros(B, self.bf, H, W)
            else:
                ht_future = feats[keys[-2]][t + 1] if t < T - 1 else flows_forward.new_zeros(B, self.bf, H, W)
            # if self.cpu_cache:
            #     feat_current = feat_current.cuda()
            #     feat_prop = feat_prop.cuda()
            # second-order deformable alignment

            flow_past = flows_forward[:, t - 1, :, :, :] if t > 0 else flows_forward.new_zeros(B, 2, H, W)
            flow_future = flows_backward[:, t, :, :, :] if t < T - 1 else flows_forward.new_zeros(B, 2, H, W)
            if 'backward' in mode:
                flow_past, flow_future = flow_future, flow_past
                lq_past, lq_future = lq_future, lq_past
            ht_past_2current = flow_warp(ht_past, flow_past.permute(0, 2, 3, 1))
            ht_future_2current = flow_warp(ht_future, flow_future.permute(0, 2, 3, 1))
            
            if idx > 0: #and self.is_with_alignment:
                # flow-guided deformable convolution
                cond = torch.cat([ht_past_2current, ht_current, ht_future_2current], dim=1)
                ht_past = torch.cat([ht_past, ht_future], dim=1)
                ht_past = self.ei_align(ht_past, cond, flow_past, flow_future)

            # concatenate and residual blocks
            # hts = [feats[k][t] for k in keys[1: -2]]
            hts = [ht_current, ht_past, ht_future_2current, lq_current, lq_past, lq_future]
            # hts = [feats[k][t] for k in keys[:-1]] + [ht_past, ht_future_2current]
            # feat = [ht_current] + [
            #     feats[k][idx]
            #     for k in feats if k not in ['spatial', mode]
            # ] + [ht_past]
            # if self.cpu_cache:
            #     feat = [f.cuda() for f in feat]

            hts = torch.cat(hts, dim=1)
            ht_past = ht_past + self.backbone(hts)
            feats[mode].append(ht_past)
            # print(mode, i, feat.shape, feat_prop.shape)
            # torch.Size([3, 128:320:64, 64, 64]) torch.Size([3, 64, 64, 64])

            # if self.cpu_cache:
            #     feats[mode][-1] = feats[mode][-1].cpu()
            #     torch.cuda.empty_cache()

        if 'backward' in mode:
            feats[mode] = feats[mode][::-1]

        return feats

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.bf = 64#config.model.basic_filter
        self.rb_filter = config.model.rb_filter
        self.alignment_filter = config.model.alignment_filter
        self.num_pb = config.model.num_pb
        self.num_sb = config.model.num_sb
        self.scale = config.model.scale
        self.nf = config.model.num_frame
        self.kind = config.model.kind     #local or global
        self.act = nn.LeakyReLU(0.1, True) #torch.nn.ReLU(True)
        self.lnl = None #LOCAL_NL(self.nf * self.bf, 16, self.act)
        self.frame2feature = nn.Sequential(*[nn.Conv2d(3, self.bf, 3, 1, 3//2), self.act])  # ResidualBlocksWithInputConv(3, self.bf, 1)
        self.head = None#nn.Conv2d(self.nf * (3 + self.bf), self.nf * self.bf, 1, 1, 1//2)
        self.merge = None#nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3//2)
        self.upscale = SimpleUpscale(self.bf, self.scale, self.act) #BASICVSR_UPSCALE(self.bf, self.scale, self.act) ## #  #UPSCALE(self.bf * 3, self.scale, self.act)
        
        direction = ['backward', 'forward']
        iteration = 1
        self.mode = [[f'{d}_{i}' for d in direction] for i in range(iteration)]
        self.mode = sum(self.mode, [])
        
        print(self.mode)
        self.units = []
        self.units.append(UNIT('precursor', self.bf, self.rb_filter, 3, 4, self.alignment_filter, self.scale, self.act, self.lnl, self.head, self.merge, self.upscale))
        self.units.append(UNIT('successor', self.bf, self.rb_filter, 3, 2, self.alignment_filter, self.scale, self.act, self.lnl, self.head, self.merge, self.upscale))
        self.units = nn.Sequential(*self.units)
        # self.units = nn.Sequential(*[UNIT('precursor', self.bf, self.nf, idx, self.scale, self.act, self.lnl, self.head, self.merge, self.upscale) for idx, _ in enumerate(self.mode)])

        print(self.kind, '{}+{}'.format(self.num_pb, self.num_sb))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                m.weight.data.normal_(0.0, 0.02)

        params=list(self.parameters())
        pnum=0
        for p in params:
            l=1
            for j in p.shape:
                l*=j
            pnum+=l
        print('Number of parameters {}'.format(pnum))

        self.spynet = SPyNet(pretrained = 'checkpoint/spynet/spynet_20210409-c6c1bd09.pth', requires_grad = True)
        

    @autocast()
    def forward(self, x, start=0):
        B, C, T, H, W = x.shape
        lqs = x.permute(0, 2, 1, 3, 4).contiguous() #B, T, C, H, W
        start = max(0, start)
        end = T - start

        sr_all =  []
        pre_sr_all =  []
        pre_ht_all = []

        feats = {}
        
        flows_forward, flows_backward = compute_flow(self.spynet, lqs)
        # flows_forward = flows_forward.permute(0, 1, 3, 4, 2)
        # flows_backward = flows_backward.permute(0, 1, 3, 4, 2)  #(B, T, H, W, 2)
        # frames_feature = self.act(self.frame2feature(lqs.view(-1, C, H, W))).view(B, T, -1, H, W)
        frames_feature = self.frame2feature(lqs.view(-1, C, H, W)).view(B, T, -1, H, W)
        feats['spatial'] = frames_feature = [frames_feature[:, _] for _ in range(T)]
        
        # frames_warp_forward = [x[:, :, 0]]
        # frames_warp_backward = []
        # for i in range(f - 1):
        #     frames_warp_forward.append(flow_warp(x[:, :, i], flows_forward[:, i]))
        #     frames_warp_backward.append(flow_warp(x[:, :, i + 1], flows_backward[:, i]))
        # frames_warp_backward.append(x[:, :, -1])

        for idx, mode in enumerate(self.mode):
            feats[mode] = []
            # if 'backward' in mode:
            #     flows = flows_backward
            # elif flows_forward is not None:
            #     flows = flows_forward
            feats = self.units[idx](lqs, feats, flows_forward, flows_backward, mode)

        return [self.upscale(lqs, feats, start, end)]



class ExplicitImplicitAlignment(ModulatedDeformConv2d):
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
        act = kwargs.pop('act', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        alignment_filter = kwargs.pop('alignment_filter', 144)
        
        super(ExplicitImplicitAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = LOCAL_NL(3 * self.out_channels + 4, 27 * self.deform_groups, alignment_filter, 16, act)
        # nn.Sequential(
        #     nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
        #     self.act,
        #     # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
        #     # self.act,
        #     # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
        #     # self.act,
        #     nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        # )

        # self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        # flow_1 = flow_1.permute(0, 3, 1, 2)
        # flow_2 = flow_2.permute(0, 3, 1, 2)
        # print(x.shape, extra_feat.shape, flow_1.shape, flow_2.shape)
        # torch.Size([3, 128, 64, 64]) torch.Size([3, 192, 64, 64]) torch.Size([3, 2, 64, 64]) torch.Size([3, 2, 64, 64])

        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
