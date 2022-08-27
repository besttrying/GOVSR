import sys
import os
import time
import numpy as np
from os.path import join,exists
import glob
from tqdm import trange, tqdm
import cv2
import math
import scipy
import torch
from torch.nn import functional as F
import json


def automkdir(path):
    if not exists(path):
        os.makedirs(path)

def automkdirs(path):
    [automkdir(p) for p in path]

def compute_psnr_torch(img1,img2):
    # mse=torch.mean((img1-img2)**2)
    # psnr_avg = 10*torch.log10(1/mse)

    b, f, h, w, c = img1.shape
    mse = torch.mean((img1 - img2) ** 2, (0, 2, 3, 4), False)
    psnr = 10 * torch.log10(1. / mse)
    psnr_avg = torch.mean(psnr)
    psnr_var = torch.var(psnr)
    return psnr_avg, psnr_var

def compute_psnr(img1,img2):
    mse=np.mean((img1-img2)**2)
    return 10*np.log(1/mse)/np.log(10)

def rgb2ycbcr(inputs):
    if inputs.shape[-1] == 1:
        return inputs
    assert inputs.shape[-1] == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
    ndims = len(inputs.shape)
    origT = [[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]]
    origOffset = [16.0, 128.0, 128.0]
    if ndims == 3:
        origT = [np.reshape(origT[i], [1, 1, 3]) / 255.0 for i in range(3)]
    if ndims == 4:
        origT = [np.reshape(origT[i], [1, 1, 1, 3]) / 255.0 for i in range(3)]
    elif ndims == 5:
        origT = [np.reshape(origT[i], [1, 1, 1, 1, 3]) / 255.0 for i in range(3)]
    output = []
    for i in range(3):
        channel=np.sum(inputs * origT[i], axis=-1) + origOffset[i] / 255.0
        channel=np.expand_dims(channel, axis=-1)
        channel=np.clip(channel, 0.0, 255.0)
        output.append(channel)
    return np.concatenate(output, -1)


def rgb2ycbcr_torch(inputs):
    if inputs.shape[-1] == 1:
        return inputs
    assert inputs.shape[-1] == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
    ndims = len(inputs.shape)
    origT = torch.Tensor([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]]).to(inputs.device) / 255.
    origOffset = torch.Tensor([16.0, 128.0, 128.0]).to(inputs.device) / 255.
    if ndims == 3:
        origT = [origT[i].view(1, 1, 3) for i in range(3)]
    if ndims == 4:
        origT = [origT[i].view(1, 1, 1, 3) for i in range(3)]
    elif ndims == 5:
        origT = [origT[i].view(1, 1, 1, 1, 3) for i in range(3)]
    output = []
    for i in range(3):
        channel=torch.sum(inputs * origT[i], axis=-1, keepdim=True) + origOffset[i]
        channel=torch.clamp(channel, 0.0, 255.0)
        output.append(channel)
    return torch.cat(output, -1)

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    filter_height, filter_width = 13, 13
    pad_w, pad_h = (filter_width-1)//2, (filter_height-1)//2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        if H%3!=0:
            r_h = 3 - (H % 3)
        if W%3!=0:
            r_w = 3 - (W % 3)
    x = F.pad(x, (pad_w, pad_w + r_w, pad_h, pad_h + r_h), 'reflect')

    gaussian_filter = torch.from_numpy(gkern(filter_height, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    #x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x

def extractlr_fromhr(hr, scale=4, device=None):
    return [i.to(device) for i in hr]

def makelr_fromhr_cuda(hr, scale=4, device=None, kind='hr'):
    if kind == 'lrhr':
        return [i.to(device) for i in hr]
    else:
        if device:
            hr=hr.to(device)
        lr=DUF_downsample(hr,scale)
        b,c,f,h,w=lr.shape
        return lr, hr

def evaluation(model, eval_data, config, rank=0, world_size=1, continue_line=False, every_kind=False):
    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    psnr_all=[]
    pre_psnr_all=[]
    time_all=0
    scale=config.model.scale
    epoch=config.train.epoch
    device=config.device
    test_runtime=[]
    in_h=128
    in_w=240
    bd=2
    border=8

    for iter_eval, (img_hq) in enumerate(tqdm(eval_data)):
        # img_hq = img_hq[:,:,:,bd*scale:(bd+in_h)*scale,bd*scale:(bd+in_w)*scale]
        img_lq, img_hq = makelr_fromhr_cuda(img_hq, scale, device)
        # img_lq, img_hq = extractlr_fromhr(img_hq, device = device)
        # img_lq = img_lq[:, :, :, :in_h, :in_w]
        # img_hq = img_hq[:, :, :, :in_h*scale, :in_w*scale]

        nb,c,f,h,w=img_lq.shape
        
        h_sr,w_sr=h*scale,w*scale
        in_lq=img_lq
        padding=1
        if h%padding !=0 or w%padding !=0:
            newh=(h//padding+1)*padding
            neww=(w//padding+1)*padding
            in_lq=F.pad(img_lq,(0,neww-w,0,newh-h,0,0))
            bic=F.pad(bic,(0,(neww-w)*scale,0,(newh-h)*scale))
        
        start.record()
        with torch.no_grad():
            img_clean = model(in_lq)
        end.record()
        torch.cuda.synchronize()
        test_runtime.append(start.elapsed_time(end) / f)

        if h%padding !=0 or w%padding !=0:
            img_clean=img_clean[:,:,:h_sr,:w_sr]
        # clean=img_clean.permute(0,2,3,4,1)
        cleans=[_.permute(0,2,3,4,1) for _ in img_clean]
        hr=img_hq.permute(0,2,3,4,1)

        if config.eval.color == 'rgb':
            psnr_cleans, psnr_hr = cleans, hr
        else:
            psnr_cleans = [rgb2ycbcr_torch(_)[:, 2:-2, 8:-8, 8:-8, 0:1] for _ in cleans]
            psnr_hr = rgb2ycbcr_torch(hr)[:, 2:-2, 8:-8, 8:-8, 0:1]
        psnrs=[list(map(lambda x: x.cpu().numpy(), compute_psnr_torch(_, psnr_hr))) for _ in psnr_cleans]

        # clean=(np.round(np.clip(cleans[0].cpu().numpy()[0, f//2] * 255, 0, 255))).astype(np.uint8)
        # cv2_imsave(join(config.path.eval_result,'{:0>4}.png'.format(iter_eval * world_size + rank)), clean)
        psnr_all.append(psnrs)

    
    psnrs = np.array(psnr_all)
    num_out = psnrs.shape[1]
    num_eval_tensorlist = list(torch.zeros([world_size], dtype=torch.float32, device = device))
    psnr_tensorlist = list(torch.zeros([world_size, num_out], dtype=torch.float32, device = device))
    var_tensorlist = list(torch.zeros([world_size, num_out], dtype=torch.float32, device = device))
    if world_size > 1:
        torch.distributed.all_gather(num_eval_tensorlist, torch.tensor(len(eval_data), device = device, dtype = torch.float32))
        torch.distributed.all_gather(psnr_tensorlist, torch.tensor(np.sum(psnrs[:, :, 0], 0, keepdims = False), device = device, dtype = torch.float32))
        torch.distributed.all_gather(var_tensorlist, torch.tensor(np.sum(psnrs[:, :, 1], 0, keepdims = False), device = device, dtype = torch.float32))
        num_eval_tensorlist = [float(p.cpu().numpy()) for p in num_eval_tensorlist]
        psnr_tensorlist = np.array([p.cpu().numpy() for p in psnr_tensorlist])
        var_tensorlist = np.array([p.cpu().numpy() for p in var_tensorlist])
        psnr_avg = np.sum(psnr_tensorlist, 0, keepdims = False) / sum(num_eval_tensorlist)
        psnr_var = np.sum(var_tensorlist, 0, keepdims = False) / sum(num_eval_tensorlist)
    else:
        psnr_avg = np.mean(psnrs[:, :, 0], 0, keepdims = False)
        psnr_var = np.mean(psnrs[:, :, 1], 0, keepdims = False)

    if rank == 0:
        with open(config.path.eval_file,'a+') as f:
            eval_dict = {'Epoch': epoch, 'PSNR': psnr_avg.tolist(), 'PSNR_var': psnr_var.tolist()}
            if every_kind:
                eval_dict = {'Epoch': epoch, 'PSNR': psnr_avg.tolist()}
                for idx, p in enumerate(psnr_all):
                    eval_dict[f'Seq_{idx:03}'] = p[0][0].tolist()
            f.write(json.dumps(eval_dict))
            if not continue_line:
                f.write('\n')
        print(json.dumps(eval_dict))
        ave_runtime = sum(test_runtime) / len(test_runtime)
        print(f'average time cost {ave_runtime} ms')

    model.train()
    
    return psnr_avg



def test_video(model, path, savepath, scale, num_frame, inp_type='truth', device=None):
    model.eval()
    automkdir(savepath)
    # print(savepath)
    prefix = os.path.split(path)[-1]
    imgs=sorted(glob.glob(join(path, inp_type, '*.png')))
    time_all=0
    num_frame=num_frame
    max_frame=len(imgs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_runtime=[]
    
    if inp_type == 'truth':
        img_hq=[cv2_imread(i) for i in imgs]
        img_hq=torch.from_numpy(np.array(img_hq)/255.).float().permute(3,0,1,2).contiguous()
        # img_hq=torch.ones(3,max_frame,720,1280)
        img_hq=img_hq.to(device)
        img_lq=DUF_downsample(img_hq.unsqueeze(0),scale)
        b, c, f, h, w = img_lq.shape
    else:
        img_lq=[cv2_imread(i) for i in imgs]
        img_lq=torch.from_numpy(np.array(img_lq)).float().permute(3,0,1,2).contiguous()/255.
        c, f, h, w = img_lq.shape
        img_lq = img_lq.to(device).unsqueeze(0)

    files_info = [os.path.split(_)[-1] for _ in imgs]

    start.record()
    with torch.no_grad():
        img_clean = model(img_lq)
    end.record()
    torch.cuda.synchronize()
    test_runtime.append(start.elapsed_time(end))  # milliseconds

    if isinstance(img_clean, (tuple, list)):
        img_clean = img_clean[0]

    sr=img_clean[0].permute(1,2,3,0)
    sr=sr.cpu().numpy()
    sr=(np.round(np.clip(sr*255,0,255))).astype(np.uint8)
    [cv2_imsave(join(savepath, files_info[i]), sr[i]) for i in range(f)]
    print('Cost {} ms in average.\n'.format(np.mean(test_runtime) / f))

    # cleans=img_clean.permute(0,2,3,4,1)
    # hr=img_hq.unsqueeze(0).permute(0,2,3,4,1)
    # psnr_cleans = rgb2ycbcr_torch(cleans)[:, 2:-2, 8:-8, 8:-8, 0:1]
    # psnr_hr = rgb2ycbcr_torch(hr)[:, 2:-2, 8:-8, 8:-8, 0:1]
    # psnrs=list(map(lambda x: x.cpu().numpy(), compute_psnr_torch(psnr_cleans, psnr_hr)))
    # print(psnrs)

    return


def test_video_ivsr(model, path, savepath, scale, num_frame, inp_type='truth', device=None):
    model.eval()
    automkdir(savepath)
    # print(savepath)
    imgs=sorted(glob.glob(join(path, inp_type, '*.png')))
    time_all=0
    num_frame=num_frame
    max_frame=len(imgs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_runtime=[]
    
    img_lq=[cv2_imread(i) for i in imgs]
    img_lq=torch.from_numpy(np.array(img_lq)).float().permute(3,0,1,2).contiguous()/255.
    c, f, h, w = img_lq.shape
    img_lq = img_lq.to(device).unsqueeze(0)

    files_info = [os.path.split(_) for _ in imgs]

    nef = 10
    for t in range(9, f, 10):
        index = np.array(list(range(t - nef, t + nef + 1)))
        index = np.clip(index, 0, f-1).tolist()
        in_lq = img_lq[:, :, index]

        start.record()
        with torch.no_grad():
            img_clean = model(in_lq)[0]
        end.record()
        torch.cuda.synchronize()
        test_runtime.append(start.elapsed_time(end) / (nef * 2 + 1) )  # milliseconds

        sr=img_clean[0].permute(1,2,3,0)[nef]
        sr=sr.cpu().numpy()
        sr=(np.round(np.clip(sr*255,0,255))).astype(np.uint8)
        cv2_imsave(join(savepath, files_info[t][-1]), sr)
    print('Cost {} ms in average.\n'.format(np.mean(test_runtime)))

    return

def test_video_clip(model, path, savepath, scale, num_frame, inp_type='truth', device=None):
    model.eval()
    automkdir(savepath)
    # print(savepath)
    imgs=sorted(glob.glob(join(path, inp_type, '*.png')))
    time_all=0
    num_frame=num_frame
    max_frame=len(imgs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_runtime=[]

    img_lq=[cv2_imread(i) for i in imgs]
    img_lq=torch.from_numpy(np.array(img_lq)).float().permute(3,0,1,2).contiguous()/255.
    c, f, h, w = img_lq.shape
    img_lq = img_lq.to(device).unsqueeze(0)

    files_info = [os.path.split(_) for _ in imgs]

    start.record()
    with torch.no_grad():
        img_clean = [model(img_lq[:, :, _ * 50:(_ + 1) * 50]) for _ in range(2)]
    end.record()
    torch.cuda.synchronize()
    test_runtime.append(start.elapsed_time(end))  # milliseconds
    img_clean = torch.cat([img_clean[_][0] for _ in range(len(img_clean))], 2)

    # if isinstance(img_clean, tuple):
    #     img_clean = img_clean[0]

    sr=img_clean[0].permute(1,2,3,0)
    sr=sr.cpu().numpy()
    sr=(np.round(np.clip(sr*255,0,255))).astype(np.uint8)
    [cv2_imsave(join(savepath, files_info[i][-1]), sr[i]) for i in range(f)]
    print('Cost {} ms in average.\n'.format(np.mean(test_runtime) / f))

    return

def save_checkpoint(model, epoch, model_folder):
    # epoch = config.train.epoch
    # model_folder = config.path.checkpoint
    model_out_path = os.path.join(model_folder , '{:0>4}.pth'.format(epoch))
    state = {"epoch": epoch ,"model": model.state_dict()}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    return

def load_checkpoint(network=None, resume='', path='', weights_init=None, rank=0):
    try:
        num_resume = int(resume[resume.rfind('/')+1:resume.rfind('.')])
    except Exception as e:
        num_resume = 0
    finally:
        if num_resume<0:
            checkpointfile=sorted(glob.glob(join(path,'*')))
            if len(checkpointfile)==0:
                resume='nofile'
            else:
                resume=checkpointfile[-1]
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage.cuda(rank))
            # checkpoint = torch.load(resume, map_location=map_location)
            # for k,v in checkpoint['model'].state_dict().items():
            #     print(k)
            start_epoch = checkpoint["epoch"]
            network.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            # if weights_init is not None:
            #     network.apply(weights_init)
            start_epoch=0
    
    return start_epoch

def adjust_learning_rate(init_lr, final_lr, epoch, epoch_decay, iteration, iter_per_epoch, optimizer, ifprint=False):
    """Sets the learning rate to the initial LR decayed by 10"""
    # lr = opt.lr * (opt.momentum ** (epoch / opt.step))#+1e-5
    lr = (init_lr-final_lr) * max((1 - (epoch + iteration / iter_per_epoch) / epoch_decay), 0) + final_lr
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:
            param_group["lr"] = lr
        else:
            param_group["lr"] = (1e-4-final_lr) * max((1 - (epoch + iteration / iter_per_epoch) / epoch_decay), 0) + final_lr
        # if i == 1:#if len(param_group['params']) == 60:
        #     continue
        #     param_group["lr"] = (2.5e-4-final_lr) * max((1 - (epoch + iteration / iter_per_epoch) / epoch_decay), 0) + final_lr
        # print(len(param_group['params']), param_group["lr"])
        
    if ifprint:
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    return lr

def cv2_imsave(img_path, img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def cv2_imread(img_path):
    img=cv2.imread(img_path)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    return img

if __name__=='__main__':
    pass

class DICT2OBJ(object):
    def __init__(self, obj, v=None):
        # if not isinstance(obj, dict):
        #     setattr(self, obj, v)
        #     return 
        for k, v in obj.items():
            if isinstance(v, dict):
                # print('dict', k, v)
                setattr(self, k, DICT2OBJ(v))
            else:
                # print('no dict', k, v)
                setattr(self, k, v)
