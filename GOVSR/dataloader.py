import os
import sys
from os.path import join,exists
import numpy as np
import glob
import random
import itertools
from functools import reduce
import torch
import torch.utils.data as data
from torch.nn import functional as F
from util import DUF_downsample, cv2_imread, cv2_imsave


def augmentation(lr,hr):
    if np.random.random() < 0.5:
        lr=lr[:,::-1,:,:]
        hr=hr[:,::-1,:,:]
    if np.random.random() < 0.5:
        lr=lr[:,:,::-1,:]
        hr=hr[:,:,::-1,:]
    if np.random.random() < 0.5:
        lr=lr.transpose(0,2,1,3)
        hr=hr.transpose(0,2,1,3)

    return lr,hr

def double_crop(lr, hr, scale=4, size=32):
    n,h,w,c=lr.shape
    w0=np.random.randint(0,w-size)
    h0=np.random.randint(0,h-size)
    lr=lr[:,h0:h0+size,w0:w0+size,:]
    hr=hr[:,h0*scale:(h0+size)*scale,w0*scale:(w0+size)*scale,:]
    
    return augmentation(lr,hr)

def double_load(lr, hr, keys=None, mode='train', data_mode='pic', crop_size=32, scale=4, num_frame=7):
    max_frame=len(lr)
    center=np.random.randint(0+num_frame//2,max_frame-1-num_frame//2)
    if mode=='eval':
        center=100
    index=np.array([center-num_frame//2+i for i in range(num_frame)])
    index=np.clip(index,0,max_frame-1).tolist()
    lr_img=np.array([cv2_imread(lr[i]) for i in index])
    hr_img=np.array([cv2_imread(hr[i]) for i in index])
    if mode == 'train':
        lr_img, hr_img=double_crop(lr_img, hr_img, scale=scale, size=crop_size)

    return lr_img, hr_img

def single_aug(hr):
    if np.random.random() < 0.5:
        hr=hr[:,::-1,:,:]
    if np.random.random() < 0.5:
        hr=hr[:,:,::-1,:]
    if np.random.random() < 0.5:
        hr=hr.transpose(0,2,1,3)

    return hr

def single_crop(hr, scale=4, size=32):
    n,h,w,c=hr.shape
    w0=np.random.randint(0,w-size*scale+1)
    h0=np.random.randint(0,h-size*scale+1)
    hr=hr[:,h0:h0+size*scale,w0:w0+size*scale,:]
    
    return single_aug(hr)

def single_load(hr, keys ,mode='train', data_mode= 'pic', crop_size=32,scale=4, num_frame=7):
    max_frame = len(keys) if keys is not None else len(hr)
    if mode == 'train':
        idx_st = np.random.randint(0, max_frame - num_frame + 1)
        index = list(range(idx_st, idx_st + num_frame))
    elif mode == 'eval':
        idx_st = 15 - num_frame//2
        index = list(range(idx_st, idx_st + num_frame))
        index = list(range(max_frame))

    hr_img = np.array([cv2_imread(hr[i]) for i in index])

    if mode == 'train':
        hr_img = single_crop(hr_img, size=crop_size)

    return hr_img

class loader_lqhq(data.Dataset):
    def __init__(self, path, mode='train', data_mode='txt', scale=4, crop_size=32, num_frame=7):
        if data_mode=='txt':
            pathlist=open(path, 'rt').read().splitlines()
            if mode=='train':
                self.lqfiles=[p.replace('origin','lr_bicx{}'.format(scale)) for p in pathlist]
                self.hqfiles=[[join(p,'im{}.png'.format(i)) for i in range(1,8)] for p in pathlist]
                self.lqfiles=[[join(p,'im{}.png'.format(i)) for i in range(1,8)] for p in self.lqfiles]
            else:
                self.hqfiles=[join(p,'truth') for p in pathlist]
                self.hqfiles=[sorted(glob.glob(join(p,'*'))) for p in self.hqfiles]
                self.lqfiles=[join(p,'input{}'.format(scale)) for p in pathlist]
                self.lqfiles=[sorted(glob.glob(join(p,'*'))) for p in self.lqfiles]
        else:
            if mode == 'train':
                lqname = 'lrx{}'.format(scale)
                hqname = 'hr'
            else:
                lqname = 'lrx{}'.format(scale)
                hqname = 'hr'
            lqfiles = sorted(glob.glob(join(path, lqname, '*'))) * 8
            self.lqfiles = [sorted(glob.glob(join(f,'*.png'))) for f in lqfiles]
            hqfiles = sorted(glob.glob(join(path, hqname, '*'))) * 8
            self.hqfiles = [sorted(glob.glob(join(f,'*.png'))) for f in hqfiles]

            if mode=='eval':
                self.lqfiles=self.lqfiles[-10:]
                self.hqfiles=self.hqfiles[-10:]

        self.crop_size = crop_size
        self.scale = scale
        self.mode = mode
        self.data_mode = data_mode
        self.num_frame = num_frame
        self.length = len(self.lqfiles)

        print('{} examples for {}'.format(self.length, mode))

    def __getitem__(self, index):
        LR, HR = double_load(self.lqfiles[index], self.hqfiles[index], keys=None, mode=self.mode,
                                 data_mode=self.data_mode, crop_size=self.crop_size, scale=self.scale, num_frame=self.num_frame)
        LR = torch.from_numpy((LR)/255.).float().permute(3, 0, 1, 2)
        HR = torch.from_numpy((HR)/255.).float().permute(3, 0, 1, 2)

        return LR, HR, None

    def __len__(self):
        return self.length


class irloader(data.Dataset):
    def __init__(self, path, mode='train', scale=4, crop_size=32, num_frame=7, rank=0, world_size=1):
        if path.endswith('txt'):
            data_mode = 'txt'
            hqname = 'truth' if mode=='train' else 'truth'
            paths = open(path, 'rt').read().splitlines()
            # self.hqfiles = [[join(p, hqname, f'{i:03}.png') for i in range(32)] for p in paths]
            # self.hqfiles = [sorted(glob.glob(join(p, hqname, '*.png'))) for p in paths]
            # self.hqfiles = [[join(p, hqname, 'im{}.png'.format(i+1)) for i in range(7)] for p in paths]
        else:
            data_mode = 'filefolder'
            hqname = 'truth' if mode=='train' else 'truth'
            paths = sorted(glob.glob(join(path, '*')))
            paths = [p for p in paths if os.path.isdir(p)]
            # self.hqfiles = [sorted(glob.glob(join(p, hqname, '*.png'))) for p in paths]

        if mode == 'train':
            paths = [ddp_shuffle_imgfiles(paths, world_size, rank) for _ in range(1)]
            paths = reduce(lambda x, y: x + y, paths)
        if mode == 'eval':
            paths = paths[rank :: world_size]
            
        if data_mode == 'txt':
            self.hqfiles = [[join(p, hqname, f'{i:03}.png') for i in range(32)] for p in paths]
        elif data_mode == 'filefolder':
            self.hqfiles = [sorted(glob.glob(join(p, hqname, '*.png'))) for p in paths]


        self.crop_size=crop_size
        self.scale=scale
        self.mode=mode
        self.num_frame=num_frame
        self.data_mode=data_mode
        self.length = len(self.hqfiles)

        print(f'{self.length} examples for {mode} with {self.data_mode}')
    
    def __getitem__(self, index):
        hqfile = self.hqfiles[index]
        keys = None
        HR = single_load(hqfile, keys, mode=self.mode, data_mode=self.data_mode, crop_size=self.crop_size, 
                                scale=self.scale, num_frame=self.num_frame)
        HR = torch.from_numpy((HR)/255.).float().permute(3,0,1,2).contiguous()

        return HR#, self.hqfiles[index][0].split('/')[-3]

    def __len__(self):
        return self.length

def ddp_shuffle_imgfiles(imgfiles, world_size=1, rank=0):
    imgfiles1 = imgfiles.copy()
    random.shuffle(imgfiles1)
    # imgfiles = imgfiles[: (len(imgfiles) // world_size) * world_size]
    # return imgfiles
    shuffle_list = [imgfiles1[(rank + i) % world_size :: world_size] for i in range(world_size)]
    shuffle_list = reduce(lambda x, y: x + y, shuffle_list)
    # shuffle_list = []
    # for i in range(world_size):
    #     position = (rank + i) % world_size
    #     shuffle_list += imgfiles[position :: world_size] #imgfiles[position * len_data_part: (position + 1) * len_data_part]
    return shuffle_list