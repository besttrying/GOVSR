import os
import sys
import time
import numpy as np
import math
import random
from os.path import join,exists
import glob
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.backends.cudnn as cudnn
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from torch import autograd
from util import automkdir, adjust_learning_rate, evaluation, load_checkpoint, save_checkpoint, makelr_fromhr_cuda, extractlr_fromhr, test_video
import dataloader
from models.common import weights_init, cha_loss
from ptflops import get_model_complexity_info

def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, config):
    torch.cuda.set_device(rank)
    config.device = device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # config.seed = random.randint(1, 10000)
    print("Random Seed: ", config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    world_size = len(config.train.gpus)
    setup(rank, world_size)

    model = config.network
    criterion = getattr(sys.modules[__name__], config.train.loss)()
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = criterion.to(device)

    dist.barrier()
    epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init = weights_init, rank = rank)

    base_params = filter(lambda p: id(p) not in list(map(id, model.module.spynet.parameters())), model.parameters())
    optimizer = torch.optim.Adam([{'params': base_params},{'params':model.module.spynet.parameters(),'lr': config.train.final_lr}], lr=config.train.init_lr, weight_decay=0)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.init_lr, weight_decay=0)

    
    config.train.epoch = epoch
    iter_per_epoch = config.train.iter_per_epoch
    epoch_decay = config.train.epoch_decay
    step = 0
    scaler = GradScaler()

    train_batch_size = config.train.batch_size // world_size + max(min(config.train.batch_size % world_size - rank, 1), 0)
    eval_dataset = [dataloader.irloader(_, mode='eval', scale=config.model.scale, num_frame=config.train.num_frame, rank = rank, world_size = config.eval.world_size) for _ in config.path.eval]
    eval_loader = [torch.utils.data.DataLoader(_, batch_size=config.eval.batch_size, shuffle=False, num_workers=config.eval.num_workers, pin_memory=True) for _ in eval_dataset]

    loss_frame_seq = list(range(config.train.sub_frame, config.train.num_frame - config.train.sub_frame))
    continue_lines = [True for _ in range(len(eval_loader) - 1)] + [False]

    PSNRs = []
    while(epoch < config.train.num_epochs):
        if step == 0:
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch, iter_per_epoch, optimizer, True)
            if config.eval.world_size > 1 or rank == 0:
                eval_psnr = [evaluation(model, i, config, rank, config.eval.world_size, j) for i, j in zip(eval_loader, continue_lines)]
                eval_psnr = float(eval_psnr[0])
                PSNRs.append(eval_psnr)
            time_start = time.time()

        train_dataset = dataloader.irloader(config.path.train, mode='train', scale=config.model.scale, crop_size=config.train.in_size, num_frame=config.train.num_frame, rank = rank, world_size = world_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=config.train.num_workers, pin_memory=True, drop_last=True)
        # print(rank, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        for iteration, (img_hq) in enumerate(train_loader):
            adjust_learning_rate(config.train.init_lr, config.train.final_lr, epoch, epoch_decay, step % iter_per_epoch, iter_per_epoch, optimizer, False)
            optimizer.zero_grad()

            # img_lq, img_hq = extractlr_fromhr(img_hq, device = device)
            img_lq, img_hq = makelr_fromhr_cuda(img_hq, config.model.scale, device)
            # torch.autograd.set_detect_anomaly(True)
            with autocast():
                it_all = model(img_lq, config.train.sub_frame)
                loss = criterion(it_all[0], img_hq[:, :, loss_frame_seq]) #+ criterion(pre_it_all, img_hq[:, :, loss_frame_seq])
                for _ in range(1, len(it_all)):
                    loss += 0.01 * criterion(it_all[_], img_hq[:, :, loss_frame_seq])
                # loss = criterion(pre_it_all, img_hq[:, :, loss_frame_seq])

            loss_v = loss.detach()
            if (loss_v > 0.5 or loss_v < 0 or math.isnan(loss_v)) and epoch > 0:
                # config.path.resume = join(config.path.checkpoint, f'{epoch:04}.pth')
                print(f'epoch {epoch}, skip iteration {iteration}, loss {loss_v}')
                # dist.destroy_process_group()
                raise RuntimeWarning(f'epoch {epoch}, iteration {iteration}, loss {loss_v}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            step += 1

            if (step % config.train.display_iter) == 0 and rank == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),"Epoch[{}/{}]({}/{}): Loss: {:.8f}".format(epoch, config.train.num_epochs, step%iter_per_epoch, iter_per_epoch, loss_v))
                sys.stdout.flush()
            
            if step % iter_per_epoch == 0:
                dist.barrier()
                if config.eval.world_size > 1 or rank == 0:
                    time_cost = time.time()-time_start
                    print(f'spent {time_cost} s')
                    epoch += 1
                    config.train.epoch = epoch

                    eval_psnr = [evaluation(model, i, config, rank, config.eval.world_size, j) for i, j in zip(eval_loader, continue_lines)]
                    eval_psnr = float(eval_psnr[0])
                    if PSNRs[-1] - eval_psnr > 1:
                        raise RuntimeWarning(f'epoch {epoch}, iteration {iteration}, loss {loss_v}, PSNR from {PSNRs[-1]} to {eval_psnr}')
                    PSNRs.append(eval_psnr)
                    sys.stdout.flush()

                # dist.barrier()

                if rank == 0:
                    save_checkpoint(model, epoch, config.path.checkpoint)
                    if epoch == config.train.num_epochs:
                        raise Exception(f'epoch {epoch} >= max epoch {config.train.num_epochs}')
                    # time_start = time.time()
                    print(f'Epoch={epoch}, lr={optimizer.param_groups[0]["lr"]}')
                time_start = time.time()
                
    # if start_epoch == config.train.num_epochs:
    #     [evaluation(model, i, config, rank, config.eval.world_size, j) for i, j in zip(eval_loader, continue_lines)]


def test(rank, config):
    print(config.path.test)
    
    torch.cuda.set_device(rank)
    config.device = device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    world_size = len(config.train.gpus)
    setup(rank, world_size)
    self_ensemble = config.test.self_ensemble

    model = config.network
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    start_epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init, rank)

    inp_type = 'truth'
    datapath = sorted(glob.glob(join(config.path.test, '*')))
    datapath = [d for d in datapath if os.path.isdir(d)][rank :: world_size]
    seqname = [os.path.split(d)[-1] for d in datapath]
    savepath = [join(config.path.test, d, config.test.save_name) for d in seqname]
    
    for i, d in enumerate(tqdm(datapath)):
        print(d, savepath[i])
        test_video(model, d, savepath[i], config.model.scale, config.train.num_frame, inp_type=inp_type, device=device)


def eval_checkpoint(rank, config):
    torch.cuda.set_device(rank)
    config.device = device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    world_size = len(config.train.gpus)
    setup(rank, world_size)
    model = config.network
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)


    eval_dataset = [dataloader.irloader(_, mode='eval', scale=config.model.scale, num_frame=config.train.num_frame, rank = rank, world_size = config.eval.world_size) for _ in config.path.eval]
    eval_loader = [torch.utils.data.DataLoader(_, batch_size=config.eval.batch_size, shuffle=False, num_workers=config.eval.num_workers, pin_memory=True) for _ in eval_dataset]
    continue_lines = [True for _ in range(len(eval_loader) - 1)] + [False]

    if config.eval.world_size > 1 or rank == 0:
        for i in range(290, 308):
            config.path.resume = join(config.path.checkpoint, f'{i:04}.pth')
            config.train.epoch = load_checkpoint(model, config.path.resume, config.path.checkpoint, weights_init, rank)
            [evaluation(model, i, config, rank, config.eval.world_size, j, False) for i, j in zip(eval_loader, continue_lines)]

def get_complexity(config, shape = (32, 180, 320), times = 1):
    T, H, W = shape
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = config.network.to(device)
    model.eval()
    test_runtime = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_lq=torch.ones(1, 3, 10, H, W).to(device)

    with torch.no_grad():
        hq = model(img_lq)

    for _ in range(times):
        start.record()

        with torch.no_grad():
            macs, params = get_model_complexity_info(model, (3, T, H, W), as_strings=False,
                                                        print_per_layer_stat=False, verbose=True, ignore_modules=[torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.PReLU]) #(3, 1, 270, 480) 180, 320

        end.record()
        torch.cuda.synchronize()
        test_runtime.append(start.elapsed_time(end))  # milliseconds

    for tt in test_runtime:
        print('Cost {} ms in average.'.format(tt / T))

    print('Computational complexity: {:,}'.format(macs))
    print('Number of parameters: {:,}'.format(params))
    