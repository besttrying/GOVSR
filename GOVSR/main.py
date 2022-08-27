import sys
import os
from os.path import join,exists
from functools import reduce
import time
from util import automkdir, DICT2OBJ, empty_gpu_memory
import yaml
import importlib
import random
from train import train, test, eval_checkpoint, get_complexity
import torch
from ptflops import get_model_complexity_info
import torch.multiprocessing as mp


def ddp_func(demo_fn, config):
    mp.spawn(demo_fn,
             args=(config,),
             nprocs=len(config.train.gpus),
             join=True)

if __name__=='__main__':
    with open('options/eilovsr_3iter_light.yml', 'r', encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file.read())
        config = DICT2OBJ(config)
        
    gpus = config.train.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(_) for _ in gpus]) if gpus else '-1'
    # reduce(lambda x, y: str(x) + ', ' + str(y), gpus[1:], str(gpus[0]) if len(gpus) > 0 else '-1')
    config.seed = random.randint(1, 10000)
    config.eval.world_size = len(gpus) if config.eval.world_size > 1 else 1

    config.path.checkpoint = join(config.path.base, config.path.checkpoint, config.model.name)
    config.path.eval_result = join(config.path.base, config.path.eval_result, config.model.name)
    config.path.resume = join(config.path.checkpoint, f'{config.train.resume:04}.pth')
    config.path.eval_file = join(config.path.base, f'eval_{config.model.name}.txt')
    automkdir(config.path.checkpoint)
    automkdir(config.path.eval_result)

    config.network = importlib.import_module(f'models.{config.model.file}').Net(config)
    
    function = getattr(sys.modules[__name__], config.function)

    if config.function == 'get_complexity':
        function(config, shape = (100, 270, 480), times = 1)
        exit()

    try:
        ddp_func(function, config)
    except RuntimeWarning as e:
        print(e)

