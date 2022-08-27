import os
import numpy as np
from os.path import join, exists
import glob

def write_file(datapath='', savepath=''):
    files=sorted(glob.glob(join(datapath,'*','*')))
    files = [f for f in files if os.path.isdir(f)]
    with open(savepath,'w') as f:
        for fil in files:
            print(fil)
            f.write(fil+'\n')
        
if __name__=='__main__':
    write_file('/home/eliza/yipeng/data/vsr/mm522_crop', '/home/eliza/yipeng/data/data_txt/mm522_train.txt')