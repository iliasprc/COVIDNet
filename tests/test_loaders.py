import datetime
import datetime
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import utils.util as util
from logger.logger import Logger
from trainer.train_utils import select_dataset
from trainer.trainer import Trainer
from utils.util import getopts, reproducibility, select_model, select_optimizer, load_checkpoint, get_arguments


def main():
    args = get_arguments()
    myargs = []# getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.getcwd().replace('/tests','')
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/trainer_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
    config.cwd = str(cwd)
    reproducibility(config)


    




    print(f'pyTorch VERSION:{torch.__version__}', )
    print(f'CUDA VERSION')

    print(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    print(f'Number CUDA Devices: {torch.cuda.device_count()}')


    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f'device: {device}')

    training_generator, val_generator, test_generator, class_dict = select_dataset(config)
    n_classes = len(class_dict)

    for loader in [training_generator,val_generator,test_generator]:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                print(data.shape,target.shape)

if __name__ == '__main__':
    main()
