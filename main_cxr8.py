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
from trainer.trainer_cxr8 import Trainer
from utils.util import getopts, reproducibility, select_model, select_optimizer, load_checkpoint, get_arguments


def main():
    args = get_arguments()
    myargs = []# getopts(sys.argv)
    now = datetime.datetime.now()
    cwd = os.getcwd()
    if len(myargs) > 0:
        if 'c' in myargs:
            config_file = myargs['c']
    else:
        config_file = 'config/trainer_config.yml'

    config = OmegaConf.load(os.path.join(cwd, config_file))['trainer']
    config.cwd = str(cwd)
    reproducibility(config)
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    cpkt_fol_name = os.path.join(config.cwd,
                                 f'checkpoints/model_{config.model.name}/dataset_{config.dataset.name}/date_{dt_string}')

    log = Logger(path=cpkt_fol_name, name='LOG').get_logger()

    best_pred_loss = 1000.0
    log.info(f"Checkpoint folder {cpkt_fol_name}")
    log.info(f"date and time = {dt_string}")

    log.info(f'pyTorch VERSION:{torch.__version__}', )
    log.info(f'CUDA VERSION')

    log.info(f'CUDNN VERSION:{torch.backends.cudnn.version()}')
    log.info(f'Number CUDA Devices: {torch.cuda.device_count()}')

    if args.tensorboard:

        writer_path = os.path.join(cpkt_fol_name+'runs/')

        writer = SummaryWriter(writer_path + util.datestr())
    else:
        writer = None

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f'device: {device}')

    training_generator, val_generator, test_generator, class_dict = select_dataset(config)
    n_classes = len(class_dict)
    model = select_model(config, n_classes)

    log.info(f"{model}")

    if (config.load):


        pth_file, _ = load_checkpoint(config.pretrained_cpkt, model, strict=True, load_seperate_layers=False)



    else:
        pth_file = None
    if (config.cuda and use_cuda):
        if torch.cuda.device_count() > 1:
            log.info(f"Let's use {torch.cuda.device_count()} GPUs!")

            model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer, scheduler = select_optimizer(model, config['model'], None)
    log.info(f'{model}')
    log.info(f"Checkpoint Folder {cpkt_fol_name} ")
    shutil.copy(os.path.join(config.cwd, config_file), cpkt_fol_name)

    trainer = Trainer(config, model=model, optimizer=optimizer,
                      data_loader=training_generator, writer=writer, logger=log,
                      valid_data_loader=val_generator, test_data_loader=test_generator, class_dict=class_dict,
                      lr_scheduler=scheduler,
                      checkpoint_dir=cpkt_fol_name)
    trainer.train()


if __name__ == '__main__':
    main()
