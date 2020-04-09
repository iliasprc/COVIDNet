import argparse

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import utils.util as util
from trainer.train import initialize, train, validation


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (args.cuda):
        torch.cuda.manual_seed(SEED)
    model, optimizer, training_generator, val_generator, test_generator = initialize(args)

    print(model)

    best_pred_loss = 1000.0
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5, verbose=True)
    print('Checkpoint folder ', args.save)
    if args.tensorboard:
        writer = SummaryWriter('./runs/' + util.datestr())
    else:
        writer = None
    for epoch in range(1, args.nEpochs + 1):
        train(args, model, training_generator, optimizer, epoch, writer)
        val_metrics, confusion_matrix = validation(args, model, val_generator, epoch, writer)

        best_pred_loss = util.save_model(model, optimizer, args, val_metrics, epoch, best_pred_loss, confusion_matrix)

        scheduler.step(val_metrics.avg_loss())


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size foe training')
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to print metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="COVID_CT", help='dataset name')
    parser.add_argument('--nEpochs', type=int, default=250, help='total number of epochs')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu for speed-up')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='use tensorboard for loggging and visualization')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='COVIDNet_large',
                        choices=('COVIDNet_small', 'resnet18', 'mobilenet_v2', 'densenet169', 'COVIDNet_large'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--root_path', type=str, default='./data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='/saved/COVIDNet' + util.datestr(),
                        help='path to checkpoint save directory ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
