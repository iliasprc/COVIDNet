import argparse
import csv
import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model.cnn import CNN
from model.model import CovidNet
from model.vit import ViT


def write_score(writer, iter, mode, metrics):
    writer.add_scalar(mode + '/loss', metrics.data['loss'], iter)
    writer.add_scalar(mode + '/acc', metrics.data['correct'] / metrics.data['total'], iter)


def write_train_val_score(writer, epoch, train_stats, val_stats):
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return


def showgradients(model):
    for param in model.parameters():
        print(type(param.data), param.size())
        print("GRADS= \n", param.grad)


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, filename='last'):
    name = os.path.join(path, filename + '_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)


def load_checkpoint(checkpoint, model, strict=True, optimizer=None, load_seperate_layers=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint1 = torch.load(checkpoint, map_location='cpu')
    print(checkpoint1.keys())
    pretrained_dict = checkpoint1['model_dict']
    model_dict = model.state_dict()
    print(pretrained_dict.keys())
    print(model_dict.keys())
    # # # 1. filter out unnecessary keys
    # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dictnew = {}
    for k, v in pretrained_dict.items():
        if 'module.' in k:
            k = k[7:]
        pretrained_dictnew[k] = v

    print(pretrained_dictnew.keys())

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if (not load_seperate_layers):
        # model.load_state_dict(checkpoint1['model_dict'] , strict=strict)p
        model.load_state_dict(pretrained_dictnew, strict=strict)

    epoch = 0
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return checkpoint1, epoch


def save_model(cpkt_dir, model, optimizer, loss, epoch, name):
    save_path = cpkt_dir
    make_dirs(save_path)

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': loss}
    name = os.path.join(cpkt_dir, name + '_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dirs_if_not_present(path):
    """
    creates new directory if not present
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_stats_files(path):
    train_f = open(os.path.join(path, 'train.csv'), 'w')
    val_f = open(os.path.join(path, 'val.csv'), 'w')
    return train_f, val_f


def read_json_file(fname):
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json_file(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_filepaths(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if ('/ c o' in line):
                break
            subjid, path, label = line.split(' ')

            paths.append(path)
            labels.append(label)
    return paths, labels


def read_filepaths2(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            # print(line, line.split('|'))
            if ('/ c o' in line):
                break
            path, label, dataset = line.split('|')
            path = path.split(' ')[-1]

            paths.append(path)
            labels.append(label)
    return paths, labels


class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):

        self.writer = writer
        self.mode = mode + '/'
        self.keys = keys
        # print(self.keys)
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + key, value, writer_step)
        self._data.total[key] += value
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_all_metrics(self, values_dict, n=1, writer_step=1):
        for key in values_dict:
            self.update(key, values_dict[key], n, writer_step)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def calc_all_metrics(self):
        """
        Calculates string with all the metrics
        Returns:
        """
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += f'{key} {d[key]:7.4f}\t'

        return s

    def print_all_metrics(self):
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += "{} {:.4f}\t".format(key, d[key])

        return s


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print(
            "Epoch:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\tPPV:{:.3f}\tsensitivity{:.3f}".format(epoch,
                                                                                                                  num_samples,
                                                                                                                  len(
                                                                                                                      trainloader) * args.batch_size,
                                                                                                                  metrics.avg(
                                                                                                                      'loss')
                                                                                                                  ,
                                                                                                                  metrics.avg(
                                                                                                                      'accuracy'),
                                                                                                                  metrics.avg(
                                                                                                                      'ppv'),
                                                                                                                  metrics.avg(
                                                                                                                      'sensitivity')))


def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + "\n SUMMARY EPOCH:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\n".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples,
                                                                                                     metrics.avg(
                                                                                                         'loss'),
                                                                                                     metrics.avg(
                                                                                                         'accuracy')))


def load_csv_file(path):
    """

    Args:
        path ():

    Returns:

    """
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        data_paths.append(item[0])
        labels.append(item[1])
    return data_paths, labels


def txt_logger(txtname, log):
    """

    Args:
        txtname ():
        log ():
    """
    with open(txtname, 'a') as f:
        for item in log:
            f.write(item)
            f.write(',')

        f.write('\n')


def write_csv(data, name):
    """

    Args:
        data ():
        name ():
    """
    with open(name, 'w') as fout:
        for item in data:
            # print(item)
            fout.write(item)
            fout.write('\n')


def check_dir(path):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.makedirs(path)


def get_lr(optimizer):
    """

    Args:
        optimizer ():

    Returns:

    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0].strip('--').strip('-')] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--log_interval', type=int, default=1000, help='steps to log.info metrics and loss')
    parser.add_argument('--dataset_name', type=str, default="COVIDx", help='dataset name COVIDx or COVID_CT')
    parser.add_argument('--nEpochs', type=int, default=250, help='total number of epochs')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--seed', type=int, default=123, help='select seed number for reproducibility')
    parser.add_argument('--classes', type=int, default=3, help='dataset classes')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=True, help='use gpu for speed-up')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='use tensorboard for loggging and visualization')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        choices=('COVIDNet_small', 'resnet18', 'mobilenet_v2', 'densenet169', 'COVIDNet_large'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--root_path', type=str, default='./data/data',
                        help='path to dataset ')
    parser.add_argument('--save', type=str, default='./saved/COVIDNet',
                        help='path to checkpoint save directory ')
    args = parser.parse_args()
    return args


def reproducibility(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    SEED = config.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def select_optimizer(model, config, checkpoint=None):
    opt = config['optimizer']['type']
    lr = config['optimizer']['lr']
    if (opt == 'Adam'):
        print(" use optimizer Adam lr ", lr)
        optimizer = optim.Adam(model.parameters(), lr=float(config['optimizer']['lr']), weight_decay=0.000001)
    elif (opt == 'SGD'):
        print(" use optimizer SGD lr ", lr)
        optimizer = optim.SGD(model.parameters(), lr=float(config['optimizer']['lr']), momentum=0.9,
                              weight_decay=float(config['optimizer']['weight_decay']))
    elif (opt == 'RMSprop'):
        print(" use RMS  lr", lr)
        optimizer = optim.RMSprop(model.parameters(), lr=float(config['optimizer']['lr']),
                                  weight_decay=float(config['optimizer']['weight_decay']))
    if (checkpoint != None):
        # print('load opt cpkt')
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        for g in optimizer.param_groups:
            g['lr'] = 0.005
        print(optimizer.state_dict()['state'].keys())

    if config['scheduler']['type'] == 'ReduceLRonPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler']['scheduler_factor'],
                                      patience=config['scheduler']['scheduler_patience'],
                                      min_lr=config['scheduler']['scheduler_min_lr'],
                                      verbose=config['scheduler']['scheduler_verbose'])
        return optimizer, scheduler

    return optimizer, None


def select_model(config, n_classes):
    if config.model.name == 'COVIDNet_small':
        return CovidNet('small', n_classes=n_classes)

    elif config.model.name == 'COVIDNet_large':
        return CovidNet('large', n_classes=n_classes)
    elif config.model.name in ['resnet18', 'mobilenet_v2', 'densenet121', 'resneXt']:
        return CNN(n_classes, config.model.name)
    elif config.model.name == 'vit':
        return ViT(
            image_size=224,
            patch_size=32,
            num_classes=3,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
