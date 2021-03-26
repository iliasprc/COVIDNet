import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from model.model import CovidNet, CNN
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


def save_checkpoint(state, is_best, path, filename='last'):
    name = os.path.join(path, filename + '_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)


def save_model(model, optimizer, args, metrics, epoch, best_pred_loss, confusion_matrix):
    loss = metrics._data.average['loss']
    save_path = args.save
    make_dirs(save_path)

    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    is_best = False
    if loss < best_pred_loss:
        is_best = True
        best_pred_loss = loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'loss':  loss },
                        is_best, save_path, args.model + "_best")
        np.save(save_path + 'best_confusion_matrix.npy', confusion_matrix.cpu().numpy())

    else:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'loss':  loss },
                        False, save_path, args.model + "_last")

    return best_pred_loss


def make_dirs(path):
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
            print(line,line.split('|'))
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
        print(self.keys)
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + key, value, writer_step)
        self._data.total[key] += value * n
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


# class Metrics:
#     def __init__(self, path, keys=None, writer=None):
#         self.writer = writer
#
#         self.data = {'correct': 0,
#                      'total': 0,
#                      'loss': 0,
#                      'accuracy': 0,
#                      }
#         self.save_path = path
#
#     def reset(self):
#         for key in self.data:
#             self.data[key] = 0
#
#     def update_key(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key, value)
#         self.data[key] += value
#
#     def update(self, values):
#         for key in self.data:
#             self.data[key] += values[key]
#
#     def avg_acc(self):
#         return self.data['correct'] / self.data['total']
#
#     def avg_loss(self):
#         return self.data['loss'] / self.data['total']
#
#     def save(self):
#         with open(self.save_path, 'w') as save_file:
#             a = 0  # csv.writer()
#             # TODO
#

def select_model(args):
    if args.model == 'COVIDNet_small':
        return CovidNet('small', n_classes=args.classes)

    elif args.model == 'COVIDNet_large':
        return CovidNet('large', n_classes=args.classes)
    elif args.model in ['resnet18', 'mobilenet_v2', 'densenet169', 'resneXt']:
        return CNN(args.classes, args.model)
    elif args.model == 'vit':
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


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print("Epoch:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\tPPV:{:.3f}\tsensitivity{:.3f}".format(epoch,
                                                                                     num_samples,
                                                                                     len(
                                                                                         trainloader) * args.batch_size,
                                                                                     metrics.avg('loss')
                                                                                     ,
                                                                                     metrics.avg('accuracy'), metrics.avg('ppv'), metrics.avg('sensitivity')))


def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + "\n SUMMARY EPOCH:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\n".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples,
                                                                                                     metrics.avg(
                                                                                                         'loss'),
                                                                                                     metrics.avg(
                                                                                                         'accuracy')))
