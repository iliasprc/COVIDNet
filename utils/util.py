import torch
import os
import shutil
import time
from collections import OrderedDict
import json
import torch.optim as optim
import pandas as pd
from model.model import CovidNet
import csv


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


def save_checkpoint(model, optimizer, epoch, acc, checkpoint, name):
    state = {'epoch': epoch,
             'model_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict(),
             'acc': acc
             }
    filepath = os.path.join(checkpoint, name + '.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")

    torch.save(state, filepath)
    print("CHECKPOINT SAVED")


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_BEST.pth.tar')


def save_model(model, args, val_stats, epoch, best_pred_loss):
    loss = val_stats[0]
    is_best = False
    if loss < best_pred_loss:
        is_best = True
        best_pred_loss = loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'val_loss': best_pred_loss},
                        is_best, args.save, args.model + "_best")
    elif epoch % 5 == 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'val_loss': best_pred_loss},
                        is_best, args.save, args.model + "last")
    with open(args.save + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
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


class Metrics:
    def __init__(self, path, keys=None, writer=None):
        self.writer = writer

        self.data = {'correct': 0,
                     'total': 0,
                     'loss': 0,
                     'accuracy': 0,
                     }
        self.save_path = path

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    def update_key(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.data[key] += value

    def update(self, values):
        for key in self.data:
            self.data[key] = values[key]

    def avg_acc(self):
        return self.data['correct'] / self.data['total']

    def avg_loss(self):
        return self.data['loss'] / self.data['total']

    def save(self):
        with open(self.save_path, 'w') as save_file:
            a = 0  # csv.writer()
            # TODO


def select_model(args):
    if args.model == 'COVIDNet':
        return CovidNet(args.classes)


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print("Epoch:{}\tSample:{}/{}\tLoss:{:.4f}\tAcc:{:.2f}   Accuracy:{:.2f}".format(epoch,
                                                                                         num_samples,
                                                                                         len(
                                                                                             trainloader) * args.batch_size,
                                                                                         metrics.data[
                                                                                             'loss'] / num_samples,
                                                                                         metrics.data[
                                                                                             'accuracy'] / num_samples,
                                                                                         metrics.data[
                                                                                             'correct'] /
                                                                                         metrics.data[
                                                                                             'total']))


def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + " SUMMARY EPOCH:{}\tSample:{}/{}\tLoss:{:.4f}\tAcc:{:.2f}   Accuracy:{:.2f}".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples * args.batch_size,
                                                                                                     metrics.data[
                                                                                                         'loss'] / num_samples,
                                                                                                     metrics.data[
                                                                                                         'accuracy'] / num_samples,
                                                                                                     metrics.data[
                                                                                                         'correct'] /
                                                                                                     metrics.data[
                                                                                                         'total']))
