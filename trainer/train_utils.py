import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader.cxr8.cxr8_dataset import CXR8Dataset
from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset
from model.metric import accuracy, sensitivity, positive_predictive_value
from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker


def select_dataset(config):
    test_params = {'batch_size': config.dataloader.test.batch_size,
                   'shuffle': False,
                   'num_workers': 2}
    val_params = {'batch_size': config.dataloader.val.batch_size,
                  'shuffle': config.dataloader.val.shuffle,
                  'num_workers': config.dataloader.val.num_workers,
                  'pin_memory': True}

    train_params = {'batch_size': config.dataloader.train.batch_size,
                    'shuffle': config.dataloader.train.shuffle,
                    'num_workers': config.dataloader.train.num_workers,
                    'pin_memory': True}
    print(config.dataset.name)
    if config.dataset.name == 'COVIDx':
        train_loader = COVIDxDataset(config, mode='train')
        val_loader = COVIDxDataset(config, mode='test')
        class_dict = train_loader.class_dict
        test_loader = None

        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **test_params)
        test_generator = None
        return training_generator, val_generator, test_generator, class_dict
    elif config.dataset.namee == 'COVID_CT':
        train_loader = CovidCTDataset(config, 'train')
        val_loader = CovidCTDataset(config, 'val')
        test_loader = CovidCTDataset(config, 'test')
        class_dict = train_loader.class_dict
        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **val_params)
        test_generator = DataLoader(test_loader, **test_params)
        return training_generator, val_generator, test_generator, class_dict
    elif config.dataset.name == 'CXR8':
        train_loader = CXR8Dataset(config, 'train')
        val_loader =  CXR8Dataset(config, 'val')
        test_loader =  CXR8Dataset(config, 'test')
        class_dict = train_loader.class_dict
        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **val_params)
        test_generator = DataLoader(test_loader, **test_params)


        return training_generator, val_generator, test_generator, class_dict


def initialize_model(args):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    model = select_model(args)

    optimizer = select_optimizer(args, model)
    if (args.cuda):
        model.cuda()
    return model, optimizer


def train(args, model, trainloader, optimizer, epoch, writer, log):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy', 'ppv', 'sensitivity']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()
    confusion_matrix = torch.zeros(args.class_dict, args.class_dict)

    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()

        output = model(input_data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        correct, total, acc = accuracy(output, target)
        pred = torch.argmax(output, dim=1)

        num_samples = batch_idx * args.batch_size + 1
        train_metrics.update_all_metrics(
            {'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc},
            writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        print_stats(args, epoch, num_samples, trainloader, train_metrics)
        for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    s = sensitivity(confusion_matrix.numpy())
    ppv = positive_predictive_value(confusion_matrix.numpy())
    print(f" s {s} ,ppv {ppv}")
    # train_metrics.update('sensitivity', s, writer_step=(epoch - 1) * len(trainloader) + batch_idx)
    # train_metrics.update('ppv', ppv, writer_step=(epoch - 1) * len(trainloader) + batch_idx)
    print_summary(args, epoch, num_samples, train_metrics, mode="Training")
    return train_metrics


def validation(args, model, testloader, epoch, writer):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy', 'ppv', 'sensitivity']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(args.class_dict, args.class_dict)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()

            output = model(input_data)

            loss = criterion(output, target)

            correct, total, acc = accuracy(output, target)
            num_samples = batch_idx * args.batch_size + 1
            _, pred = torch.max(output, 1)

            num_samples = batch_idx * args.batch_size + 1
            for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics(
                {'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc},
                writer_step=(epoch - 1) * len(testloader) + batch_idx)

    print_summary(args, epoch, num_samples, val_metrics, mode="Validation")
    s = sensitivity(confusion_matrix.numpy())
    ppv = positive_predictive_value(confusion_matrix.numpy())
    print(f" s {s} ,ppv {ppv}")
    val_metrics.update('sensitivity', s, writer_step=(epoch - 1) * len(testloader) + batch_idx)
    val_metrics.update('ppv', ppv, writer_step=(epoch - 1) * len(testloader) + batch_idx)
    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return val_metrics, confusion_matrix
