import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader.covid_ct_dataset import CovidCTDataset
from data_loader.covidxdataset import COVIDxDataset
from model.metric import accuracy, sensitivity, positive_predictive_value
from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker


def initialize(args):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    model = select_model(args)

    optimizer = select_optimizer(args, model)
    if (args.cuda):
        model.cuda()

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 2}

    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 1}
    if args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.root_path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.root_path,
                                   dim=(224, 224))

        test_loader = None

        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **test_params)
        test_generator = None

    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='./data/covid_ct_dataset',
                                      txt_COVID='./data/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='./data/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='./data/covid_ct_dataset',
                                    txt_COVID='./data/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='./data/covid_ct_dataset/valCT_NonCOVID.txt')
        test_loader = CovidCTDataset('test', root_dir='./data/covid_ct_dataset',
                                     txt_COVID='./data/covid_ct_dataset/testCT_COVID.txt',
                                     txt_NonCOVID='./data/covid_ct_dataset/testCT_NonCOVID.txt')

        training_generator = DataLoader(train_loader, **train_params)
        val_generator = DataLoader(val_loader, **test_params)
        test_generator = DataLoader(test_loader, **test_params)

    return model, optimizer, training_generator, val_generator, test_generator


def train(args, model, trainloader, optimizer, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy', 'ppv', 'sensitivity']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)

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
    train_metrics.update('sensitivity', s, writer_step=(epoch - 1) * len(trainloader) + batch_idx)
    train_metrics.update('ppv', ppv, writer_step=(epoch - 1) * len(trainloader) + batch_idx)
    print_summary(args, epoch, num_samples, train_metrics, mode="Training")
    return train_metrics


def validation(args, model, testloader, epoch, writer):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'total', 'accuracy']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
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
    val_metrics.update('sensitivity', s, writer_step=(epoch - 1) * len(testloader) + batch_idx)
    val_metrics.update('ppv', ppv, writer_step=(epoch - 1) * len(testloader) + batch_idx)
    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return val_metrics, confusion_matrix
