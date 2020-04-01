import torch
import os
import torch.nn as nn
from utils.util import Metrics, print_stats, print_summary, select_model, select_optimizer
from model.metric import accuracy, top_k_acc
from COVIDXDataset.dataset import COVIDxDataset
from torch.utils.data import DataLoader


def initialize(args):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
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

    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224))
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                               dim=(224, 224))
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, optimizer, training_generator, val_generator


def train(args, model, trainloader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metrics = Metrics()
    metrics.reset()
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

        num_samples = batch_idx * args.batch_size + 1
        metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})
        print_stats(args, epoch, num_samples, trainloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Training")
    return metrics


def validation(args, model, testloader, epoch):
    model.eval()
    criterion = nn.CrossEntropyLoss(size_average='mean')

    metrics = Metrics()
    metrics.reset()
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

            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc})
            print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Validation")
    return metrics
