import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

import pandas as pd

train_ = './data_loader/cxr8/train_images.txt'
test_ = './data_loader/cxr8/test_images.txt'


def read_cxr8(path):
    d = pd.read_csv(path)

    classes = []
    class2index = {}
    images = d[['image']].values.tolist()
    labels = d[['label']].values.tolist()

    for multi_label in labels:
        l = multi_label[0].split('|')

        for i in l:
            if i not in classes:
                classes.append(i)

    i = 0
    for c in classes:
        class2index[c] = i
        i += 1
    assert len(images) == len(labels)

    return images, labels, classes, class2index


class CXR8Dataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, config, mode, dim=(224, 224)):
        self.config = config
        self.root = self.config.dataset.input_data + '/'

        self.dim = dim

        testfile = 'data/test_split.txt'
        trainfile = 'data/train_split.txt'

        self.paths, self.labels, self.classes, self.class_dict = read_cxr8(os.path.join(config.cwd,
                                                                                        train_))

        if mode == 'train':
            split = int(0.2 * len(self.paths))
            self.labels = self.labels[split:]
            self.paths = self.paths[split:]
            self.do_augmentation = True
        elif mode == 'val':
            split = int(0.2 * len(self.paths))
            self.labels = self.labels[:split]
            self.paths = self.paths[:split]
            self.do_augmentation = False
        if (mode == 'test'):
            self.paths, self.labels, _, _ = read_cxr8(os.path.join(config.cwd,
                                                                   test_))

            self.do_augmentation = False
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(self.root + self.paths[index][0], self.dim)
        # image_tensor = torch.randn(3,224,224).float()
        labels = self.labels[index][0].split('|')
        y = torch.zeros(len(self.classes))

        for i in labels:
            if i != 'No Finding':
                y[self.class_dict[i]] = 1
            else:
                print(i, y)

        # print(y)
        return image_tensor, y.float()

    def load_image(self, img_path, dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        if self.do_augmentation:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image_tensor = transform(image)

        return image_tensor
