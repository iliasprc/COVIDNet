import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import read_filepaths2

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class COVIDxDataset(Dataset):
    """
    Dataset for reading the COVIDx images and labels.
    """
    def __init__(self, config: dict, mode: str, dim: tuple=(224, 224)):
        """
        Initializes the dataset.

        Args:
            config (dict): Dictionary that contains the configuration.
            mode (str): Mode in which the dataset is used (can be "train" or "test").
            dim (tuple, optional): Dimensions of the image.

        Returns:
            None
        """
        self.config = config
        self.root = self.config.dataset.input_data + '/' + mode + '/'

        self.dim = dim
        self.class_dict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        self.CLASSES = len(self.class_dict)
        testfile = './data/test_split.txt'
        trainfile = './data/train_split.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths2(trainfile)
            self.do_augmentation = True
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths2(testfile)

            self.do_augmentation = False
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self) -> int:
        """
        Computes the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple:
        """
        Gets an image and its corresponding label at the given index.

        Args:
            index (int): The index to retrieve the data from.

        Returns:
            tuple: Contains the image tensor and label tensor.
        """

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(self.class_dict[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path: str, dim: tuple) -> torch.Tensor:
        """
        Loads an image, applies transformations and converts it to tensor.

        Args:
            img_path (str): The path of the image to be loaded.
            dim (tuple): Dimensions of the image.


        Returns:
            torch.Tensor: The processed image tensor.
        """
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
