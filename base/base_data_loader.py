import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class Base_dataset(Dataset):
    def __init__(self, config, mode, classes):
        """

        Args:
            config (yaml): configuration file
            mode (string): 'train' 'validation' 'test'
            classes (int):
        """
        super(Base_dataset, self).__init__()

        self.classes = classes

        self.mode = mode
        self.list_IDs, self.list_glosses = [], []
        self.config = config

    def __len__(self):
        """

        Returns:
        (int) Number of  samples
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Basically, __getitem__ of a torch dataset.
        Args:
            index (int): Index of the sample to be loaded.
        """

        raise NotImplementedError

    def feature_loader(self, index):
        """

        Args:
            index (int): Index of the sample to be loaded.
        """
        raise NotImplementedError

    def video_loader(self, index):
        """
        Load video function
        Args:
            index (int): Index of the sample to be loaded.
        """
        raise NotImplementedError
