import datasets
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, Callable
import numpy as np

import os

def just_load_dataset_from_hf_or_file_and_make_val_test_dataset(dataset_name_or_path, train_val_test_split=[0.8, 0.1, 0.1], **kwargs):

    input_label = kwargs.pop('input_label', 'input')
    output_label = kwargs.pop('output_label', 'output')

    # Check if it's a Hugging Face dataset
    if isinstance(dataset_name_or_path, str) or (os.path.isdir(dataset_name_or_path) and os.path.exists(os.path.join(dataset_name_or_path, 'dataset_dict.json'))): 
        try:
            dataset = datasets.load_from_disk(dataset_name_or_path)
        except:
            dataset = load_dataset(dataset_name_or_path, **kwargs)
    else:
        # Handle file path loading
        if os.path.isfile(dataset_name_or_path):
            # Single file case
            file_extension = os.path.splitext(dataset_name_or_path)[1].lower()
            if file_extension in ['.csv', '.tsv']:
                dataset = load_dataset('csv', data_files=dataset_name_or_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        elif 'data_files' in kwargs:
            # Multiple files case
            for split, file_name in kwargs['data_files'].items():
                kwargs['data_files'][split] = os.path.join(dataset_name_or_path, file_name)
            kwargs['data_files'] = dict(kwargs['data_files']) # resolve the paths. does nothing but without it load_dataset fails
            dataset = load_dataset('csv', **kwargs)
        else:
            raise ValueError("Invalid dataset_name_or_path or missing data_files in kwargs")
    if input_label != 'x':
        dataset = dataset.rename_column(input_label, "x")
    if output_label != 'z':
        dataset = dataset.rename_column(output_label, "z")

    # Split dataset according to train_val_test_split
    train_ratio, val_ratio, test_ratio = train_val_test_split
    
    # Split into train and test
    if not 'test' in dataset:
        train_test_split = train_ratio / (train_ratio + test_ratio)
        split = dataset['train'].train_test_split(1 - train_test_split)
        train_dataset, test_dataset = split['train'], split['test']
    else:
        train_dataset = dataset['train']
        test_dataset = dataset['test']

    # Split remaining train into train and validation
    if not 'validation' in dataset:
        train_val_split = train_ratio / (train_ratio + val_ratio)
        split = train_dataset.train_test_split(1 - train_val_split)
        train_dataset, val_dataset = split['train'], split['test']
    else:
        train_dataset = dataset['train']
        val_dataset = dataset['validation']

    # Create DatasetDict with the correct splits
    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    return dataset


def make_val_test_dataset(self):
    # Merge all subsets into 'train' if it's a DatasetDict
    if isinstance(self.dataset, dict):
        train_datasets = [ds for split, ds in self.dataset.items() if split != 'test' and split != 'validation']
        self.dataset = {
            'train': datasets.concatenate_datasets(train_datasets),
            'test': self.dataset.get('test', None),
            'validation': self.dataset.get('validation', None)
        }

    # Split according to train_val_test_split
    train_ratio, val_ratio, test_ratio = self.train_val_test_split

    # Split train into train and test if no test set exists
    if 'test' not in self.dataset or self.dataset['test'] is None:
        train_test_split = train_ratio / (train_ratio + test_ratio)
        split = self.dataset['train'].train_test_split(train_test_split)
        self.dataset['train'], self.dataset['test'] = split['train'], split['test']

    # Split remaining train into train and validation if no validation set exists
    if 'validation' not in self.dataset or self.dataset['validation'] is None:
        train_val_split = train_ratio / (train_ratio + val_ratio)
        split = self.dataset['train'].train_test_split(train_val_split)
        self.dataset['train'], self.dataset['validation'] = split['train'], split['test']

    # Reassign the dataset
    self.dataset = datasets.DatasetDict(self.dataset)


class AbstractDataset(Dataset):
    def __init__(self, dataset, supervision_ratio):
        self.dataset = dataset
        self.supervision_ratio = torch.tensor(supervision_ratio).float()
        self.assign_data_type()
        # self.random_sequence = self._create_random_batch_of_tokens(len(self.dataset), vocab_size=20, max_num_tokens=20)
        self.random_sequence = torch.ones(len(self.dataset), 20).long() * (-1)

    def __len__(self):
        return len(self.dataset)

    def assign_data_type(self):
        dataset_length = len(self.dataset)
        self.sup_len_xz = int(self.supervision_ratio[0] * dataset_length)
        self.sup_len_x = int((1 - self.supervision_ratio[0]) * (1 - self.supervision_ratio[1]) * dataset_length)

        self.data_type = np.ones((dataset_length, 2), dtype=np.bool_)
        self.data_type[self.sup_len_xz:self.sup_len_xz + self.sup_len_x] = np.array([1, 0], dtype=np.bool_)
        self.data_type[self.sup_len_xz + self.sup_len_x:] = np.array([0, 1], dtype=np.bool_)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data_type = self.data_type[idx]

        return {
            "id": idx,
            "x": item['x'],
            "y": self.random_sequence[idx],
            "z": item['z'],
            "data_type": data_type
        }
    
    def _create_random_batch_of_tokens(self, batchsize, vocab_size, max_num_tokens, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        random_batch_of_tokens = torch.randint(4, vocab_size, (batchsize, max_num_tokens))
        random_batch_of_tokens[:, 0] = 3
        ending_index = torch.randint(1, max_num_tokens, (batchsize,))
        for i in range(batchsize):
            random_batch_of_tokens[i, ending_index[i]] = 2
            random_batch_of_tokens[i, ending_index[i]+1:] = 0
        return random_batch_of_tokens

