from typing import Any, Dict, Optional, Tuple, Callable
import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import hydra
from src.data.dataset import AbstractDataset
from src.data.samplers import RandomSupervisionSampler, RandomSupervisionSamplerDDP

class AbstractPLDataModule(LightningDataModule):
    """
    A `LightningDataModule` for custom datasets with supervision ratios.

    This module implements the key methods required by PyTorch Lightning:
    - prepare_data
    - setup
    - train_dataloader
    - val_dataloader
    - test_dataloader
    - teardown
    - state_dict
    - load_state_dict

    It also handles supervision ratios and custom sampling for semi-supervised learning.
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        supervision_ratio: list,
        data_type_sampling_probability: list,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        **kwargs
    ) -> None:
        """Initialize the CustomLightningDataModule.

        :param dataset_name_or_path: The name or path of the dataset to load.
        :param supervision_ratio: The supervision ratio for the training set.
        :param data_type_sampling_probability: The sampling probabilities for different data types.
        :param batch_size: The batch size. Defaults to 32.
        :param num_workers: The number of workers for data loading. Defaults to 0.
        :param pin_memory: Whether to pin memory in data loaders. Defaults to False.
        :param seed: Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.supervision_ratio = supervision_ratio
        self.data_type_sampling_probability = torch.tensor(data_type_sampling_probability).float()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.kwargs = kwargs

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        # dataset_dict = hydra.utils.call(self.hparams.dataset, train_val_test_split=self.hparams.train_val_test_split) # only to test and debug, comment out before production

    def prepare_data(self) -> None:
        """
        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        I am doing it in a way, but i think this is the only way!
        """
        # moved to setup
        # AbstractPLDataModule.Data = hydra.utils.instantiate(self.hparams.dataset, train_val_test_split=self.hparams.train_val_test_split)
        # After setting up the datasets, remove long data points and print statistics
        # self.tokenizer_x = self.trainer.model.tokenizer_x
        # self.tokenizer_z = self.trainer.model.tokenizer_z
        # if self.kwargs.get('remove_long_data_points_and_print_stats', False):
        #     self.remove_long_data_points_and_print_stats(AbstractPLDataModule.Data)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and set up datasets. This method is called on every GPU in distributed training.

        :param stage: Current stage of training ('fit', 'validate', 'test', or 'predict').
        """
        self.tokenizer_x = self.trainer.model.tokenizer_x
        self.tokenizer_z = self.trainer.model.tokenizer_z
        if self.kwargs.get('remove_long_data_points_and_print_stats', False):
            self.remove_long_data_points_and_print_stats(AbstractPLDataModule.Data)

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = hydra.utils.call(self.hparams.dataset, train_val_test_split=self.hparams.train_val_test_split)
            
            self.data_train = AbstractDataset(dataset['train'], self.supervision_ratio)
            self.data_val = AbstractDataset(dataset['validation'], [1.0, 0.0])  # Fully supervised
            self.data_test = AbstractDataset(dataset['test'], [1.0, 0.0])  # Fully supervised

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.kwargs.get('use_ddp', False):
            sampler = randomSupervisionSamplerDDP(
                dataset=self.data_train,
                data_type_sampling_probability=self.data_type_sampling_probability,
                batch_size=self.batch_size_per_device
            )
        else:
            sampler = randomSupervisionSampler(
                self.data_train,
                self.data_type_sampling_probability,
                batch_size=self.batch_size_per_device,
                generator=g
            )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._get_dataloader(self.data_val)

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._get_dataloader(self.data_test)

    def _get_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Helper method to create a dataloader for validation and test sets.

        :param dataset: The dataset to create a dataloader for.
        :return: A DataLoader instance.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after fit or test.

        :param stage: The stage being torn down ('fit', 'validate', 'test', or 'predict').
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def remove_long_data_points_and_print_stats(self, dataset):
        """
        Remove long data points and print statistics for all splits.
        """
        for split, split_dataset in dataset.items():
            if split_dataset is None:
                continue

            stats = {
                'max_x_length': 0,
                'max_z_length': 0,
                'larger_than_max_x': 0,
                'larger_than_max_z': 0,
                'total_samples': len(split_dataset),
                'filtered_data': []
            }

            for i, item in enumerate(split_dataset):
                x_length, z_length = self._get_lengths(item)
                self._update_stats(stats, x_length, z_length)

                if self._is_valid_length(x_length, z_length):
                    item['id'] = len(stats['filtered_data'])
                    stats['filtered_data'].append(item)

            self._update_dataset(split, split_dataset, stats['filtered_data'])
            self._print_stats(split, stats)

    def _get_lengths(self, item):
        x_encoding = self.tokenizer_x(item['x'], truncation=False, return_tensors="pt")
        z_encoding = self.tokenizer_z(item['z'], truncation=False, return_tensors="pt")
        # x_encoding, z_encoding = self.collate_fn([item])
        return x_encoding['input_ids'].shape[1], z_encoding['input_ids'].shape[1]

    def _update_stats(self, stats, x_length, z_length):
        stats['max_x_length'] = max(stats['max_x_length'], x_length)
        stats['max_z_length'] = max(stats['max_z_length'], z_length)
        stats['larger_than_max_x'] += x_length > self.kwargs["max_x_length"]
        stats['larger_than_max_z'] += z_length > self.kwargs["max_z_length"]

    def _is_valid_length(self, x_length, z_length):
        return (x_length <= self.kwargs["max_x_length"] and 
                z_length <= self.kwargs["max_z_length"])

    def _update_dataset(self, split, split_dataset, filtered_data):
        split_dataset.datum = filtered_data
        setattr(self, f'data_{split}', split_dataset)

    def _print_stats(self, split, stats):
        print(f"\nStatistics for {split} split:")
        print(f"Max x length before cut-off: {stats['max_x_length']}")
        print(f"Max z length before cut-off: {stats['max_z_length']}")
        print(f"Percentage of x data points that are cut off: {stats['larger_than_max_x'] / stats['total_samples']:.2%}")
        print(f"Percentage of z data points that are cut off: {stats['larger_than_max_z'] / stats['total_samples']:.2%}")
        print(f"Total samples after filtering: {len(stats['filtered_data'])}")

    def collate_fn(self, batch):
        x_texts = [item['x'] for item in batch]
        z_texts = [item['z'] for item in batch]
        data_type = np.array([item['data_type'] for item in batch])
        data_type = torch.tensor(data_type)
        ids = np.array([item['id'] for item in batch])
        ids = torch.tensor(ids)
        
        x_encodings = self.tokenizer_x(x_texts, padding=True, return_tensors="pt", add_special_tokens=True)
        z_encodings = self.tokenizer_z(z_texts, padding=True, return_tensors="pt", add_special_tokens=True)
        
        return {
            'x_ids': x_encodings['input_ids'],
            'x_mask': x_encodings['attention_mask'],
            'z_ids': z_encodings['input_ids'],
            'z_mask': z_encodings['attention_mask'],
            'data_type': data_type,
            'ids': ids
        }

if __name__ == "__main__":
    _ = AbstractPLDataModule("path/to/dataset", [0.8, 0.1], [0.7, 0.3])