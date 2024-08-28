from typing import Any, Dict, Optional, Tuple, Callable
import torch
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
        collate_fn: Callable = None,
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
        :param collate_fn: The collate function to use in data loaders.
        :param seed: Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.supervision_ratio = supervision_ratio
        self.data_type_sampling_probability = torch.tensor(data_type_sampling_probability).float()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.seed = seed
        self.kwargs = kwargs

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        dataset_dict = hydra.utils.call(self.hparams.dataset, train_val_test_split=self.hparams.train_val_test_split) # only to test and debug, comment out before production

    def prepare_data(self) -> None:
        """
        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        _ = hydra.utils.instantiate(self.hparams.dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and set up datasets. This method is called on every GPU in distributed training.

        :param stage: Current stage of training ('fit', 'validate', 'test', or 'predict').
        """
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

        # After setting up the datasets, remove long data points and print statistics
        self.tokenizer_x = self.trainer.model.tokenizer_x
        self.tokenizer_z = self.trainer.model.tokenizer_z
        if self.kwargs.get('remove_long_data_points', False):
            self.remove_long_data_points()
        if self.kwargs.get('print_max_lengths', False):
            self.print_max_lengths()
 

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

    def remove_long_data_points(self):
        """
        Remove data points that exceed the maximum length for x or z.
        This method modifies the train, validation, and test datasets in-place.
        """
        for split, dataset in [('train', self.data_train), ('val', self.data_val), ('test', self.data_test)]:
            if dataset is None:
                continue

            filtered_data = []
            counter = 0
            for i in range(len(dataset)):
                collated_batch = self.collate_fn([dataset[i]], cut_to_max_length=False)
                if (collated_batch['x_ids'].shape[1] <= self.dataset_parameters["max_x_length"] and 
                    collated_batch['z_ids'].shape[1] <= self.dataset_parameters["max_z_length"]):
                    data_i = dataset[i]
                    data_i['id'] = counter
                    counter += 1
                    filtered_data.append(data_i)
            
            dataset.datum[split] = filtered_data
            setattr(self, f'data_{split}', dataset)

    def print_max_lengths(self):
        """
        Print statistics about the maximum lengths of x and z in the training dataset,
        and the percentage of data points that exceed the specified maximum lengths.
        """
        if self.data_train is None:
            print("Training data not available. Please run setup() first.")
            return

        max_x_length = 0
        larger_than_max_x_length = 0
        max_z_length = 0
        larger_than_max_z_length = 0
        total_samples = len(self.data_train)

        for i in range(total_samples):
            collated_batch = self.collate_fn([self.data_train[i]], cut_to_max_length=False)
            x_length = collated_batch['x_ids'].shape[1]
            z_length = collated_batch['z_ids'].shape[1]

            max_x_length = max(max_x_length, x_length)
            max_z_length = max(max_z_length, z_length)

            if x_length > self.dataset_parameters["max_x_length"]:
                larger_than_max_x_length += 1
            if z_length > self.dataset_parameters["max_z_length"]:
                larger_than_max_z_length += 1

        print(f"Max x length before cut-off: {max_x_length}")
        print(f"Max z length before cut-off: {max_z_length}")
        print(f"Percentage of x data points that are cut off: {larger_than_max_x_length / total_samples:.2%}")
        print(f"Percentage of z data points that are cut off: {larger_than_max_z_length / total_samples:.2%}")

if __name__ == "__main__":
    _ = AbstractPLDataModule("path/to/dataset", [0.8, 0.1], [0.7, 0.3])