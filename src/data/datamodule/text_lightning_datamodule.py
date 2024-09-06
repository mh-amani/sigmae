from typing import Any, Dict, Optional
import torch
from src.data.datamodule.abstract_lightning_datamodule import AbstractPLDataModule
import numpy as np

class TextPLDataModule(AbstractPLDataModule):

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
        super().__init__(
            dataset=dataset,
            supervision_ratio=supervision_ratio,
            data_type_sampling_probability=data_type_sampling_probability,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            **kwargs
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and set up datasets. This method is called on every GPU in distributed training.

        :param stage: Current stage of training ('fit', 'validate', 'test', or 'predict').
        """
        self.tokenizer_x = self.trainer.model.tokenizer_x
        self.tokenizer_z = self.trainer.model.tokenizer_z
        if self.kwargs.get('remove_long_data_points_and_print_stats', False):
            self.remove_long_data_points_and_print_stats(AbstractPLDataModule.Data)

        self._setup(stage)

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

    @staticmethod
    def collate_fn(batch, tokenizer_x, tokenizer_z):
        x_texts = [item['x'] for item in batch]
        z_texts = [item['z'] for item in batch]
        data_type = np.array([item['data_type'] for item in batch])
        data_type = torch.tensor(data_type)
        ids = np.array([item['id'] for item in batch])
        ids = torch.tensor(ids)
        
        x_encodings = tokenizer_x(x_texts, padding=True, return_tensors="pt", add_special_tokens=True)
        z_encodings = tokenizer_z(z_texts, padding=True, return_tensors="pt", add_special_tokens=True)
        
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