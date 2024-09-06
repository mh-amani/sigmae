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
        self.processor_x = self.trainer.model.processor_x
        self.processor_z = self.trainer.model.processor_z
        self._setup(stage)

    @staticmethod
    def collate_fn(batch, processor_z, processor_x=None):
        x_items = [item['x'] for item in batch]
        z_items = [item['z'] for item in batch]
        data_type = np.array([item['data_type'] for item in batch])
        data_type = torch.tensor(data_type)
        ids = np.array([item['id'] for item in batch])
        ids = torch.tensor(ids)
        
        if processor_x is not None:
            x_encodings = processor_x(x_items, padding=True, return_tensors="pt", add_special_tokens=True)
        else:
            x_encodings = torch.tensor(x_items)
        z_encodings = processor_z(z_items, padding=True, return_tensors="pt", add_special_tokens=True)
        
        return {
            'x': x_encodings,
            'z': z_encodings,
            'data_type': data_type,
            'ids': ids
        }


if __name__ == "__main__":
    _ = AbstractPLDataModule("path/to/dataset", [0.8, 0.1], [0.7, 0.3])