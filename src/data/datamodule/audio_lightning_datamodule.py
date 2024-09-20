from typing import Any, Dict, Optional
import torch
from src.data.datamodule.abstract_lightning_datamodule import AbstractPLDataModule
import numpy as np
from functools import partial

class AudioPLDataModule(AbstractPLDataModule):

    def __init__(
        self,
        dataset: Dict[str, Any],
        supervision_ratio: list,
        data_type_sampling_probability: list,
        max_audio_length: int,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        reduction_factor: int = 2,
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
        
        self.reduction_factor = reduction_factor
        self.max_audio_length = max_audio_length
        self.collate_fn = partial(self.collate_fn, reduction_factor=self.reduction_factor, max_audio_length=self.max_audio_length)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and set up datasets. This method is called on every GPU in distributed training.

        :param stage: Current stage of training ('fit', 'validate', 'test', or 'predict').
        """
        self.processor_z = self.trainer.model.processor_z
        # possibly a tokenizer for text data ????
        self.processor_x = None
        self._setup(stage)

    @staticmethod
    def collate_fn(batch, processor_z, reduction_factor ,processor_x=None, max_audio_length=None):
        x_items = [item['x']for item in batch]
        if isinstance(batch[0]["z"], dict):
            z_items = [item['z']["array"] for item in batch]
        else:
            z_items = [item['z'] for item in batch]
            
        data_type = np.array([item['data_type'] for item in batch])
        data_type = torch.tensor(data_type)
        ids = np.array([item['id'] for item in batch])
        ids = torch.tensor(ids)
        
        processed_data = processor_z(
            text= x_items,
            audio_target= z_items,
            sampling_rate= 16000,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=max_audio_length,
            return_tensors="pt",
        )
        
        z_labels = processed_data['labels']
        z_labels_attention_mask = processed_data['decoder_attention_mask']
        to_remove = z_labels.shape[1] % reduction_factor

        if to_remove != 0:
            z_labels = z_labels[:, : - to_remove]
            z_labels_attention_mask = z_labels_attention_mask[:, : - to_remove]
                
        z_encodings = processor_z(
            audio = z_items,
            padding=True,
            sampling_rate = 16000 ,
            return_tensors="pt",
        )
        
        return {
            'x_ids': processed_data['input_ids'], # audio
            'x_mask': processed_data['attention_mask'],
            'z_ids': z_encodings["input_values"],
            'z_mask': z_encodings['attention_mask'],
            'z_labels': z_labels, #spectrogram
            'z_labels_attention_mask': z_labels_attention_mask,
            'data_type': data_type,
            'ids': ids
        }


if __name__ == "__main__":
    _ = AbstractPLDataModule("path/to/dataset", [0.8, 0.1], [0.7, 0.3])