import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .sigmae_lit_module_im_to_text_base import SigmaeLitModuleImageToTextBase
from src.models.components.vit_unprocessor import UnViTImageProcessor
from PIL import Image
import numpy as np

class SigmaeLitModuleImageToTextAutoRegImage(SigmaeLitModuleImageToTextBase):
    def __init__(
        self,
        models_config,
        model_params,
        optimizer,
        scheduler,
    ) -> None:
        super().__init__(
            models_config,
            model_params,
            optimizer,
            scheduler,
        )
        # for debugging and accessing the model's parameters and gradients.
        # self.automatic_optimization=False

    def _initialize_metrics(self) -> None:
        # loss function is L1Loss
        self.continuous_criterion = torch.nn.L1Loss(reduction='mean')
        self.discrete_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.training_modes = self.hparams['model_params']['training_modes']
        self.training_probs = self.hparams['model_params']['training_probs']
        self.logging_freq = self.hparams['model_params']['logging_freq']
        self.accuracies, self.losses = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        for split in ['learn', 'val', 'test']: # you can't use 'train' as a key ... it's a reserved word
            self.accuracies.update({split: torch.nn.ModuleDict()})
            self.losses.update({split: torch.nn.ModuleDict()})
            for training_mode in self.training_modes:
                # if space.endswith('x'): # text space
                #     self.accuracies[split].update({space: MaxMetric(top_k=1)})
                # else:
                #     self.accuracies[split].update({space: StructuralSimilarityIndexMeasure()})
                self.losses[split].update({training_mode:  MeanMetric()})

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.processor_z = hydra.utils.instantiate(models_config.sequence_model_zx.processor, _recursive_=False)
        self.unprocessor_z = unprocessor = UnViTImageProcessor(self.processor_z,original_height=self.hparams['model_params']['image_size'],original_width=self.hparams['model_params']['image_size'],)

        self._initialize_autoreg_wrapped_models(models_config)
        # self._initialize_symbolic_autoencoder_wrappers(models_config)
        # self.image_size = self.hparams['model_params']['image_size']
        self.image_size = self.processor_z.size['height']
        self.unfold = torch.nn.Unfold(kernel_size=(4,4), stride=(4,4))
        self.fold = torch.nn.Fold(output_size=(28, 28), kernel_size=(4,4), stride=(4,4))
        self.linear_head = torch.nn.Linear(in_features=self.hparams['model_params']['d_model'], out_features=self.image_size**2//4)

    def forward(self, x, z, data_type=None, stage='learn', training_mode=None) -> torch.Tensor:
        outputs = {}
        labels = {}

        vit_processed_z = self.processor_z(z, padding=True, return_tensors="pt", add_special_tokens=True)['pixel_values'].to(self.device)
        z_patches = (self.unfold(z.permute(0,3,1,2)).permute(0,2,1) - 255.0/2) / (255.0/2)
        z_patches_embeds = self.discretizer_z.decoder_embedding(z_patches)

        # Use specified training_mode during validation; otherwise, select randomly
        if training_mode is None:
            cointoss = torch.rand(1).item()
            for i in range(len(self.training_modes)):
                if cointoss < self.training_probs[i]:
                    training_mode = self.training_modes[i]
                    break
                cointoss -= self.training_probs[i]

        labels[training_mode] = {}

        if training_mode == 'zxz' or training_mode == 'zxz_unteacherforced':
            if training_mode == 'zxz':
                outputs[training_mode] = self.symbolic_autoencoder_wrapper_zxz(x_embeds_enc=vit_processed_z,  z_embeds_dec=z_patches_embeds if stage=='learn' else None, 
                                                                    z_attention_mask=None, teacher_force_z=(stage=='learn'),)
                outputs[training_mode]['id_z'] = outputs[training_mode]['id_z'][:, :-1, ...] if stage=='learn' else outputs[training_mode]['id_z']
            elif training_mode == 'zxz_unteacherforced':
                # mini_batch_size = 100
                # vit_processed_z = vit_processed_z[:mini_batch_size]
                # z_patches = z_patches[:mini_batch_size]
                outputs[training_mode] = self.symbolic_autoencoder_wrapper_zxz(x_embeds_enc=vit_processed_z,  teacher_force_z=False)
            output_image = self.fold(outputs[training_mode]['id_z'].permute(0, 2, 1))
            outputs[training_mode]['image'] = output_image * 255.0/2 + 255.0/2 #(self.fold(z_patches.permute(0, 2, 1)) * 255.0/2 + 255.0/2 - z.permute(0,3,1,2)).var(): tensor(1.1724e-12, device='cuda:0')
            labels[training_mode]['image'] = z.permute(0,3,1,2)
            outputs[training_mode]['image_caption'] = outputs[training_mode]['id_y']
            # penalize the patches
            outputs[training_mode]['logit'] = outputs[training_mode]['id_z']
            labels[training_mode]['logit']  = z_patches

        elif training_mode == 'xzx':
            random_batch_of_tokens = self._create_random_batch_of_tokens(batchsize=z_patches_embeds.shape[0], max_num_tokens=10)
            outputs[training_mode] = self.symbolic_autoencoder_wrapper_xzx(x_ids=random_batch_of_tokens, z_ids=random_batch_of_tokens, teacher_force_z=True)
            outputs[training_mode]['image'] = self.fold(outputs[training_mode]['id_y'].permute(0, 2, 1)) * 255.0/2 + 255.0/2 #(self.fold(z_patches.permute(0, 2, 1)) * 255.0/2 + 255.0/2 - z.permute(0,3,1,2)).var(): tensor(1.1724e-12, device='cuda:0')
            labels[training_mode]['image'] = z.permute(0,3,1,2)
            outputs[training_mode]['image_caption'] = random_batch_of_tokens
            # penalize the patches
            outputs[training_mode]['logit'] = outputs['xzx']['logit_z']
            labels[training_mode]['logit']  = random_batch_of_tokens

        return outputs, labels
    

        
    def _create_random_batch_of_tokens(self, batchsize, max_num_tokens):
        random_batch_of_tokens = torch.randint(4, self.discretizer_x.vocab_size, (batchsize, max_num_tokens)).to(self.device)
        random_batch_of_tokens[:, 0] = 3
        ending_index = torch.randint(1, max_num_tokens, (batchsize,))
        for i in range(batchsize):
            random_batch_of_tokens[i, ending_index[i]] = 2
            random_batch_of_tokens[i, ending_index[i]+1:] = 0
        return random_batch_of_tokens

