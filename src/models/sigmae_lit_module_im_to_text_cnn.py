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

class SigmaeLitModuleImageToTextCNN(SigmaeLitModuleImageToTextBase):
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


    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.processor_z = hydra.utils.instantiate(models_config.sequence_model_zx.processor, _recursive_=False)
        self.unprocessor_z = UnViTImageProcessor(self.processor_z,original_height=self.hparams['model_params']['image_size'],original_width=self.hparams['model_params']['image_size'],)

        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)
        self.image_size = self.processor_z.size['height']
        self.linear_head = torch.nn.Linear(in_features=self.hparams['model_params']['d_model'], out_features=self.image_size**2//4)

    def forward(self, x, z, data_type=None, stage='learn', training_mode=None) -> torch.Tensor:
        outputs = {}
        labels = {}

        vit_processed_z = self.processor_z(z, padding=True, return_tensors="pt", add_special_tokens=True)['pixel_values'].to(self.device)
        
        
        # Use specified training_mode during validation; otherwise, select randomly
        # if training_mode is None:
        #     cointoss = torch.rand(1).item()
        #     for i in range(len(self.training_modes)):
        #         if cointoss < self.training_probs[i]:
        #             training_mode = self.training_modes[i]
        #             break
        #         cointoss -= self.training_probs[i]
        training_mode='zxz'

        labels[training_mode] = {}
        outputs[training_mode] = {}

        # if training_mode == 'zxz' or training_mode == 'zxz_unteacherforced':
        # if training_mode == 'zxz':
        y_output = self.auto_reg_wrapped_model_zx(input_embeds_enc=vit_processed_z,)

        embeddings = y_output['vector_encoder']
        # pad the embeddings to self.hparams['model_params']['max_x_length'], with additional pad embeddings: self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
        pad_embeddings = torch.zeros(embeddings.shape[0], self.hparams['model_params']['max_x_length'] - embeddings.shape[1], embeddings.shape[2]).to(self.device)
        pad_embeddings[..., :] = self.discretizer_x.decoder_embedding.weight[self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']]
        embeddings = torch.cat([embeddings, pad_embeddings], dim=1)
        embeddings = self.linear_head(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], self.image_size//2, self.image_size//2)
        decoder_outputs = self.auto_reg_wrapped_model_xz(embeddings, )
        outputs[training_mode] = {'id_z': decoder_outputs, 'image':self.unprocessor_z.unprocess(decoder_outputs.detach().cpu().numpy()),     # decoder_outputs * 255.0/2 + 255.0/2,
                                     'image_caption': y_output['id'], 'logit': decoder_outputs}

        # penalize the patches
        labels[training_mode]= {'logit': vit_processed_z, 'image': self.unprocessor_z.unprocess(vit_processed_z.cpu().numpy())}

        # elif training_mode == 'xzx':
        #     random_batch_of_tokens = self._create_random_batch_of_tokens(batchsize=z_patches_embeds.shape[0], max_num_tokens=10)
        #     outputs[training_mode] = self.symbolic_autoencoder_wrapper_xzx(x_ids=random_batch_of_tokens, z_ids=random_batch_of_tokens, teacher_force_z=True)
        #     outputs[training_mode]['image'] = self.fold(outputs[training_mode]['id_y'].permute(0, 2, 1)) * 255.0/2 + 255.0/2 #(self.fold(z_patches.permute(0, 2, 1)) * 255.0/2 + 255.0/2 - z.permute(0,3,1,2)).var(): tensor(1.1724e-12, device='cuda:0')
        #     labels[training_mode]['image'] = z.permute(0,3,1,2)
        #     outputs[training_mode]['image_caption'] = random_batch_of_tokens
        #     # penalize the patches
        #     outputs[training_mode]['logit'] = outputs['xzx']['logit_z']
        #     labels[training_mode]['logit']  = random_batch_of_tokens

        return outputs, labels
    
    def model_step(self, batch, batch_idx=-1, stage='learn', log_things=True):
        x, z, data_type = batch['x'], batch['z'], batch['data_type']
        z = z[..., 0:self.hparams['model_params']['num_channels']]
        data_type = torch.all(data_type, dim=0)

        total_loss = 0

        # if stage == 'val':
        #     # Loop over each training mode during validation
        #     for training_mode in self.training_modes:
        #         outputs, labels = self.forward(x, z, data_type, stage=stage, training_mode=training_mode)

        #         # Compute loss
        #         loss = self.compute_loss(outputs, labels, training_mode, stage)
        #         total_loss += loss

        #         # Log metrics
        #         log_kwargs = self.logging_kwargs[stage]
        #         self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)

        #         # Log images only for the first batch
        #         if batch_idx == 0:
        #             self._log_output_samples(
        #                 outputs[training_mode]['image'],
        #                 labels[training_mode]['image'],
        #                 outputs[training_mode]['image_caption'],
        #                 stage,
        #                 training_mode,
        #                 num_images=10
        #             )
        # else:
            # Existing behavior for training and testing
        outputs, labels = self.forward(x, z, data_type, stage=stage)
        training_mode = list(outputs.keys())[0]
        loss = self.compute_loss(outputs, labels, training_mode, stage)
        total_loss = loss

        # Log metrics
        log_kwargs = self.logging_kwargs[stage]
        self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)

        # Conditional logging during training
        if log_things and self.global_step % self.logging_freq == 0:
            self._log_output_samples(
                outputs[training_mode]['image'],
                labels[training_mode]['image'],
                outputs[training_mode]['image_caption'],
                stage,
                training_mode,
                num_images=10
            )

        return total_loss
   
    def make_transform_image_patch_to_input_image(self):
        def transform_image_patch_to_input_image(outputs):
            image_patches = outputs['id']
            processed_image = self.fold(image_patches.permute(0, 2, 1))
            return {'vector_encoder': processed_image, 'output_attention_mask': None}
        return transform_image_patch_to_input_image

