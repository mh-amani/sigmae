
import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .sigmae_lit_module_im_to_text_cnn import SigmaeLitModuleImageToTextBase
from src.models.components.vit_unprocessor import UnViTImageProcessor
from PIL import Image
import numpy as np


class SigmaeLitModuleImageToTextFullCNN(SigmaeLitModuleImageToTextBase):
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
        self.reshape_img_size = (self.image_size)//(len(self.hparams.models_config.sequence_model_zx.model.down_block_types))
        self.z_linear_head = torch.nn.Linear(in_features=self.hparams['model_params']['d_model'], out_features=(self.reshape_img_size)**2) # 
        # self.linear_head = torch.nn.Linear(in_features=self.hparams['model_params']['d_model'], out_features=self.image_size**2//4) 
        self.x_linear_head = torch.nn.Linear(in_features=(self.reshape_img_size)**2, out_features=self.hparams['model_params']['d_model']) #
            
    def forward(self, x, z, data_type=None, stage='learn', training_mode=None) -> torch.Tensor:
        outputs = {}
        labels = {}

        training_mode='zxz'
        
        vit_processed_z = self.processor_z(z, padding=True, return_tensors="pt", add_special_tokens=True)['pixel_values'].to(self.device)

        labels[training_mode] = {}
        outputs[training_mode] = {}

        # if training_mode == 'zxz' or training_mode == 'zxz_unteacherforced':
        # if training_mode == 'zxz':
        y_img_latent = self.auto_reg_wrapped_model_zx(vit_processed_z,)
   
        y_continous_embeddings = self.x_linear_head(y_img_latent.flatten(-2, -1))
        y_output = self.auto_reg_wrapped_model_zx.output_discretizer(y_continous_embeddings)
        
        eos_token_id = self.auto_reg_wrapped_model_zx.control_token_ids['output_eos_token_id']
        pad_token_id = self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
        
        predicted_ids = y_output['id']
        #find the first eos token of each sequence
        # Create a mask where EOS token appears
        eos_mask = (predicted_ids == eos_token_id).int()

        # Get the first occurrence of EOS token for each sample
        first_eos_indices = torch.argmax(eos_mask, dim=1)

        # If no EOS token found, argmax will return 0, so we can mask it out where eos_mask.sum(dim=1) == 0
        first_eos_indices[eos_mask.sum(dim=1) == 0] = predicted_ids.size(1)
        
        # Create a tensor to hold the indices at which to start replacing with PAD
        seq_indices = torch.arange(predicted_ids.size(1), device=predicted_ids.device).unsqueeze(0)
        replace_mask = seq_indices > first_eos_indices.unsqueeze(1)
        
        # Replace tokens based on the mask
        true_ids = predicted_ids.clone()
        true_ids[replace_mask] = pad_token_id

        p_not_eos = (1.0 - y_output["score"][...,eos_token_id]).cumprod(dim=-1)
        # p_not_eos = 
        
        #make attention mask
        y_attention_mask = (torch.logical_not(true_ids == pad_token_id))
        if self.auto_reg_wrapped_model_zx.config["soft_average"]['p_eos_forward']:
            y_attention_mask = p_not_eos
        elif self.auto_reg_wrapped_model_zx.config["soft_average"]['p_eos_backward']:
            y_attention_mask = y_attention_mask + (p_not_eos - p_not_eos.detach())
            
        y_vector_encoder = y_output['vector_encoder'] * y_attention_mask.unsqueeze(-1)
        
        embeddings = y_vector_encoder
        # pad the embeddings to self.hparams['model_params']['max_x_length'], with additional pad embeddings: self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
        pad_embeddings = torch.zeros(embeddings.shape[0], self.hparams['model_params']['max_x_length'] - embeddings.shape[1], embeddings.shape[2]).to(self.device)
        pad_embeddings[..., :] = self.discretizer_x.decoder_embedding.weight[self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']]
        embeddings = torch.cat([embeddings, pad_embeddings], dim=1)
        embeddings = self.z_linear_head(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], self.reshape_img_size, self.reshape_img_size)
        decoder_outputs = self.auto_reg_wrapped_model_xz(embeddings, )

        outputs[training_mode] = {'id_z': decoder_outputs, 'image':self.unprocessor_z.unprocess(decoder_outputs.detach().cpu().numpy()),     # decoder_outputs * 255.0/2 + 255.0/2,
                                     'image_caption': y_output['id'], 'logit': decoder_outputs}

        # penalize the patches
        labels[training_mode]= {'logit': vit_processed_z, 'image': self.unprocessor_z.unprocess(vit_processed_z.cpu().numpy())}


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

