import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .sigmae_lit_module_im_to_text_base import SigmaeLitModuleImageToTextBase
from src.models.components.vit_unprocessor import UnViTImageProcessor
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np
from torchmetrics.classification import Accuracy

# Import the VQGanVAE from Lucidrains' DALLE-pytorch repository
from dalle_pytorch import VQGanVAE

# Define a custom VQGAN VAE class to handle the forward pass and loss computation
class CustomVQGanVAE(VQGanVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # Encode the images
        z = self.model.encoder(x)
        
        # Quantize the latents
        quant_z, commit_loss, indices = self.model.quantize(z)
        indices = indices[-1].view(x.shape[0], z.shape[2], z.shape[3])
        # torch.round(quant_z[0][:, :, 0].T - self.model.quantize.embedding.weight[738], decimals=2) # almost zero
        # Decode the quantized latents
        recon_images = self.model.decoder(quant_z)
        
        # Compute reconstruction loss
        # recon_loss = torch.nn.functional.mse_loss(recon_images, x)
        
        # Total loss combines reconstruction and commitment losses
        # loss = recon_loss + commit_loss.mean()
        loss = commit_loss.mean()
        
        return {
            'reconstructions': recon_images,
            'loss': loss,
            'commitment_loss': commit_loss.mean(),
            'code_indices': indices
        }


class SigmaeLitModuleImageToTextOnVQVAETokens(SigmaeLitModuleImageToTextBase):
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
        self.processor_z = ViTImageProcessor(size={"height": 256, "width": 256})
        self.unprocessor_z = UnViTImageProcessor(self.processor_z,original_height=self.hparams['model_params']['image_size'],original_width=self.hparams['model_params']['image_size'],)
        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)
        
        self.vqvae = CustomVQGanVAE().to(self.device)
        self.vqvae.train()
        self.training_modes = ['vqvae', 'xzx']
        self.training_probs = [0.0, 1.0]

    def _initialize_metrics(self) -> None:
        # loss function is L1Loss
        self.continuous_criterion = torch.nn.L1Loss(reduction='mean')
        self.discrete_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.logging_freq = self.hparams['model_params']['logging_freq']
        self.accuracies, self.losses = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        for split in ['learn', 'val', 'test']: # you can't use 'train' as a key ... it's a reserved word
            self.accuracies.update({split: torch.nn.ModuleDict()})
            self.losses.update({split: torch.nn.ModuleDict()})
            for training_mode in self.training_modes:
                self.losses[split].update({training_mode:  MeanMetric()})
                if training_mode.endswith('x'): # text space
                    self.accuracies[split].update({training_mode: MaxMetric()})
                    self.accuracies[split].update({training_mode: Accuracy(task="multiclass", num_classes=1024)})
                # else:
                #     self.accuracies[split].update({training_mode: StructuralSimilarityIndexMeasure()})
                

    
    def forward(self, x, z, data_type=None, stage='learn', training_mode=None) -> torch.Tensor:
        outputs = {}
        labels = {}

        vit_processed_z = self.processor_z(z, padding=True, return_tensors="pt", add_special_tokens=True)['pixel_values'].to(self.device)
    
        # Use specified training_mode during validation; otherwise, select randomly
        if self.current_epoch == 0:
            training_mode = 'vqvae'
        elif training_mode is None:
            training_mode = 'xzx'
            # cointoss = torch.rand(1).item()
            # for i in range(len(self.training_modes)):
            #     if cointoss < self.training_probs[i]:
            #         training_mode = self.training_modes[i]
            #         break
            #     cointoss -= self.training_probs[i]

        labels[training_mode] = {}

        if training_mode == 'vqvae':
            outputs[training_mode] = {}
            labels[training_mode] = {}
            output = self.vqvae(vit_processed_z)
            loss = output['loss']
            recon_images = output['reconstructions']
            code_indices = output['code_indices']

            outputs[training_mode]['image'] = self.unprocessor_z.unprocess(recon_images.detach().cpu().numpy())
            labels[training_mode]['image'] = self.unprocessor_z.unprocess(vit_processed_z.detach().cpu().numpy())
            outputs[training_mode]['image_caption'] = code_indices
            # penalize the patches
            outputs[training_mode]['logit'] = recon_images
            labels[training_mode]['logit']  = vit_processed_z

        if training_mode == 'xzx':
            # self.vqvae.eval()
            with torch.no_grad():
                codes = self.vqvae(vit_processed_z)['code_indices'].reshape(z.shape[0], -1) # .detach()
                bos_token_id = self.auto_reg_wrapped_model_xz.control_token_ids['output_bos_token_id']
                prepended_codes = torch.cat([bos_token_id*torch.ones((codes.shape[0], 1), dtype=codes.dtype, device=codes.device), codes], dim=1)
            # torch.cuda.empty_cache()
            zxz_outputs = self.symbolic_autoencoder_wrapper_xzx(x_ids=prepended_codes, 
                                                                z_ids=prepended_codes, teacher_force_z=True)
            outputs[training_mode] = zxz_outputs
            outputs[training_mode]['logit'] = outputs[training_mode]['logit_z']
            labels[training_mode]['logit'] = prepended_codes
            
            recon_codes = outputs[training_mode]['id_z'][:, :-1]
            recond_embeds = self.vqvae.model.quantize.embedding(recon_codes.reshape(codes.shape[0], 16, 16)).permute(0, 3, 1, 2)
            
            with torch.no_grad():
                recon_images = self.vqvae.model.decoder(recond_embeds)

            outputs[training_mode]['image'] = self.unprocessor_z.unprocess(recon_images.detach().cpu().numpy())
            labels[training_mode]['image'] = self.unprocessor_z.unprocess(vit_processed_z.detach().cpu().numpy())
            outputs[training_mode]['image_caption'] = recon_codes
            

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

        if stage == 'val':
            # Loop over each training mode during validation
            # for training_mode in self.training_modes:
            outputs, labels = self.forward(x, z, data_type, stage=stage)
            training_mode = list(outputs.keys())[0]
            # Compute loss
            loss = self.compute_loss(outputs, labels, training_mode, stage)
            total_loss += loss

            # Log metrics
            log_kwargs = self.logging_kwargs[stage]
            self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)
            if training_mode.endswith('x'):
                self.log(f"{stage}/{training_mode}/accuracy", self.accuracies[stage][training_mode], **log_kwargs)

            # Log images only for the first batch
            if batch_idx == 0:
                self._log_output_samples(
                    outputs[training_mode]['image'],
                    labels[training_mode]['image'],
                    outputs[training_mode]['image_caption'],
                    stage,
                    training_mode,
                    num_images=10
                )
        else:
        # Existing behavior for training and testing
            outputs, labels = self.forward(x, z, data_type, stage=stage)
            training_mode = list(outputs.keys())[0]
            loss = self.compute_loss(outputs, labels, training_mode, stage)
            total_loss = loss

            # Log metrics
            log_kwargs = self.logging_kwargs[stage]
            self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)
            if training_mode.endswith('x'):
                self.log(f"{stage}/{training_mode}/accuracy", self.accuracies[stage][training_mode], **log_kwargs)

            # Conditional logging during training
            # if log_things and self.global_step % self.logging_freq == 0:
            #     self._log_output_samples(
            #         outputs[training_mode]['image'],
            #         labels[training_mode]['image'],
            #         outputs[training_mode]['image_caption'],
            #         stage,
            #         training_mode,
            #         num_images=10
            #     )
            if batch_idx == 0:
                self._log_output_samples(
                    outputs[training_mode]['image'],
                    labels[training_mode]['image'],
                    outputs[training_mode]['image_caption'],
                    stage,
                    training_mode,
                    num_images=10
                )

        return total_loss
    
    def compute_loss(self, outputs, labels, training_mode, stage):
        if training_mode.endswith('x'):  # Text space
            non_pad_mask = labels[training_mode]['logit'][..., 1:] != self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
            labels_for_loss = labels[training_mode]['logit'][..., 1:].clone()
            labels_for_loss[~non_pad_mask] = -100
            loss = self.discrete_criterion(
                outputs[training_mode]['logit'][..., :-1, :].permute(0, 2, 1),
                labels_for_loss
            )
            self.accuracies[stage][training_mode](outputs[training_mode]['logit'][..., :-1, :].argmax(dim=-1), labels[training_mode]['logit'][..., 1:])
        else:  # Continuous space
            loss = self.continuous_criterion(
                outputs[training_mode]['logit'],
                labels[training_mode]['logit']
            )

        self.losses[stage][training_mode](loss)
        return loss


    def _prepare_to_show_output_samples(self, z_pred, z_true, symbolic_sequence, num_images=10) -> None:
        # log 10 images every 2 epochs
        eos_token_id = -1
        num_images = min(num_images, len(z_true))
        
        combined_images = []
        logged_sequences = []
        for i in range(num_images):
            # Convert tensors or arrays to NumPy arrays if needed
            true_img = z_true[i].detach().cpu().numpy() if not isinstance(z_true[i], np.ndarray) else z_true[i]
            pred_img = z_pred[i].detach().cpu().numpy() if not isinstance(z_pred[i], np.ndarray) else z_pred[i]

            # If the image is in format (C, H, W), convert it to (H, W, C)
            if true_img.shape[0] == 3:  # Assuming color images with 3 channels
                true_img = np.transpose(true_img, (1, 2, 0))
            if pred_img.shape[0] == 3:
                pred_img = np.transpose(pred_img, (1, 2, 0))
            if true_img.shape[0] == 1:  # Assuming grayscale images with 1 channel
                true_img = np.squeeze(true_img, axis=0)
            if pred_img.shape[0] == 1:
                pred_img = np.squeeze(pred_img, axis=0)

            # Convert arrays to 0-255 range (assuming the input is float)
            if true_img.max() <= 1.0:
                true_img = np.clip(true_img * 255.0, 0, 255)
            if pred_img.max() <= 1.0:
                pred_img = np.clip(pred_img * 255.0, 0, 255)
            
            true_img = np.clip(true_img, 0, 255).astype(np.uint8)
            pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)

            # Convert NumPy arrays to PIL Images
            true_img_pil = Image.fromarray(true_img)
            pred_img_pil = Image.fromarray(pred_img)

            # Combine the images horizontally
            combined_img = Image.new('RGB', (true_img_pil.width + pred_img_pil.width, true_img_pil.height))
            combined_img.paste(true_img_pil, (0, 0))
            combined_img.paste(pred_img_pil, (true_img_pil.width, 0))

            # Append the combined image
            combined_images.append(combined_img)

            # Extract symbolic sequence up to the eos_token_id
            symbol_sequence_list = symbolic_sequence[i].detach().cpu().tolist()
            trimmed_sequence = []
            for token in symbol_sequence_list:
                if token == eos_token_id:
                    break
                trimmed_sequence.append(token)

            # Convert sequence to a string or any readable format for logging
            logged_sequences.append(f"Sample {i}: {' '.join(map(str, trimmed_sequence))}")

        return combined_images, logged_sequences

    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        """
        # Define different learning rates
        lr_main = self.hparams.optimizer.lr  # Main learning rate
        lr_vqvae = lr_main / 10  # VQ-VAE learning rate (10 times less)

        # Separate the parameters for the two models
        vqvae_params = list(self.vqvae.parameters())  # Parameters for self.vqvae
        other_params = [p for n, p in self.named_parameters() if "vqvae" not in n]  # All other parameters

        # Instantiate the optimizer with parameter groups
        # Manually instantiate the optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {"params": vqvae_params, "lr": lr_vqvae},  # Set lower lr for vqvae
            {"params": other_params, "lr": lr_main},   # Set main lr for other params
            ])
        # Check if a scheduler is provided
        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler.pop('scheduler_config', None)
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **scheduler_config,
                },
            }

        return {"optimizer": optimizer}

    

    def _initialize_autoreg_wrapped_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.sequence_model_xz = hydra.utils.instantiate(models_config.sequence_model_xz.model)
        self.sequence_model_zx = hydra.utils.instantiate(models_config.sequence_model_zx.model)
        self.sequence_model_xz_unwrapped = hydra.utils.instantiate(models_config.sequence_model_xz.model_unwrapper, self.sequence_model_xz)
        self.sequence_model_zx_unwrapped = hydra.utils.instantiate(models_config.sequence_model_zx.model_unwrapper, self.sequence_model_zx)
        
        # making it a dictionary from an OmegaConf object
        discretizer_z_config = OmegaConf.to_container(models_config.discretizer_z.config, resolve=True)
        discretizer_x_config = OmegaConf.to_container(models_config.discretizer_x.config, resolve=True)
        # if discretizer_z_config.get('dimensions', None) is None:
        models_config.discretizer_z.pop('config')
        models_config.discretizer_x.pop('config')
        self.discretizer_z = hydra.utils.instantiate(models_config.discretizer_z, configs=discretizer_z_config)
        self.discretizer_x = hydra.utils.instantiate(models_config.discretizer_x, configs=discretizer_x_config)
        # make the embeddings of the sequence models the same as the embeddings of the discretizers
        if models_config.get('inherit_model_embedding_weights_for_discretizers', False):
            # Encoder Embedding
            self._set_discretizer_weights(self.discretizer_z.encoder_embedding, self.sequence_model_zx_unwrapped['encoder_embedding'])
            self._set_discretizer_weights(self.discretizer_x.encoder_embedding, self.sequence_model_xz_unwrapped['encoder_embedding'])
            # Decoder Embedding
            self._set_discretizer_weights(self.discretizer_z.decoder_embedding, self.sequence_model_xz_unwrapped['decoder_embedding'])
            self._set_discretizer_weights(self.discretizer_x.decoder_embedding, self.sequence_model_zx_unwrapped['decoder_embedding'])
            # Linear Head (for the linear layers)
            self._set_discretizer_weights(self.discretizer_z.linear_head, self.sequence_model_xz_unwrapped['linear_head'])
            self._set_discretizer_weights(self.discretizer_x.linear_head, self.sequence_model_zx_unwrapped['linear_head'])

        models_config.sequence_model_zx.config.control_token_ids= {'output_eos_token_id': 2, 'output_bos_token_id': 3, 
            'output_pad_token_id': 0, 'output_unknown_token_id': 1, 'input_pad_token_id':0,}
        if models_config.sequence_model_zx.config.get('output_prepending_ids', None) is None:
            models_config.sequence_model_zx.config['output_prepending_ids'] = [3]

        models_config.sequence_model_xz.config.control_token_ids= {'output_eos_token_id': 2, 'output_bos_token_id': 3, 
            'output_pad_token_id': 0, 'output_unknown_token_id': 1, 'input_pad_token_id':0,}
        if models_config.sequence_model_xz.config.get('output_prepending_ids', None) is None:
            models_config.sequence_model_xz.config['output_prepending_ids'] = [3]

        
        autoreg_sequence_model_xz = {'_target_': models_config.sequence_model_xz._target_, 'config': models_config.sequence_model_xz.config}
        autoreg_sequence_model_zx = {'_target_': models_config.sequence_model_zx._target_, 'config': models_config.sequence_model_zx.config}
        self.auto_reg_wrapped_model_xz = hydra.utils.instantiate(autoreg_sequence_model_xz, 
                vector_model=self.sequence_model_xz, input_discretizer=self.discretizer_x, output_discretizer=self.discretizer_z,)
        self.auto_reg_wrapped_model_zx = hydra.utils.instantiate(autoreg_sequence_model_zx, 
                vector_model=self.sequence_model_zx, input_discretizer=self.discretizer_z, output_discretizer=self.discretizer_x,)


    def _initialize_symbolic_autoencoder_wrappers(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.symbolic_autoencoder_wrapper_xzx = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_xzx, 
                self.auto_reg_wrapped_model_xz, self.auto_reg_wrapped_model_zx)
        self.symbolic_autoencoder_wrapper_xzx.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_xzx.config['transform_xy_outputs_to_y_inputs']
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, 
                self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
        self.symbolic_autoencoder_wrapper_zxz.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_zxz.config['transform_xy_outputs_to_y_inputs']