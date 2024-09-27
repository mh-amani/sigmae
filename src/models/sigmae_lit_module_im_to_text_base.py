import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.models.components.vit_unprocessor import UnViTImageProcessor
from .sigmae_lit_module_base import SigmaeLitModuleBase
from PIL import Image
import numpy as np

class SigmaeLitModuleImageToTextBase(SigmaeLitModuleBase):
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
                self.losses[split].update({training_mode:  MeanMetric()})
                # if training_mode.endswith('x'): # text space
                #     self.accuracies[split].update({training_mode: MaxMetric(top_k=1)})
                # else:
                #     self.accuracies[split].update({training_mode: StructuralSimilarityIndexMeasure()})
                

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.processor_z = hydra.utils.instantiate(models_config.sequence_model_zx.processor, _recursive_=False)
        self.unprocessor_z = UnViTImageProcessor(self.processor_z,original_height=self.hparams['model_params']['image_size'],original_width=self.hparams['model_params']['image_size'],)
        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)


    def forward(self, x, z, data_type=None, stage='learn', training_mode=None) -> torch.Tensor:
        NotImplementedError("The forward method must be implemented in the derived class.")
    
    def model_step(self, batch, batch_idx=-1, stage='learn', log_things=True):
        x, z, data_type = batch['x'], batch['z'], batch['data_type']
        z = z[..., 0:self.hparams['model_params']['num_channels']]
        data_type = torch.all(data_type, dim=0)

        total_loss = 0

        if stage == 'val':
            # Loop over each training mode during validation
            for training_mode in self.training_modes:
                outputs, labels = self.forward(x, z, data_type, stage=stage, training_mode=training_mode)

                # Compute loss
                loss = self.compute_loss(outputs, labels, training_mode, stage)
                total_loss += loss

                # Log metrics
                log_kwargs = self.logging_kwargs[stage]
                self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)

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
    
    def compute_loss(self, outputs, labels, training_mode, stage):
        if training_mode.endswith('x'):  # Text space
            non_pad_mask = labels[training_mode]['logit'][..., 1:] != self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
            labels_for_loss = labels[training_mode]['logit'][..., 1:].clone()
            labels_for_loss[~non_pad_mask] = -100
            loss = self.discrete_criterion(
                outputs[training_mode]['logit'][..., :-1, :].permute(0, 2, 1),
                labels_for_loss
            )
        else:  # Continuous space
            loss = self.continuous_criterion(
                outputs[training_mode]['logit'],
                labels[training_mode]['logit']
            )

        self.losses[stage][training_mode](loss)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx, stage='val')
        self.log("val/loss", loss, **self.logging_kwargs['val'])
        return loss

        
    def _create_random_batch_of_tokens(self, batchsize, max_num_tokens, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        random_batch_of_tokens = torch.randint(4, self.discretizer_x.vocab_size, (batchsize, max_num_tokens)).to(self.device)
        random_batch_of_tokens[:, 0] = 3
        ending_index = torch.randint(1, max_num_tokens, (batchsize,))
        for i in range(batchsize):
            random_batch_of_tokens[i, ending_index[i]] = 2
            random_batch_of_tokens[i, ending_index[i]+1:] = 0
        return random_batch_of_tokens

    def _prepare_to_show_output_samples(self, z_pred, z_true, symbolic_sequence, num_images=10) -> None:
        # log 10 images every 2 epochs
        eos_token_id = self.auto_reg_wrapped_model_zx.control_token_ids['output_eos_token_id']
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
            

    def _log_output_samples(self, z_pred, z_true, symbolic_sequence, stage, training_mode, num_images=10) -> None:
        combined_images, logged_sequences = self._prepare_to_show_output_samples(
            z_pred, z_true, symbolic_sequence, num_images
        )
        name = f"{stage}/{training_mode}/output_samples"
        self.logger.log_image(key=name, images=combined_images, caption=logged_sequences)



    def _initialize_autoreg_wrapped_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.sequence_model_xz = hydra.utils.instantiate(models_config.sequence_model_xz.model)
        self.sequence_model_zx = hydra.utils.instantiate(models_config.sequence_model_zx.model)
        self.sequence_model_xz.train()
        self.sequence_model_zx.train()

        # making it a dictionary from an OmegaConf object
        discretizer_z_config = OmegaConf.to_container(models_config.discretizer_z.config, resolve=True)
        discretizer_x_config = OmegaConf.to_container(models_config.discretizer_x.config, resolve=True)
        models_config.discretizer_z.pop('config')
        models_config.discretizer_x.pop('config')
        self.discretizer_z = hydra.utils.instantiate(models_config.discretizer_z, configs=discretizer_z_config)
        self.discretizer_x = hydra.utils.instantiate(models_config.discretizer_x, configs=discretizer_x_config)
        if models_config.sequence_model_zx.config.get('inherit_model_embedding_weights_for_discretizers', False):
            self.sequence_model_zx_unwrapped = hydra.utils.instantiate(models_config.sequence_model_zx.model_unwrapper, self.sequence_model_zx)
            self._set_discretizer_weights(self.discretizer_z.encoder_embedding, self.sequence_model_zx_unwrapped['encoder_embedding'])
            self._set_discretizer_weights(self.discretizer_x.decoder_embedding, self.sequence_model_zx_unwrapped['decoder_embedding'])
            self._set_discretizer_weights(self.discretizer_x.linear_head, self.sequence_model_zx_unwrapped['linear_head'])
        if models_config.sequence_model_xz.config.get('inherit_model_embedding_weights_for_discretizers', False):
            self.sequence_model_xz_unwrapped = hydra.utils.instantiate(models_config.sequence_model_xz.model_unwrapper, self.sequence_model_xz)
            self._set_discretizer_weights(self.discretizer_x.encoder_embedding, self.sequence_model_xz_unwrapped['encoder_embedding'])
            self._set_discretizer_weights(self.discretizer_z.decoder_embedding, self.sequence_model_xz_unwrapped['decoder_embedding'])
            self._set_discretizer_weights(self.discretizer_z.linear_head, self.sequence_model_xz_unwrapped['linear_head'])
            # desc_z_dec_shape = self.discretizer_z.decoder_embedding.weight.data.shape
            # self.discretizer_z.decoder_embedding.weight.data = self.sequence_model_xz_unwrapped['decoder_embedding'].weight.clone()[:self.discretizer_z.decoder_embedding_dim].T[:desc_z_dec_shape[0], :desc_z_dec_shape[1]] 
            # self._set_discretizer_weights(self.discretizer_z.linear_head, self.sequence_model_xz_unwrapped['linear_head'])
            

        # config for the autoregressive wrapper
        models_config.sequence_model_xz.config.control_token_ids= {'input_pad_token_id': 0}
        
        models_config.sequence_model_zx.config.control_token_ids= {'output_eos_token_id': 2, 'output_bos_token_id': 3, 
            'output_pad_token_id': 0, 'output_unknown_token_id': 1, 'input_pad_token_id':None,}
        if models_config.sequence_model_zx.config.get('output_prepending_ids', None) is None:
            models_config.sequence_model_zx.config['output_prepending_ids'] = [3]


        output_prepending_embeds_dec = torch.randn(1, self.discretizer_z.decoder_embedding_dim) / torch.sqrt(torch.tensor([self.discretizer_z.decoder_embedding_dim]))

        models_config_sequence_model_xz = OmegaConf.to_container(models_config.sequence_model_xz.config, resolve=True)
        models_config_sequence_model_xz['output_prepending_embeds_dec'] = output_prepending_embeds_dec
        
        autoreg_sequence_model_xz = {'_target_': models_config.sequence_model_xz._target_, 'config': models_config_sequence_model_xz}
        autoreg_sequence_model_zx = {'_target_': models_config.sequence_model_zx._target_, 'config': models_config.sequence_model_zx.config}
        self.auto_reg_wrapped_model_xz = hydra.utils.instantiate(autoreg_sequence_model_xz, 
                vector_model=self.sequence_model_xz, input_discretizer=self.discretizer_x, output_discretizer=self.discretizer_z,)
        self.auto_reg_wrapped_model_zx = hydra.utils.instantiate(autoreg_sequence_model_zx, 
                vector_model=self.sequence_model_zx, input_discretizer=self.discretizer_z, output_discretizer=self.discretizer_x,)
        
        

    def _initialize_symbolic_autoencoder_wrappers(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, 
                self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
        self.symbolic_autoencoder_wrapper_zxz.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_zxz.config['transform_xy_outputs_to_y_inputs']

        self.symbolic_autoencoder_wrapper_xzx = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_xzx,
                self.auto_reg_wrapped_model_xz, self.auto_reg_wrapped_model_zx)
        self.symbolic_autoencoder_wrapper_xzx.processor_z = self.processor_z
    
        # Assign the closure
        self.symbolic_autoencoder_wrapper_xzx.transform_xy_outputs_to_y_inputs = self.make_transform_image_patch_to_input_image()


    def make_transform_image_patch_to_input_image(self):
        def transform_image_patch_to_input_image(outputs):
            image_patches = outputs['id']
            processed_image = self.fold(image_patches.permute(0, 2, 1))
            return {'vector_encoder': processed_image, 'output_attention_mask': None}
        return transform_image_patch_to_input_image

    



    # Helper functions


    # def model_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], stage, log_things=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Perform a single model step on a batch of data.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

    #     :return: A tuple containing (in order):
    #         - A tensor of losses.
    #         - A tensor of predictions.
    #         - A tensor of target labels.
    #     """
    #     x, z, data_type = batch['x'], batch['z'], batch['data_type']
    #     z = z[..., 0:self.hparams['model_params']['num_channels']]
    #     data_type = torch.all(data_type, dim=0)
        
    #     outputs, labels = self.forward(x, z, data_type, stage=stage)
        
    #     # compute losses, predictions and update metrics
    #     # losses = {}
    #     # preds = {}
    #     loss = 0
    #     training_mode = list(outputs.keys())[0]
    #     # Create a mask for non-pad tokens
    #     if training_mode.endswith('x'): # text space
    #         non_pad_mask = labels[training_mode]['logit'][..., 1:] != self.auto_reg_wrapped_model_zx.control_token_ids['output_pad_token_id']
    #         # Temporarily replace pad tokens with -100
    #         labels_for_loss = labels[training_mode]['logit'][..., 1:].clone()
    #         labels_for_loss[~non_pad_mask] = -100
    #         loss = self.discrete_criterion(outputs[training_mode]['logit'][..., :-1, :].permute(0, 2, 1), labels_for_loss)
            
    #         self.losses[stage][training_mode](loss)
    #         # preds[training_mode] = torch.argmax(outputs[training_mode]['logit'], dim=-1)
    #         # Use the non-pad mask for accuracy calculation
    #         # self.accuracies[stage][space](preds[space][..., :-1][non_pad_mask], labels[space][..., 1:][non_pad_mask])

    #     else: 
    #         loss = self.continuous_criterion(outputs[training_mode]['logit'], labels[training_mode]['logit'])
    #         self.losses[stage][training_mode](loss)
    #         # self.accuracies[stage][space]['continous output'](outputs['zxz']['image'], labels['zxz']['image'])
    
    #     # Log metrics
    #     # Use the logging kwargs based on the current stage
    #     log_kwargs = self.logging_kwargs[stage]
    #     self.log(f"{stage}/{training_mode}/loss", self.losses[stage][training_mode], **log_kwargs)        
    #     # self.log(f"{stage}/{space}/acc", self.accuracies[stage][space]['token'], **log_kwargs)             

    #     # if last batch of the epoch, log the output samples
    #     if log_things and self.global_step % self.hparams['models_config']['logging_freq'] == 0:
    #         self._log_output_samples(outputs[training_mode]['image'], labels[training_mode]['image'], outputs[training_mode]['id_z'], stage, training_mode, self.hparams['models_config']['logging_freq'], num_images=10)

    #     return loss
    
    # # the train loop to debug the grads
    # def training_step(self, batch, batch_idx):
            
    #     loss = self.model_step(batch, stage='learn')
    #     opt = self.optimizers()
    #     # scale losses by 1/N (for N batches of gradient accumulation)
    #     self.manual_backward(loss)

    #     # accumulate gradients of N batches
    #     opt.step()
    #     opt.zero_grad()
