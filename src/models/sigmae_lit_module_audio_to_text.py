import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from .sigmae_lit_module_base import SigmaeLitModuleBase


class SigmaeLitModuleAudioToText(SigmaeLitModuleBase):
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
  
    def _initialize_hparams(self) -> None:
        pass

    def _initialize_metrics(self) -> None:
        # loss function is L1Loss
        self.criterion = torch.nn.L1Loss(reduction='mean')

        self.accuracies, self.losses = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        for split in ['learn', 'val', 'test']: # you can't use 'train' as a key ... it's a reserved word
            self.accuracies.update({split: torch.nn.ModuleDict()})
            self.losses.update({split: torch.nn.ModuleDict()})
            for space in ['zxz']:
                self.accuracies[split].update({space: torch.nn.ModuleDict()})
                self.losses[split].update({space: torch.nn.ModuleDict()})
                for medium in ['continous output']:
                    # metric objects for calculating and averaging accuracy across batches
                    self.accuracies[split][space].update({medium: Accuracy(task="multiclass", num_classes=num_classes_x if (space == 'zx' or space == 'xzx') else num_classes_z)})
                    # for averaging loss across batches
                    self.losses[split][space].update({medium: MeanMetric()})

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.processor_z = hydra.utils.instantiate(models_config.sequence_model_zx.processor, _recursive_=False)

        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)
        self.accuracies = torch.nn.ModuleDict()
        self.losses = torch.nn.ModuleDict()

    def forward(self, x, z, data_type, stage='learn') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        outputs = {}
        labels = {}
        uni_channel_z = z[:, 0:1, ...]
        z_patches = self.unfold(uni_channel_z).permute(0, 2, 1)
        z_patches_embeds = self.discretizer_z.decoder_embedding(z_patches)
        zxz_outputs = self.symbolic_autoencoder_wrapper_zxz(x_embeds_enc=z,  z_embeds_dec=z_patches_embeds, 
                                                            z_attention_mask=torch.ones_like(z_patches, dtype=torch.bool),
                                                            teacher_force_z=True)
        outputs['zxz'] = zxz_outputs
        outputs['zxz']['logit'] = zxz_outputs['id_z']
        labels['zxz'] = z_patches

        return outputs, labels

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # stage = self.trainer.state.stage._value_ # stages are 'fit', 'validate', 'test', 'predict', 'sanity_check'
        x, z, data_type = batch['x'], batch['z'], batch['data_type']
        data_type = torch.all(data_type, dim=0)
        unprocessed_z = batch['z_unrpocessed'].permute(0, 3, 1, 2)[:, 0:1, ...]

        outputs, labels = self.forward(x, z, data_type, stage=stage)

        output_image = self.fold(outputs['zxz']['logit'][:, :-1, ...].permute(0, 2, 1))

        # compute losses, predictions and update metrics

        # loss = self.criterion(outputs['zxz']['logit'], labels['zxz'])
        loss= self.criterion(output_image, unprocessed_z/255)
        self.loss.update(loss)

        # log 10 images every epoch
        if self.current_epoch % 2 == 0:
            self.logger.log_image(key='input_image', images=[unprocessed_z[i] for i in range(10)])
            self.logger.log_image(key='output_image', images=[output_image[i] for i in range(10)])
        return loss
    
    # def training_step(self, batch, batch_idx):
        
    #     x, z, data_type = batch['x'], batch['z'], batch['data_type']
    #     data_type = torch.all(data_type, dim=0)
    #     unprocessed_z = batch['z_unrpocessed'].permute(0, 3, 1, 2)[:, 0:1, ...]
    #     # forward pass
    #     # torch.autograd.set_detect_anomaly(True)
    #     outputs, labels = self.forward(x, z, data_type, stage='train')
    #     labels = torch.zeros_like(outputs['vector_encoder'])
    #     loss = torch.nn.functional.mse_loss(outputs['vector_encoder'], labels)
        
    #     # loss = self.model_step(batch, stage='learn')
    #     opt = self.optimizers()
    #     # scale losses by 1/N (for N batches of gradient accumulation)
    #     self.manual_backward(loss)

    #     # accumulate gradients of N batches
    #     opt.step()
    #     opt.zero_grad()

        

    def _initialize_autoreg_wrapped_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.sequence_model_xz = hydra.utils.instantiate(models_config.sequence_model_xz.model)
        self.sequence_model_zx = hydra.utils.instantiate(models_config.sequence_model_zx.model)
        self.sequence_model_xz_unwrapped = hydra.utils.instantiate(models_config.sequence_model_xz.model_unwrapper, self.sequence_model_xz)
        self.sequence_model_zx_unwrapped = hydra.utils.instantiate(models_config.sequence_model_zx.model_unwrapper, self.sequence_model_zx)
        
        # making it a dictionary from an OmegaConf object
        discretizer_z_config = OmegaConf.to_container(models_config.discretizer_z.config, resolve=True)
        if discretizer_z_config.get('dimensions', None) is None:
            discretizer_z_config['encoder_embedding'] = self.sequence_model_zx_unwrapped['encoder_embedding']
            discretizer_z_config['decoder_embedding'] = self.sequence_model_xz_unwrapped['decoder_embedding']
            discretizer_z_config['linear_head'] = self.sequence_model_xz_unwrapped['linear_head']
        # making it a dictionary from an OmegaConf object
        discretizer_x_config = OmegaConf.to_container(models_config.discretizer_x.config, resolve=True)
        if discretizer_x_config.get('dimensions', None) is None:
            discretizer_x_config['encoder_embedding'] = self.sequence_model_xz_unwrapped['encoder_embedding']
            discretizer_x_config['decoder_embedding'] = self.sequence_model_zx_unwrapped['decoder_embedding']
            discretizer_x_config['linear_head'] = self.sequence_model_zx_unwrapped['linear_head']
        models_config.discretizer_z.pop('config')
        models_config.discretizer_x.pop('config')
        self.discretizer_z = hydra.utils.instantiate(models_config.discretizer_z, configs=discretizer_z_config)
        self.discretizer_x = hydra.utils.instantiate(models_config.discretizer_x, configs=discretizer_x_config)

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
        self.unfold = torch.nn.Unfold(kernel_size=(4,4), stride=(4,4))
        self.fold = torch.nn.Fold(output_size=(28, 28), kernel_size=(4,4), stride=(4,4))
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, 
                self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
        self.symbolic_autoencoder_wrapper_zxz.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_zxz.config['transform_xy_outputs_to_y_inputs']