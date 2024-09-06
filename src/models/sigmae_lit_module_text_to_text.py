import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.sigmae_lit_module_base import SigmaeLitModuleBase


class SigmaeLitModuleTextToText(SigmaeLitModuleBase):
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
        self.usezxz_with_supervised_training = self.hparams.get('usezxz_with_supervised_training', False)
        self.usexzx_with_supervised_training = self.hparams.get('usexzx_with_supervised_training', False)

    def _initialize_metrics(self) -> None:
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        num_classes_x = self.tokenizer_x.vocab_size
        num_classes_z = self.tokenizer_z.vocab_size
        self.accuracies, self.losses = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        for split in ['learn', 'val', 'test']: # you can't use 'train' as a key ... it's a reserved word
            self.accuracies.update({split: torch.nn.ModuleDict()})
            self.losses.update({split: torch.nn.ModuleDict()})
            for space in ['xz', 'zx', 'xzx', 'zxz']:
                self.accuracies[split].update({space: torch.nn.ModuleDict()})
                self.losses[split].update({space: torch.nn.ModuleDict()})
                for medium in ['token', 'sequence']:
                    # metric objects for calculating and averaging accuracy across batches
                    self.accuracies[split][space].update({medium: Accuracy(task="multiclass", num_classes=num_classes_x if (space == 'zx' or space == 'xzx') else num_classes_z)})
                    # for averaging loss across batches
                    self.losses[split][space].update({medium: MeanMetric()})

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.tokenizer_x = hydra.utils.instantiate(models_config.sequence_model_xz.tokenizer, _recursive_=False)
        self.tokenizer_z = hydra.utils.instantiate(models_config.sequence_model_zx.tokenizer, _recursive_=False)

        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)


    def forward(self, x_ids, x_mask, z_ids, z_mask, data_type, stage='learn') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        outputs = {}
        labels = {}
        if (data_type[0] and data_type[1]) or stage!='learn':
            xz_outputs = self.auto_reg_wrapped_model_xz(input_ids=x_ids, output_ids=z_ids, teacher_force_output=True)
            outputs['xz'] = xz_outputs
            labels['xz'] = z_ids
            
            zx_outputs = self.auto_reg_wrapped_model_zx(input_ids=z_ids, output_ids=x_ids, teacher_force_output=True)
            outputs['zx'] = zx_outputs
            labels['zx'] = x_ids

        if (data_type[0] and not data_type[1]) or (stage!='learn') or (data_type[0] and data_type[1] and self.usexzx_with_supervised_training):
            xzx_outputs = self.symbolic_autoencoder_wrapper_xzx(x_ids=x_ids, z_ids=x_ids)
            outputs['xzx'] = xzx_outputs
            outputs['xzx']['logit'] = outputs['xzx']['logit_z']
            labels['xzx'] = x_ids

        if (data_type[1] and not data_type[0]) or (stage!='learn') or (data_type[0] and data_type[1] and self.usezxz_with_supervised_training):
            zxz_outputs = self.symbolic_autoencoder_wrapper_zxz(x_ids=z_ids, z_ids=z_ids)
            outputs['zxz'] = zxz_outputs
            outputs['zxz']['logit'] = outputs['zxz']['logit_z']
            labels['zxz'] = z_ids
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
        x_ids, x_mask, z_ids, z_mask, data_type = batch['x_ids'], batch['x_mask'], batch['z_ids'], batch['z_mask'], batch['data_type']
        data_type = torch.all(data_type, dim=0)

        # # update and log metrics
        # self.log(f"{stage}/x_data_available", float(data_type[0]), sync_dist=True)
        # self.log(f"{stage}/z_data_available", float(data_type[1]),  sync_dist=True)
        # self.log(f"{stage}/global_step", float(self.global_step), sync_dist=True)

        # forward pass
        outputs, labels = self.forward(x_ids, x_mask, z_ids, z_mask, data_type, stage=stage)
        # compute losses, predictions and update metrics
        losses = {}
        preds = {}
        loss = 0
        for space in outputs.keys():
            for space in outputs.keys():
                # Create a mask for non-pad tokens
                non_pad_mask = labels[space][..., 1:] != self.tokenizer_x.pad_token_id if space in ['zx', 'xzx'] else \
                        labels[space][..., 1:] != self.tokenizer_z.pad_token_id
                
                # Temporarily replace pad tokens with -100
                labels_for_loss = labels[space][..., 1:].clone()
                labels_for_loss[~non_pad_mask] = -100

                losses[space]= self.criterion(outputs[space]['logit'][..., :-1, :].permute(0, 2, 1), labels_for_loss)
                self.losses[stage][space]['token'].update(losses[space])
                loss += losses[space]

                preds[space] = torch.argmax(outputs[space]['logit'], dim=-1)
                # Use the non-pad mask for accuracy calculation
                self.accuracies[stage][space]['token'](preds[space][..., :-1][non_pad_mask], labels[space][..., 1:][non_pad_mask])

                # Log metrics
                # Use the logging kwargs based on the current stage
                log_kwargs = self.logging_kwargs[stage]
                self.log(f"{stage}/{space}/loss", self.losses[stage][space]['token'], 
                        metric_attribute=f"losses_{stage}_{space}_token", **log_kwargs)
                
                self.log(f"{stage}/{space}/acc", self.accuracies[stage][space]['token'], 
                        metric_attribute=f"accuracies_{stage}_{space}_token", **log_kwargs)

        # loss = losses['xz']
        return loss
        

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

        models_config.sequence_model_xz.config.control_token_ids= {'input_pad_token_id': self.tokenizer_x.pad_token_id,
            'output_eos_token_id': self.tokenizer_x.eos_token_id,   
            'output_pad_token_id': self.tokenizer_x.pad_token_id,
            'output_unknown_token_id': self.tokenizer_x.unk_token_id}
        
        models_config.sequence_model_zx.config.control_token_ids= {'input_pad_token_id': self.tokenizer_z.pad_token_id,
            'output_eos_token_id': self.tokenizer_z.eos_token_id,   
            'output_pad_token_id': self.tokenizer_z.pad_token_id,
            'output_unknown_token_id': self.tokenizer_z.unk_token_id}
        
        if models_config.sequence_model_xz.config.get('output_prepending_ids', None) is None:
            models_config.sequence_model_xz.config['output_prepending_ids'] = [self.tokenizer_x.bos_token_id]
            # warn the user that the output_prepending_ids is set to the bos_token_id
            print("Warning: output_prepending_ids is set to the bos_token_id")
        if models_config.sequence_model_zx.config.get('output_prepending_ids', None) is None:
            models_config.sequence_model_zx.config['output_prepending_ids'] = [self.tokenizer_z.bos_token_id]
        
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