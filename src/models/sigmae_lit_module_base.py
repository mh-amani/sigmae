from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper
from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from typing import Any, Dict, Tuple

import hydra
from omegaconf import OmegaConf

import torch
import numpy as np
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class SigmaeLitModuleBase(LightningModule):
    """Example of a `LightningModule`

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        models_config: Dict[str, torch.nn.Module],
        model_params: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a `SigmaeLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param collator: The collator to use for training.
        :param tokenizer: The tokenizer to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self._initialize_models(models_config)

        self._initialize_metrics()

        self._initialize_hparams()
    
    def _initialize_hparams(self) -> None:
        self.usezxz_with_supervised_training = self.hparams.get('usezxz_with_supervised_training', False)
        self.usexzx_with_supervised_training = self.hparams.get('usexzx_with_supervised_training', False)

    def _initialize_metrics(self) -> None:
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        num_classes_x = self.tokenizer_x.vocab_size
        num_classes_z = self.tokenizer_z.vocab_size
        self.accuracies, self.losses = {}, {}
        for split in ['train', 'val', 'test']:
            self.accuracies[split] = {}
            self.losses[split] = {}
            for space in ['xz', 'zx', 'xzx', 'zxz']:
                self.accuracies[split][space] = {}
                self.losses[split][space] = {}
                for medium in ['token', 'sequence']:
                    # metric objects for calculating and averaging accuracy across batches
                    self.accuracies[split][space][medium] = Accuracy(task="multiclass", num_classes=num_classes_x if (space == 'zx' or space == 'xzx') else num_classes_z)
                    # for averaging loss across batches
                    self.losses[split][space][medium] = MeanMetric()
                
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.tokenizer_x = hydra.utils.instantiate(models_config.sequence_model_xz.tokenizer, _recursive_=False)
        self.tokenizer_z = hydra.utils.instantiate(models_config.sequence_model_zx.tokenizer, _recursive_=False)

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
        self.auto_reg_wrapped_model_xz = hydra.utils.instantiate(autoreg_sequence_model_xz, vector_model=self.sequence_model_xz, input_discretizer=self.discretizer_x, output_discretizer=self.discretizer_z,)
        self.auto_reg_wrapped_model_zx = hydra.utils.instantiate(autoreg_sequence_model_xz, vector_model=self.sequence_model_zx, input_discretizer=self.discretizer_z, output_discretizer=self.discretizer_x,)

        self.symbolic_autoencoder_wrapper_xzx = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_xzx, self.auto_reg_wrapped_model_xz, self.auto_reg_wrapped_model_zx)
        self.symbolic_autoencoder_wrapper_xzx.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_xzx.config['transform_xy_outputs_to_y_inputs']
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
        self.symbolic_autoencoder_wrapper_zxz.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_zxz.config['transform_xy_outputs_to_y_inputs']

    def forward(self, x_ids, x_mask, z_ids, z_mask, data_type, stage='train') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        outputs = {}
        labels = {}
        if (data_type[0] and data_type[1]) or stage!='fit':
            xz_outputs = self.auto_reg_wrapped_model_xz(input_ids=x_ids, output_ids=z_ids,
                teacher_force_output=True, max_output_length=z_mask.shape[1])
            outputs['xz'] = xz_outputs
            labels['xz'] = z_ids
            
            zx_outputs = self.auto_reg_wrapped_model_zx(input_ids=z_ids, output_ids=x_ids,
                teacher_force_output=True, max_output_length=x_mask.shape[1])
            outputs['zx'] = zx_outputs
            labels['zx'] = x_ids

        if (data_type[0] and not data_type[1]) or (stage!='fit') or (data_type[0] and data_type[1] and self.usexzx_with_supervised_training):
            xzx_outputs = self.symbolic_autoencoder_wrapper_xzx(x_ids=x_ids, z_ids=x_ids)
            outputs['xzx'] = xzx_outputs
            outputs['xzx']['logit'] = outputs['xzx']['logit_z']
            labels['xzx'] = x_ids

        if (data_type[1] and not data_type[0]) or (stage!='fit') or (data_type[0] and data_type[1] and self.usezxz_with_supervised_training):
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

        # update and log metrics
        self.log(f"{stage}/x_data_available", float(data_type[0]), sync_dist=True)
        self.log(f"{stage}/z_data_available", float(data_type[1]),  sync_dist=True)
        self.log(f"{stage}/global_step", float(self.global_step), sync_dist=True)

        # forward pass
        outputs, labels = self.forward(x_ids, x_mask, z_ids, z_mask, data_type, stage=stage)
        # compute losses, predictions and update metrics
        losses = {}
        preds = {}
        for space in outputs.keys():
            loss = self.criterion(outputs[space]['logit'][..., :-1, :].permute(0, 2, 1), labels[space][..., 1:])
            preds[space] = torch.argmax(outputs[space]['logit'], dim=-1)
            losses[space] = loss

            self.losses[stage][space]['token'].update(losses[space])
            loss += losses[space]
            self.accuracies[stage][space]['token'](preds[space][..., :-1], labels[space][..., 1:])
            
            # Use the metric_attribute parameter to specify the attribute name
            self.log(f"{stage}/{space}/loss", self.losses[stage][space]['token'], 
                     on_step=False, on_epoch=True, prog_bar=True, 
                     metric_attribute=f"losses_{stage}_{space}_token")
            
            self.log(f"{stage}/{space}/acc", self.accuracies[stage][space]['token'], 
                     on_step=False, on_epoch=True, prog_bar=True, 
                     metric_attribute=f"accuracies_{stage}_{space}_token")

        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for split in self.accuracies.keys():
            for space in self.accuracies[split].keys():
                for medium in self.accuracies[split][space].keys():
                    self.accuracies[split][space][medium].reset()
                    self.losses[split][space][medium].reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch, stage='train')
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, stage='val')

        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        acc = self.accuracies['val']['xz']['token'].compute()
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, stage='test')

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams['model_params'].get('compile', False) and stage == "fit":
            for model in [self.sequence_model_xz, self.sequence_model_zx]:
                model = torch.compile(model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SigmaAELitModule(None, None, None, None)
