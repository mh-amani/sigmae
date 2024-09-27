from typing import Any, Dict, Tuple, List
import hydra
import torch
from lightning import LightningModule
from omegaconf import OmegaConf

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

        self.logging_kwargs = {
            'learn': {'on_step': True, 'on_epoch': False, 'prog_bar': True, 'sync_dist': True},
            'val': {'on_step': False, 'on_epoch': True, 'prog_bar': True, 'sync_dist': True},
            'test': {'on_step': False, 'on_epoch': True, 'prog_bar': True, 'sync_dist': True}
        }    
    
    def _initialize_hparams(self) -> None:
        NotImplementedError

    def _initialize_metrics(self) -> None:
        NotImplementedError

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        NotImplementedError

    def forward(self, x_ids, x_mask, z_ids, z_mask, data_type, stage='learn') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        NotImplementedError

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        NotImplementedError

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # for split in self.accuracies.keys():
        #     for space in self.accuracies[split].keys():
        #         for medium in self.accuracies[split][space].keys():
        #             self.accuracies[split][space][medium].reset()
        #             self.losses[split][space][medium].reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch, stage='learn')
        self.log("learn/loss", loss, **self.logging_kwargs['learn'])
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, stage='val')
        self.log("val/loss", loss, **self.logging_kwargs['val'])
    
    def on_validation_epoch_end(self) -> None:
        # log learning rate
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch, stage='test')
        self.log("test/loss", loss, **self.logging_kwargs['test'])

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
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            # Create a copy of self.hparams.scheduler without modifying the original
            scheduler_config = self.hparams.scheduler.get('scheduler_config', {})

            # Copy everything except 'scheduler_config' into a new OmegaConf object
            scheduler_copy = OmegaConf.masked_copy(self.hparams.scheduler, self.hparams.scheduler.keys())
            scheduler_copy.pop('scheduler_config', None)  # Exclude scheduler_config from the new copy

            # Instantiate the scheduler using the new copy
            scheduler = hydra.utils.instantiate(scheduler_copy, optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **scheduler_config,  # Add the scheduler_config from the original config
                },
            }

        return {"optimizer": optimizer}


    # def on_fit_start(self) -> None:
    #     """Lightning hook that is called when the fit begins."""
    #     print('my name is masani and i hate my life')
    #     # self.configure_optimizers()
    #     self.configure_optimizers()
    #     self.trainer.optimizers
    #     return super().on_fit_start()
    
    def on_load_checkpoint(self, checkpoint):
        checkpoint["optimizer_states"] = []
        
    # Utility function to cleanly set the embedding or linear layer weights from the source model
    def _set_discretizer_weights(self, target, source, clone=True):
        """
        Sets the weights (and bias if applicable) of the target layer from the source layer.
        Supports both Embedding and Linear layers.
        
        Args:
            target (nn.Module): The target layer (embedding or linear) whose weights are being set.
            source (nn.Module): The source layer from which to copy weights (and bias if applicable).
            clone (bool): Whether to clone the weights and bias to avoid in-place modification.
        """
        # Handle the weights for embedding or linear layers
        if (isinstance(target, torch.nn.Embedding) or isinstance(target, torch.nn.Linear)) and \
            (isinstance(source, torch.nn.Embedding) or isinstance(source, torch.nn.Linear)):
            # For Embedding or Linear layer, handle .weight
            vocab_size, embedding_dim = target.weight.shape
            if clone:
                target.weight.data = source.weight[:vocab_size, :embedding_dim].clone()
            else:
                target.weight.data = source.weight[:vocab_size, :embedding_dim]
            # If the layer is Linear and has a bias term, copy the bias as well
            if isinstance(target, torch.nn.Linear) and target.bias is not None and isinstance(source, torch.nn.Linear):
                target.bias.data = source.bias[:target.bias.shape[0]].clone() if clone else source.bias[:target.bias.shape[0]]
                
if __name__ == "__main__":
    _ = SigmaeLitModuleBase(None, None, None, None)




# to do manual gradient and peek into the grad

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


# then you can do:
# # print out all param's grads. 
# grad_dict = {}

# # Loop over each named parameter in the model
# for name, param in self.sequence_model_zx.named_parameters():
#     if param.grad is not None:  # Check if the parameter has a gradient
#         grad_dict[name] = param.grad.clone().detach()  # Clone the gradient to avoid any modifications
