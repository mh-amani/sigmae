from src.models.sigmae_lit_base_module import SigmaAELitModule

class SigmaAELitText2TextModule(SigmaAELitModule):
   

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
            for space in ['xz', 'zx', 'xzx', 'zxx']:
                self.accuracies[split][space] = {}
                self.losses[split][space] = {}
                for medium in ['token', 'sequence']:
                    # metric objects for calculating and averaging accuracy across batches
                    self.accuracies[split][space][medium] = Accuracy(task="multiclass", num_classes=num_classes_x if space == 'x' else num_classes_z)
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
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
    

    def forward(self, x_ids, x_mask, z_ids, z_mask, data_type, stage='train') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        outputs = {}
        data_type = torch.all(data_type, dim=0)
        if (data_type[0] and data_type[1]) or stage!='train':
            xz_outputs = self.auto_reg_wrapped_model_xz(input_ids=x_ids, output_ids=z_ids,
                teacher_force_output=True, max_output_length=z_mask.shape[1])
            
            zx_outputs = self.auto_reg_wrapped_model_zx(input_ids=z_ids, input_attention_mask=z_mask,
                output_ids=x_ids, output_attention_mask=x_mask,
                teacher_force_output=True, max_output_length=x_mask.shape[1])
        
        if (data_type[0] and not data_type[1]) or (stage!='train') or (data_type[0] and data_type[1] and self.usexzx_with_supervised_training):
            xzx_outputs = self.symbolic_autoencoder_wrapper_xzx(input_ids=x_ids, input_attention_mask=x_mask,)

        if (data_type[1] and not data_type[0]) or (stage!='train') or (data_type[0] and data_type[1] and self.usezxz_with_supervised_training):
            zxz_outputs = self.symbolic_autoencoder_wrapper_zxz(input_ids=z_ids, input_attention_mask=z_mask,)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        stage = self.trainer.state.stage._value_ # stages are 'fit', 'validate', 'test', 'predict', 'sanity_check'

        x_ids, x_mask, z_ids, z_mask, data_type = batch['x_ids'], batch['x_mask'], batch['z_ids'], batch['z_mask'], batch['data_type']
        
        logits = self.forward(x_ids, x_mask, z_ids, z_mask, data_type, stage=stage)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels


if __name__ == "__main__":
    _ = SigmaAELitText2TextModule(None, None, None, None)
