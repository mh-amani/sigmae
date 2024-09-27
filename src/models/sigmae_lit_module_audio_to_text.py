import torch
import hydra
from typing import Tuple, Dict
from omegaconf import OmegaConf
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from .sigmae_lit_module_base import SigmaeLitModuleBase
import copy
from transformers.models.speecht5.modeling_speecht5 import SpeechT5SpectrogramLoss
import wandb
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
        config = self.sequence_model_xz.config
        
        self.criterion_zxz = SpeechT5SpectrogramLoss(config)
        self.criterion_xz = SpeechT5SpectrogramLoss(config)
        self.criterion_zx = torch.nn.CrossEntropyLoss()

        self.accuracies, self.losses = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        
        for split in ['learn', 'val', 'test']: # you can't use 'train' as a key ... it's a reserved word
            self.accuracies.update({split: torch.nn.ModuleDict()})
            self.losses.update({split: torch.nn.ModuleDict()})
            for space in ['zxz','xz','zx']:
                # self.accuracies[split].update({space: torch.nn.ModuleDict()})
                self.losses[split].update({space: torch.nn.ModuleDict()})
                for medium in ['output', "no_grad", "tf"]:
                    # metric objects for calculating and averaging accuracy across batches
                    # self.accuracies[split][space].update({medium: Accuracy(task="multiclass", num_classes=num_classes_x if (space == 'zx' or space == 'xzx') else num_classes_z)})
                    # for averaging loss across batches
                    self.losses[split][space].update({medium: MeanMetric()})

    def _initialize_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.processor_z = hydra.utils.instantiate(models_config.sequence_model_zx.processor, _recursive_=False)

        self._initialize_autoreg_wrapped_models(models_config)
        self._initialize_symbolic_autoencoder_wrappers(models_config)
        self.accuracies = torch.nn.ModuleDict()
        self.losses = torch.nn.ModuleDict()
        
    def on_train_start(self) -> None:
        self.symbolic_autoencoder_wrapper_zxz.to(self.device)
        
    def _process_audio_outputs(
        self,
        outputs_before_postnet_spectrogram,
        outputs_after_postnet_spectrogram,
        eos_logits,
        cross_attention
    ):
        
        if len(outputs_before_postnet_spectrogram.shape) == 4:
                outputs_before_postnet_spectrogram = outputs_before_postnet_spectrogram.flatten(1, 2)

        if len(outputs_after_postnet_spectrogram.shape) == 4:
            outputs_after_postnet_spectrogram = outputs_after_postnet_spectrogram.flatten(1, 2)
            
        if len(eos_logits.shape) == 3:
            eos_logits = eos_logits.flatten(1, 2)

        reduction_factor = self.discretizer_z.config["reduction_factor"]
        return \
            outputs_before_postnet_spectrogram[:, :-reduction_factor, ...], \
            outputs_after_postnet_spectrogram[:, :-reduction_factor, ...], \
            eos_logits[:, :-reduction_factor, ...], \
            (c_att[..., :-1,:] for c_att in cross_attention)
        
                
    def forward(self, x_ids, x_mask, z_ids, z_mask, data_type, z_labels, z_labels_attention_mask, teacher_force = True ,stage='learn') -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        
        outputs = {}
        labels = {}
        
        #z is audio (already sort of embed)
        #problem here --> Decoder embeddging expects spectrum, not raw audio
        reduction_factor = self.discretizer_z.config["reduction_factor"]
        z_embeds = self.discretizer_z.decoder_embedding_from_id(z_labels)
        z_attention_mask = z_labels_attention_mask[:, reduction_factor-1::reduction_factor]
        #NOT WORKING

        forced_z_length = int(z_labels.size(1)/reduction_factor) + 1 if not teacher_force else None #-1 cause prepending embedding is added


        if (data_type[0] and data_type[1]) or stage!='learn':
            
            xz_outputs = self.auto_reg_wrapped_model_xz(input_ids=x_ids, output_embeds_dec=z_embeds,output_attention_mask=z_attention_mask, teacher_force_output=True)
            outputs['xz'] = xz_outputs
            outputs['xz']["outputs_before_postnet_spectrogram"], outputs['xz']["outputs_after_postnet_spectrogram"] = xz_outputs["logit"]       
            outputs["xz"]["eos_logits"] = xz_outputs["score"]
            outputs["xz"]["attention_mask"] = x_mask
            outputs["xz"]["cross_attentions"] = xz_outputs["cross_attentions"]
            
            zx_outputs = self.auto_reg_wrapped_model_zx(input_ids=z_ids, input_attention_mask =  z_mask, output_ids= x_ids, teacher_force_output=True)
            outputs['zx'] = zx_outputs
            labels['zx'] = x_ids
            
        # else:
        #     with torch.no_grad():
        #         xz_outputs = self.auto_reg_wrapped_model_xz(input_ids=x_ids, output_embeds_dec=z_embeds,output_attention_mask=z_attention_mask, teacher_force_output=True)
        #         outputs['xz'] = xz_outputs
        #         outputs['xz']["outputs_before_postnet_spectrogram"], outputs['xz']["outputs_after_postnet_spectrogram"] = xz_outputs["logit"]       
        #         outputs["xz"]["eos_logits"] = xz_outputs["score"]
        #         outputs["xz"]["attention_mask"] = x_mask
        #         outputs["xz"]["cross_attentions"] = xz_outputs["cross_attentions"]
                
        #         zx_outputs = self.auto_reg_wrapped_model_zx(input_ids=z_ids, input_attention_mask =  z_mask, output_ids= x_ids, teacher_force_output=True)
        #         outputs['zx'] = zx_outputs
        #         labels['zx'] = x_ids
                

        # if (data_type[0] and not data_type[1]) or (stage!='learn') or (data_type[0] and data_type[1] and self.usexzx_with_supervised_training):
        #     xzx_outputs = self.symbolic_autoencoder_wrapper_xzx(x_ids=z_ids, z_ids=x_ids)
        #     outputs['xzx'] = xzx_outputs
        #     outputs['xzx']['logit'] = outputs['xzx']['logit_z']
        #     labels['xzx'] = x_ids

        if (data_type[1] and not data_type[0]) or (stage!='learn'): #or (data_type[0] and data_type[1]):
            
            bsz = z_ids.size(0)
            
            #prepend bos token
            y_prepending_ids =  torch.tensor([self.processor_z.tokenizer.bos_token_id]).expand(bsz, 1).to(z_ids.device)
            
            zxz_outputs = self.symbolic_autoencoder_wrapper_zxz(
                x_ids=z_ids,
                x_attention_mask=z_mask,
                y_prepending_ids=y_prepending_ids,
                z_embeds_dec=z_embeds if teacher_force else None, 
                z_attention_mask=z_attention_mask if teacher_force else None,
                forced_z_length = forced_z_length,
                teacher_force_z=teacher_force, 
            )
            
            outputs['zxz'] = zxz_outputs
            outputs['zxz']['logit'] = outputs['zxz']['logit_z']
            labels['zxz'] = z_ids
            
            outputs["zxz"] = {}
        
            speech = zxz_outputs["id_z"]
            
            outputs_before_postnet_spectrogram, outputs_after_postnet_spectrogram = zxz_outputs["logit_z"]
            
            outputs["zxz"]["audio"] = speech
            
            outputs["zxz"]["outputs_before_postnet_spectrogram"] = outputs_before_postnet_spectrogram
            
            outputs["zxz"]["outputs_after_postnet_spectrogram"] = outputs_after_postnet_spectrogram
            
            outputs["zxz"]["eos_logits"] = zxz_outputs["score_z"]
            
            outputs["zxz"]["y_attention_mask"] = zxz_outputs["y_attention_mask"]
            
            outputs["zxz"]["z_cross_attentions"] = zxz_outputs["z_cross_attentions"]
            
            outputs["zxz"]["id_y"] = zxz_outputs["id_y"]
        
        
        # #TODO: won't work for non-teacher forcing for scores
        # zxz_outputs = self.symbolic_autoencoder_wrapper_zxz(
        #     x_ids=z_ids,
        #     x_attention_mask=z_mask,
        #     z_embeds_dec=z_embeds, 
        #     z_attention_mask=z_attention_mask,
        #     teacher_force_z=teacher_force,
            
        # 
        return outputs, labels
    
    def on_load_checkpoint(self, checkpoint):
        checkpoint["optimizer_states"] = []
        checkpoint["callbacks"] = {}
    
    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx= batch_idx, stage='val')
        self.log("val/loss", loss, **self.logging_kwargs['val'])
        return loss

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage, batch_idx = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # stage = self.trainer.state.stage._value_ # stages are 'fit', 'validate', 'test', 'predict', 'sanity_check'
        x_ids, x_mask, z_ids, z_mask, data_type, z_labels, z_labels_attention_mask = \
            batch['x_ids'], batch['x_mask'], batch['z_ids'], batch['z_mask'], batch['data_type'], batch['z_labels'], batch['z_labels_attention_mask']
        
        data_type = torch.all(data_type, dim=0)
        teacher_force = True
        if stage == "test":
            teacher_force = False
        elif stage == "val":
            teacher_force = True if torch.rand(1).item() < 0.5 else False
        else:
            #get number between 0 and 1 (uniform distribution)
            # prob = torch.rand(1)
            # if prob > self.hparams.model_params.teacher_force_prob:
            teacher_force = True
        
  
        outputs, labels = self.forward(
            x_ids,
            x_mask,
            z_ids,
            z_mask,
            data_type,
            z_labels,
            z_labels_attention_mask,
            teacher_force=teacher_force,
            stage=stage
        )

        # TODO: 
        # 2. Logging of audio occasionally
        
        
        #output_image = self.fold(outputs['zxz']['logit'][:, :-1, ...].permute(0, 2, 1))
        # compute losses, predictions and update metrics
        
        #zxz loss
        loss_zx = None
        loss_zxz = None
        loss_xz = None
        total_loss = 0

        if "zxz" in outputs:
            #sometimes output can be (batch_size, seq_len, reduction_factor, vocab_size) instead of (batch_size, seq_len * reduction_factor, vocab_size)
            outputs_before_postnet_spectrogram = outputs['zxz']['outputs_before_postnet_spectrogram']
            outputs_after_postnet_spectrogram = outputs['zxz']['outputs_after_postnet_spectrogram']
            logits = outputs['zxz']['eos_logits']
            
            attention_mask = outputs["zxz"]["y_attention_mask"]
            cross_attentions = outputs["zxz"]["z_cross_attentions"]
            (
                outputs_before_postnet_spectrogram,
                outputs_after_postnet_spectrogram,
                logits,
                cross_attentions
            ) = \
                self._process_audio_outputs(
                    outputs_before_postnet_spectrogram,
                    outputs_after_postnet_spectrogram,
                    logits,
                    cross_attentions
                )
            try:
                #sometimes this randomly fails not sure why
                loss_zxz = self.criterion_zxz(
                    attention_mask,
                    outputs_before_postnet_spectrogram,
                    outputs_after_postnet_spectrogram,
                    logits,
                    z_labels,
                    cross_attentions,
                )
            except:
                loss_zxz = 0
               
            total_loss += loss_zxz
            if stage == "val" and teacher_force:
                self.losses[stage]['zxz']['tf'](loss_zxz)
                self.log(f"{stage}/zxz/tf/loss", self.losses[stage]["zxz"]['tf'], **self.logging_kwargs[stage])
            else:
                self.losses[stage]['zxz']['output'](loss_zxz)
                self.log(f"{stage}/zxz/loss", self.losses[stage]["zxz"]['output'], **self.logging_kwargs[stage])
        
        if "xz" in outputs:    
            xz_attention_mask = outputs["xz"]["attention_mask"]
            xz_outputs_before_postnet_spectrogram = outputs['xz']['outputs_before_postnet_spectrogram']
            xz_outputs_after_postnet_spectrogram = outputs['xz']['outputs_after_postnet_spectrogram']
            xz_logits = outputs['xz']['eos_logits']
            xz_cross_attentions = outputs['xz']['cross_attentions']
            
            (
                xz_outputs_before_postnet_spectrogram,
                xz_outputs_after_postnet_spectrogram,
                xz_logits,
                xz_cross_attentions
            ) = \
                self._process_audio_outputs(
                    xz_outputs_before_postnet_spectrogram,
                    xz_outputs_after_postnet_spectrogram,
                    xz_logits,
                    xz_cross_attentions
                )
            

            loss_xz = self.criterion_xz(
                xz_attention_mask,
                xz_outputs_before_postnet_spectrogram,
                xz_outputs_after_postnet_spectrogram,
                xz_logits,
                z_labels,
                xz_cross_attentions,
            )
            
            
            # if (data_type[0] and data_type[1]) or stage!='learn':
            total_loss += loss_xz
            self.losses[stage]['xz']['output'](loss_xz)
            self.log(f"{stage}/xz/loss", self.losses[stage]["xz"]['output'], **self.logging_kwargs[stage])
            # else:
            #     self.losses[stage]['xz']['no_grad'](loss_xz)
            #     self.log(f"{stage}/xz/no_grad_loss", self.losses[stage]["xz"]['no_grad'], **self.logging_kwargs[stage])
                
        if "zx" in outputs:
                        
            loss_zx = self.criterion_zx(
                outputs['zx']["logit"].permute(0, 2, 1)[..., :-1],
                labels['zx'][..., 1:],
            )
            
            # if (data_type[0] and data_type[1]) or stage!='learn':
            total_loss += loss_zx
            self.losses[stage]['zx']['output'](loss_zx)
            self.log(f"{stage}/zx/loss", self.losses[stage]["zx"]['output'], **self.logging_kwargs[stage])
            # else:
            #     self.losses[stage]['zx']['no_grad'](loss_zx)
            #     self.log(f"{stage}/zx/no_grad_loss", self.losses[stage]["zx"]['no_grad'], **self.logging_kwargs[stage])
        
        # self.losses[stage]['zxz']['continous output'].update(loss)
        if (stage == "val" and  batch_idx < 5) or (stage == "test"):
            self._log_output_samples(outputs["zxz"]["audio"], z_labels, x_ids, outputs["zxz"]["id_y"], batch['ids'], stage = stage)
        # log 10 images every epoch
        # if self.current_epoch % 2 == 0:
        #     self.logger.log_image(key='input_image', images=[unprocessed_z[i] for i in range(10)])
        #     self.logger.log_image(key='output_image', images=[output_image[i] for i in range(10)])
        return total_loss
    
    
    def _log_output_samples(self, z_pred, z_true, gt_text, pred_text, ids, stage, freq=400, num_audio=5) -> None:
        
        # log 10 images every 2 epochs
        
        
        columns = ["Sample id", "Generated Vocoded Audio (zhat)", "Generated Text (x)","Ground Truth Vocoded Audio", "Ground Truth Text"]
        data = []

        if z_true.shape[0] <= num_audio:
            num_audio = None
                    
        with torch.no_grad():
            ground_truth_audio_vocoded_audio = self.vocoder(z_true[:num_audio])
            
        gt_text = self.processor_z.tokenizer.batch_decode(gt_text[:num_audio], skip_special_tokens=False)
        pred_toks = pred_text[:num_audio]
        pred_txt = self.processor_z.tokenizer.batch_decode(pred_toks, skip_special_tokens=False)
        
        for i,(pred,gt) in enumerate(zip(z_pred[:num_audio].cpu().detach().numpy(), ground_truth_audio_vocoded_audio.cpu().detach().numpy())):
            
            pred_audio = wandb.Audio(pred, sample_rate=16000)
            gt_audio = wandb.Audio(gt, sample_rate=16000)
            data.append([ids[i], pred_audio, pred_txt[i],  gt_audio, gt_text[i]])

        self.logger.log_table(key = f"{stage}/audio", columns=columns, data=data, step = self.global_step)
            


    def _initialize_autoreg_wrapped_models(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.sequence_model_xz = hydra.utils.instantiate(models_config.sequence_model_xz.model)
        self.sequence_model_xz.train()
        self.sequence_model_zx = hydra.utils.instantiate(models_config.sequence_model_zx.model)
        self.sequence_model_zx.train()
        self.sequence_model_xz_unwrapped = hydra.utils.instantiate(models_config.sequence_model_xz.model_unwrapper, self.sequence_model_xz)
        self.sequence_model_zx_unwrapped = hydra.utils.instantiate(models_config.sequence_model_zx.model_unwrapper, self.sequence_model_zx)
        
        # making it a dictionary from an OmegaConf object
        discretizer_z_config = OmegaConf.to_container(models_config.discretizer_z.config, resolve=True)
        discretizer_x_config = OmegaConf.to_container(models_config.discretizer_x.config, resolve=True)
        
        if models_config.get('inherit_model_embedding_weights_for_discretizers', False):
            discretizer_z_config["encoder_embedding"] = self.sequence_model_zx_unwrapped['encoder_embedding']
            discretizer_z_config["decoder_embedding"] = self.sequence_model_xz_unwrapped['decoder_embedding']
            discretizer_z_config["linear_head"] = self.sequence_model_xz_unwrapped['linear_head']
            
            discretizer_x_config["encoder_embedding"] = self.sequence_model_xz_unwrapped['encoder_embedding']
            discretizer_x_config["decoder_embedding"] = self.sequence_model_zx_unwrapped['decoder_embedding']
            discretizer_x_config["linear_head"] = self.sequence_model_zx_unwrapped['linear_head']
        
        models_config.discretizer_z.pop('config')
        models_config.discretizer_x.pop('config')
        # {'decoder_embedding_dim': 768, 'vocab_size': 80, 'encoder_embedding_dim': 768, 'unembedding_dim': 81}

        self.discretizer_z = hydra.utils.instantiate(models_config.discretizer_z, configs=discretizer_z_config)
        #for now, I'll be initializing a random speaker embedding
        
        # self.speaker_embeddings = torch.tensor(load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[7306]["xvector"]).unsqueeze(0)
        self.speaker_embeddings = (torch.zeros(1, models_config['speaker_embedding_dim'])).requires_grad_(False)
        self.discretizer_z.set_speaker_embeddings(self.speaker_embeddings)
        
        self.discretizer_x = hydra.utils.instantiate(models_config.discretizer_x, configs=discretizer_x_config)

        # config for the autoregressive wrapper
        #text to speech (x to z)
        # output_prepending_embeds_dec = self.sequence_model_xz.speecht5.decoder.prenet(
        #     torch.zeros(1, 1, text2speech_model.config.num_mel_bins),
        #     speaker_embeddings
        # )
        models_config_sequence_model_xz = OmegaConf.to_container(models_config.sequence_model_xz.config, resolve=True)
        models_config_sequence_model_zx = OmegaConf.to_container(models_config.sequence_model_zx.config, resolve=True)
       
        x_processor = hydra.utils.instantiate(models_config.sequence_model_xz.processor)
        xz_control_token_ids = {'input_pad_token_id': x_processor.tokenizer.pad_token_id}
        self.spectrogram_pad = torch.zeros(1, 1, discretizer_z_config["dimensions"]["vocab_size"])
        with torch.no_grad():
            self.output_prepending_embeds_dec = self.sequence_model_xz.speecht5.decoder.prenet(
                self.spectrogram_pad,
                self.speaker_embeddings
            )
        output_pad_embed_dec = self.output_prepending_embeds_dec.clone()
        output_pad_embed_enc = self.output_prepending_embeds_dec.clone()
    
        models_config_sequence_model_xz["control_token_ids" ]= xz_control_token_ids
        models_config_sequence_model_xz["output_prepending_embeds_dec"] = self.output_prepending_embeds_dec
        models_config_sequence_model_xz["output_pad_embed_dec"] = output_pad_embed_dec
        models_config_sequence_model_xz["output_pad_embed_enc"] = output_pad_embed_enc
        models_config_sequence_model_xz["vocoder"] = hydra.utils.instantiate(models_config_sequence_model_xz["vocoder"])
        models_config_sequence_model_xz["vocoder"].eval()
        self.vocoder = models_config_sequence_model_xz["vocoder"]
     
        z_processor = hydra.utils.instantiate(models_config.sequence_model_zx.processor)
        models_config_sequence_model_zx["control_token_ids" ]= {
            'input_pad_token_id': z_processor.tokenizer.pad_token_id,
            'output_eos_token_id': z_processor.tokenizer.eos_token_id, 
            'output_pad_token_id': z_processor.tokenizer.pad_token_id,
            'output_unknown_token_id': z_processor.tokenizer.unk_token_id
        }
        models_config_sequence_model_zx['output_prepending_ids'] = [z_processor.tokenizer.bos_token_id]
        
        autoreg_sequence_model_xz = {'_target_': models_config.sequence_model_xz._target_, 'config': models_config_sequence_model_xz}
        autoreg_sequence_model_zx = {'_target_': models_config.sequence_model_zx._target_, 'config': models_config_sequence_model_zx}
        
        self.auto_reg_wrapped_model_xz = hydra.utils.instantiate(autoreg_sequence_model_xz, 
                vector_model=self.sequence_model_xz_unwrapped["vector_model"], input_discretizer=self.discretizer_x, output_discretizer=self.discretizer_z,)
        self.auto_reg_wrapped_model_zx = hydra.utils.instantiate(autoreg_sequence_model_zx, 
                vector_model=self.sequence_model_zx_unwrapped["vector_model"], input_discretizer=self.discretizer_z, output_discretizer=self.discretizer_x,)

    def _initialize_symbolic_autoencoder_wrappers(self, models_config: Dict[str, torch.nn.Module]) -> None:
        self.symbolic_autoencoder_wrapper_zxz = hydra.utils.instantiate(models_config.symbolic_autoencoder_wrapper_zxz, 
                self.auto_reg_wrapped_model_zx, self.auto_reg_wrapped_model_xz)
        self.symbolic_autoencoder_wrapper_zxz.transform_xy_outputs_to_y_inputs = self.symbolic_autoencoder_wrapper_zxz.config['transform_xy_outputs_to_y_inputs']
      
        
    def _set_discretizer_weights(self, target, source, clone=False):
        """
        Sets the weights (and bias if applicable) of the target layer from the source layer.
        Supports both Embedding and Linear layers.
        
        Args:
            target (nn.Module): The target layer (embedding or linear) whose weights are being set.
            source (nn.Module): The source layer from which to copy weights (and bias if applicable).
            clone (bool): Whether to clone the weights and bias to avoid in-place modification.
        """
        if clone:
            target = copy.deepcopy(target)
        else:
            target = source