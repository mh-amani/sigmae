from symbolic_bottleneck.auto_reg_wrapper import OutContAutoRegWrapper
from typing import Dict, Any
import torch

class AudioOutAutoRegWrapper(OutContAutoRegWrapper):
    
    REQUIRED_CONFIG_KEYS = [
        "control_token_ids.input_pad_token_id",
        "output_prepending_embeds_dec",
        "vocoder"
    ]
    
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ) -> None:

        super().__init__(
            vector_model=vector_model,
            input_discretizer=input_discretizer,
            output_discretizer=output_discretizer,
            config=config,
        )

        self.vocoder = config["vocoder"]
        self.output_pad_embed_dec = self.config['output_pad_embed_dec'].requires_grad_(True)
        self.output_pad_embed_enc = self.config['output_pad_embed_enc'].requires_grad_(True)
    
    def teacher_forced_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
    ):
        outputs = super().teacher_forced_model_forward(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
        )
        # batch_size, seq_len * reduction_factor, num_mel_bins
        bsz, _, num_mel_bins = outputs['id'].shape
        reduction_factor = self.output_discretizer.reduction_factor
        
        ids_device = outputs['id'].device
        outputs['id'] = torch.cat([outputs['id'][:, i::reduction_factor, :].unsqueeze(-2) for i in range(0,reduction_factor)], dim=-2).to(ids_device)
        
        outputs['eos_flag'] = outputs["sampled_eos"]
        outputs['p_not_eos'] = outputs["p_not_eos"]
        outputs['output_attention_mask'] = output_attention_mask
        return outputs 
    
    def discretize_output(self, hidden_state, cross_attentions = None ,past_preds = None ,teacher_forced = False, output_ids=None, output_attention_mask=None):
        if teacher_forced:
            discretized_output = self.output_discretizer(
                hidden_state,
                supervision=True,
            )
            
        else:
            discretized_output = self.output_discretizer(
                hidden_state,
                past_preds = past_preds,
                supervision=False,
            )
        
        discretized_output["cross_attentions"] = cross_attentions
        return discretized_output   
    
    def one_step_sequential_forward(
        self,
        input_embeds, 
        output_embeds_dec,
        ids,
        input_attention_mask=None,
        output_attention_mask=None,
        last_step_states={},
    ) -> Dict[str, Any]:
    
         # If you're using cached key values or encoder outputs or what not, you should pass them here. and you should check
        # the model source code to see if the gradients backpropagate through these values from next step or not.
        args_for_seq_forward = self.prepare_args_for_model_forward(
            input_embeds=input_embeds,
            input_attention_mask=input_attention_mask,
            output_embeds_dec=output_embeds_dec,
            output_attention_mask=output_attention_mask,
        )
        
        args_for_seq_forward = {**last_step_states, **args_for_seq_forward}
        if "use_cache" not in args_for_seq_forward:
            args_for_seq_forward['use_cache'] = self.config['use_last_step_states']
        
        output = self.model(output_hidden_states = True, output_attentions= True, return_dict = True , **args_for_seq_forward)

        # output_embed = output['decoder_hidden_states'][-1]
        output_embed = output[self.hidden_state_key][-1][:, -1:, :] # does the same thing as above size: (batch_size, 1, hidden_size)
        # output of the encoder to be used in generation, I don't get why the key values are needed though, only query values are useful
        # maybe using this is blocking the gradient flow, I should check this
        encoder_last_hidden_state = output.encoder_last_hidden_state
        past_key_values = output.past_key_values
        hidden_state = output.encoder_hidden_states
        encoder_attentions = output.encoder_attentions

        # past_preds = torch.zeros(bsz, 1, self.model.config.reduction_factor, self.model.config.num_mel_bins).to(ids.device)
        # if ids.shape[1] > 0:
        #     past_preds = torch.cat([past_preds, ids], dim=1)
        past_preds = ids if ids.shape[1] > 0 else None
        discretizer_output = self.discretize_output(output_embed, past_preds =  past_preds,teacher_forced=False)
        
        
        seq_forward_output = {
            # 'id': discretizer_output['id'],
            # 'score': discretizer_output['score'],
            # 'logit': discretizer_output['logit'], 
            # 'vector_encoder': discretizer_output['vector_encoder'],
            # 'vector_decoder': discretizer_output['vector_decoder'], 
            # 'quantization_loss': discretizer_output['quantization_loss'],
            'past_key_values': past_key_values,
            'encoder_last_hidden_state': encoder_last_hidden_state, 
            'hidden_state': hidden_state,
            'encoder_attentions': encoder_attentions,
            **discretizer_output
        }
        
        current_eos_flag = seq_forward_output["sampled_eos"]

        p_eos = seq_forward_output["p_eos"]

        p_not_eos = seq_forward_output["p_not_eos"]
        return({
            'id': seq_forward_output['id'],
            'score': seq_forward_output['score'],
            'logit': seq_forward_output['logit'], 
            'vector_encoder': seq_forward_output['vector_encoder'],
            'vector_decoder': seq_forward_output['vector_decoder'], 
            'quantization_loss': seq_forward_output['quantization_loss'],
            'eos_flag': current_eos_flag,
            'p_eos': p_eos, 
            'p_not_eos': p_not_eos,
            'past_key_values': seq_forward_output["past_key_values"],
            'encoder_last_hidden_state': seq_forward_output["encoder_last_hidden_state"], 
            'hidden_state': seq_forward_output["hidden_state"],
            'encoder_attentions': seq_forward_output["encoder_attentions"],
            'cross_attentions': output.cross_attentions,
        })
                
     
    
    def return_output_dict(self, outputs) -> Dict[str, Any]:
        outputs_before_postnet_spectrogram = outputs['id']
        outputs_after_postnet_spectrogram = outputs["logit"]
        scores_eos = outputs['score']
        #from (bsz,seq_len, reduction_factor, spectrogram_dim) to (bsz,seq_len * reduction_factor, spectrogram_dim)
        postnet_input = outputs['id'].flatten(1,2)
        #from (bsz,seq_len * reduction_factor, spectrogram_dim) to (bsz,seq_len * reduction_factor, spectrogram_dim)
        vector_encoder = self.output_discretizer.linear_head.postnet(postnet_input)
        with torch.no_grad():
            vocoded_ids = self.vocoder(vector_encoder)
        
        return {
            'id': vocoded_ids,
            'score': scores_eos,
            'score_list': None,
            'logit': (outputs_before_postnet_spectrogram, outputs_after_postnet_spectrogram),
            'vector_encoder': vector_encoder,
            'vector_decoder': outputs['vector_decoder'],
            'output_attention_mask': outputs['output_attention_mask'],
            'eos_flag': outputs['eos_flag'],
            'p_not_eos': outputs['p_not_eos'],
            'quantization_loss': outputs['quantization_loss'],
            "cross_attentions": outputs["cross_attentions"],
        }
    
    def prepare_seq_forward_params(
        self,
        input_embeds = None,
        input_attention_mask = None, 
        output_embeds_enc = None,
        output_embeds_dec = None,
        output_attention_mask = None,
        max_output_length = None,
        preprend_length = None,
    ) -> Dict[str, Any]:
        
        params = super().prepare_seq_forward_params(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask, 
            output_embeds_enc = output_embeds_enc,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
            max_output_length = max_output_length,
            preprend_length = preprend_length,
        )
        output_dim = self.output_discretizer.vocab_size
        ids = torch.zeros(output_embeds_dec.shape[0], max_output_length-preprend_length, self.output_discretizer.reduction_factor ,output_dim).to(input_embeds.device)
        
        p_not_eoss = [torch.ones(input_embeds.shape[0], 1, requires_grad=True).to(input_embeds.device)]
        eos_flags = torch.zeros(input_embeds.shape[0], 1, dtype=torch.bool).to(input_embeds.device)
        
        #scores are for p_eos, p_not_eos
        scores = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, self.output_discretizer.reduction_factor).to(input_embeds.device)
        #logits is outputs_after_postnet
        logits = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, self.output_discretizer.reduction_factor, output_dim).to(input_embeds.device)

        cross_attentions = None
        
        params["cross_attentions"] = cross_attentions
        params["logits"] = logits
        params["scores"] = scores
        params['p_not_eoss'] = p_not_eoss
        params['eos_flags'] = eos_flags
        params['ids'] = ids
        output_attention_masks = output_attention_mask.float().requires_grad_(True).to(input_embeds.device)
        
        params['output_attention_masks'] = output_attention_masks
        
        return params
    
    def prepare_one_step_seq_forward_params(
        self,
        step,
        preprend_length,
        input_embeds,
        input_attention_mask,
        output_embeds_decs,
        output_attention_masks,
        ids,
        **kwargs
    ):
    
        return {
            "input_embeds": input_embeds,
            "input_attention_mask": input_attention_mask,
            "output_embeds_dec": output_embeds_decs[:, :step + preprend_length],
            "output_attention_mask": output_attention_masks[:, :step + preprend_length] > 0, # this is to make sure that the attention mask is binary, otherwise we had NAN in the loss! for longish sequences
            "ids": ids[:, :step],
        }
        
    def should_continue_forward(self, eos_flags):
        
        if self.forced_z_length is not None:
            return self.step < self.forced_z_length
        
        if torch.any(eos_flags):
            return False
     
        return not torch.all(eos_flags)
    
        
        
    def post_one_step_seq_forward(
        self,
        current_output,
        step,
        ids,
        scores,
        logits,
        p_not_eoss,
        output_attention_masks,
        eos_flags,
        quantization_loss,
        output_embeds_encs,
        output_embeds_decs,
        cross_attentions,
        **kwargs,
    ):
        
        outputs = super().post_one_step_seq_forward(
            current_output=current_output,
            step=step,
            ids=ids,
            quantization_loss=quantization_loss,
            output_embeds_encs=output_embeds_encs,
            output_embeds_decs=output_embeds_decs,
            **kwargs
        )
        
        outputs["cross_attentions"] = current_output["cross_attentions"]
        p_not_eoss.append( current_output['p_not_eos'] * p_not_eoss[step])
        output_attention_masks = torch.cat((output_attention_masks, self.attention_mask(eos_flags, p_not_eoss[step])[:, 0].reshape(-1, 1)), dim=1).to(output_attention_masks.device)
        scores[:, step] = current_output['score']
        logits[:, step] = current_output['logit']
        
        outputs["p_not_eoss"] = p_not_eoss
        outputs["output_attention_masks"] = output_attention_masks
        outputs["eos_flags"] = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
        outputs["scores"] = scores
        outputs["logits"] = logits
        
        return outputs
        
    def attention_mask(self, eos_flags, p_not_eos):
        true_attention_mask = torch.logical_not(eos_flags)
        if self.config['soft_average']['p_eos_forward']:
            return p_not_eos
        elif self.config['soft_average']['p_eos_backward']:
            return true_attention_mask + (p_not_eos - p_not_eos.detach())
        else:
            return true_attention_mask
        
    def return_seq_forward_output_dict(
        self,
        step,
        preprend_length,
        ids,
        scores,
        logits,
        p_not_eoss,
        output_embeds_encs,
        output_embeds_decs,
        output_attention_masks,
        eos_flags,
        quantization_loss,
        cross_attentions,
        **kwargs
    ):
                
        # cut the tensors to the actual length, remove 0s and 1s.
        ids = ids[:, :step]
        scores = scores[:, :step]
        logits = logits[:, :step]
        p_not_eoss = p_not_eoss[1:]
        output_embeds_encs = output_embeds_encs[:, :step]
        output_embeds_decs = output_embeds_decs[:, :step + preprend_length]
        output_attention_masks = output_attention_masks[:, :step + preprend_length]

        # enforce pad tokens and embeds where attention mask is zero
        binary_attention_mask = (output_attention_masks > 0).unsqueeze(-1).unsqueeze(-1)

        # pad out the random tokens after first eos token
        ids = ids * binary_attention_mask[:, -ids.shape[1]:] + torch.zeros_like(binary_attention_mask[:, -ids.shape[1]:]) * \
                    torch.logical_not(binary_attention_mask)[:, -ids.shape[1]:]
        #skipfirstcauseno prepend
        output_embeds_encs = output_embeds_encs * output_attention_masks[:,1:].unsqueeze(-1) + \
            self.output_pad_embed_enc.to(output_attention_masks.device) * torch.logical_not(output_attention_masks[:,1:]).unsqueeze(-1)
        output_embeds_decs = output_embeds_decs * output_attention_masks.unsqueeze(-1) + \
            self.output_pad_embed_dec.to(output_attention_masks.device) * torch.logical_not(output_attention_masks).unsqueeze(-1)

        
        return {
            'id': ids,
            'score': scores,
            'logit': logits,
            'vector_encoder': output_embeds_encs,
            'vector_decoder': output_embeds_decs,
            'quantization_loss': quantization_loss,
            'output_attention_mask': output_attention_masks,
            'eos_flag': eos_flags,
            'p_not_eos': p_not_eoss,
            'cross_attentions': cross_attentions,
        }
        