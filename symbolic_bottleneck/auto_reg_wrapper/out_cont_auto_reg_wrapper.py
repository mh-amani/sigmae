from .auto_reg_wrapper import AbstractAutoRegWrapper
import torch
from typing import Dict, Any
import numpy as np

class OutContAutoRegWrapper(AbstractAutoRegWrapper):
    
    
    REQUIRED_CONFIG_KEYS = [
        "control_token_ids.input_pad_token_id",
        "output_prepending_embeds_dec"
    ]
    
    NO_INPUT_ATTENTION_MASK: bool = False
        
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
        
        
        self.output_prepending_embeds_dec = self.config.get("output_prepending_embeds_dec", None)
        if self.output_prepending_embeds_dec is not None:
            self.output_prepending_embeds_dec.requires_grad_(True)
        assert self.output_prepending_embeds_dec is not None, "output_prepending_embeds_dec should be provided"
    
    def prepare_inputs(
        self,
        input_ids=None,
        output_ids=None,
        input_attention_mask=None,
        output_attention_mask=None,
        input_embeds_enc=None,
        output_embeds_enc=None,
        output_embeds_dec=None,
    ):
                
        assert (input_ids is not None) != (input_embeds_enc is not None), "Either input_ids or input_embeds should be provided"
        if not self.NO_INPUT_ATTENTION_MASK:
            assert (input_embeds_enc is not None and input_attention_mask is not None) or (input_embeds_enc is None and input_attention_mask is None), "input_embeds and input_attention_mask should be provided together or not at all. HINT: if you're model does not accept input_attention_mask, set the class variable NO_INPUT_ATTENTION_MASK to True"

        if input_ids is not None:
            input_embeds_enc = self.input_discretizer.encoder_embedding_from_id(input_ids)
            input_attention_mask = torch.logical_not(torch.eq(input_ids, self.control_token_ids['input_pad_token_id']))

        assert (output_ids is None), 'for continuous output, output_ids should not be provided'
        assert output_embeds_enc is None, 'output_embeds_enc should not be provided, only the output_embeds_dec should be provided in case of teacher forcing'
        if output_embeds_dec is None:
            output_embeds_dec = self.output_prepending_embeds_dec.repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
        else:
            # prepend the output_embeds_dec with the output_prepending_embeds_dec
            output_embeds_dec = torch.cat((self.output_prepending_embeds_dec.repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device), output_embeds_dec), dim=1)

        return {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "input_embeds": input_embeds_enc,
            "output_embeds_enc": output_embeds_enc,
            "output_embeds_dec": output_embeds_dec,
            "output_attention_mask": output_attention_mask,
        }
    
    def teacher_forced_model_forward( self, 
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
        **kwargs
    ):
        output = super().teacher_forced_model_forward(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
        )
        return output
        
    def discretize_output(self, hidden_state, teacher_forced = False, output_ids=None, output_attention_mask=None):
        if teacher_forced:
            discretized_output = self.output_discretizer(
                hidden_state,
                supervision=True,
            )
            
        else:
            discretized_output = self.output_discretizer(
                hidden_state,
                supervision=False,
            )
        
        return discretized_output
    
    
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
    ):
        return {
            "inputs_embeds": input_embeds,
            "attention_mask": input_attention_mask,
            "decoder_inputs_embeds": output_embeds_dec,
        }
        
    def return_output_dict(self, outputs) -> Dict[str, Any]:
        return {
            'id': outputs['id'],
            'score': None,
            'score_list': None,
            'logit': None,
            'vector_encoder': outputs['vector_encoder'],
            'vector_decoder': outputs['vector_decoder'],
            'output_attention_mask': outputs['output_attention_mask'],
            'eos_flag': None,
            'p_not_eos': None,
            'quantization_loss': outputs['quantization_loss']
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
        
        # initialize tensors
        quantization_loss = 0
        
        #TODO: MIGHT UNCOMMENT LATER FOR VARIABLE LENGTH OUTPUTS
        # pad_embed_enc = self.output_discretizer.encoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
        # pad_embed_dec = self.output_discretizer.decoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
       
        # scores = torch.empty(input_embeds.shape[0], max_output_length-preprend_length, discretizer.vocab_size).to(input_embeds).fill_(0.0)
        output_dim = self.output_discretizer.unembedding_dim
        ids = torch.zeros(output_embeds_dec.shape[0], max_output_length-preprend_length, output_dim).to(input_embeds)
        
        
        #TODO: MIGHT UNCOMMENT LATER FOR VARIABLE LENGTH OUTPUTS
        # p_not_eoss = [torch.ones(input_embeds.shape[0], 1, requires_grad=True).to(input_embeds)]
        # eos_flags = torch.zeros(input_embeds.shape[0], 1, dtype=torch.bool).to(input_embeds)
        if output_embeds_enc is not None:
            output_embeds_encs = output_embeds_enc.requires_grad_(True)
        else:
            output_embeds_encs = None
        output_embeds_decs = output_embeds_dec.requires_grad_(True)
        
        
        return {
            "quantization_loss": quantization_loss,
            "ids": ids,
            "output_embeds_encs": output_embeds_encs,
            "output_embeds_decs": output_embeds_decs,
            "input_embeds": input_embeds,
            "input_attention_mask": input_attention_mask,
            "output_attention_mask": output_attention_mask,
        }
        
    def should_continue_forward(self):
        #TODO: MIGHT EDIT LATER FOR VARIABLE LENGTH OUTPUTS
        return True
    
    def prepare_one_step_seq_forward_params(
        self,
        step,
        preprend_length,
        input_embeds,
        input_attention_mask,
        output_embeds_decs,
        **kwargs
    ):
    
        return {
            "input_embeds": input_embeds,
            "input_attention_mask": input_attention_mask,
            "output_embeds_dec": output_embeds_decs[:, :step + preprend_length],
        }
        
    def post_one_step_seq_forward(
        self,
        current_output,
        step,
        ids,
        quantization_loss,
        output_embeds_encs,
        output_embeds_decs,
        **kwargs,
    ):
        
        ids[:, step] = current_output['id'].squeeze(1)
        #TODO: MIGHT UNCOMMENT LATER FOR VARIABLE LENGTH OUTPUTS
        # p_not_eoss.append( (1 - current_output['p_eos']) * p_not_eoss[step])
        # eos_flags = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
        # quantization_loss += (current_output['quantization_loss'] * torch.logical_not(eos_flags).float())
        # quantization_loss = quantization_loss * (current_output['quantization_loss'] * torch.logical_not(eos_flags))
        quantization_loss = current_output['quantization_loss'] # THIS IS A DUMMY LINE
        if step == 0:
            output_embeds_encs = current_output['vector_encoder']
        else:
            output_embeds_encs = torch.cat((output_embeds_encs, current_output['vector_encoder']), dim=1)
        output_embeds_decs = torch.cat((output_embeds_decs, current_output['vector_decoder']), dim=1)
        
        return {
            "ids": ids,
            # "eos_flags": eos_flags,
            "quantization_loss": quantization_loss,
            "output_embeds_encs": output_embeds_encs,
            "output_embeds_decs": output_embeds_decs,
            **kwargs
        }
        
    def return_seq_forward_output_dict(
        self,
        step,
        preprend_length,
        ids,
        output_embeds_encs,
        output_embeds_decs,
        quantization_loss,
        **kwargs
    ):
        ids = ids[:, :step]
        # cut the tensors to the actual length, remove 0s and 1s.
        output_embeds_encs = output_embeds_encs[:, :step]
        output_embeds_decs = output_embeds_decs[:, :step + preprend_length]

        return {
            'id': ids,
            'vector_encoder': output_embeds_encs,
            'vector_decoder': output_embeds_decs,
            'quantization_loss': quantization_loss,
        }
        
    
        
    