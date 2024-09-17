from typing import Any, Dict
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper
import torch

class AudioInputAutoRegWrapper(AutoRegWrapper):
    
    NO_INPUT_ATTENTION_MASK: bool = True
    #TODO: I may be overfitting on T5Speech models here. Will need to generalize this later.
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ):
        super().__init__(
            vector_model=vector_model,
            input_discretizer=input_discretizer,
            output_discretizer=output_discretizer,
            config=config,
        )
    
    def prepare_inputs(
        self,
        input_ids=None,
        input_attention_mask=None,
        input_embeds_enc=None,
        output_ids=None,
        output_embeds_enc=None,
        output_embeds_dec=None,
        output_attention_mask=None,
    ):
        if input_ids is not None:
            enc_emb_out = self.input_discretizer.encoder_embedding_from_id(input_ids, input_attention_mask)
            
            input_embeds_enc, input_attention_mask = enc_emb_out if isinstance(enc_emb_out, tuple) else (enc_emb_out, input_attention_mask)
            
            if isinstance(enc_emb_out, tuple) and input_attention_mask is None:
                raise ValueError("input_attention_mask is required ! for models using your input discretizer")
            
            if input_attention_mask is None:
                input_attention_mask = torch.logical_not(torch.eq(input_ids, self.control_token_ids['input_pad_token_id']))
        
        return super().prepare_inputs(
            input_ids=None,
            input_attention_mask=input_attention_mask,
            input_embeds_enc=input_embeds_enc,
            output_ids=output_ids,
            output_embeds_enc=output_embeds_enc,
            output_embeds_dec=output_embeds_dec,
            output_attention_mask=output_attention_mask,
        )
        
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
        **kwargs
    ):
        return {
            "inputs_embeds": input_embeds,
            "attention_mask": input_attention_mask,
            "decoder_inputs_embeds": output_embeds_dec,
            "decoder_attention_mask": output_attention_mask,            
        }