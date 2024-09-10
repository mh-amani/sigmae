from typing import Any, Dict
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper
import torch

class ImageInputAutoRegWrapper(AutoRegWrapper):
    
    NO_INPUT_ATTENTION_MASK: bool = True
    
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
        input_embeds_enc = None,
        output_ids = None,
        output_embeds_enc = None,
        output_embeds_dec = None,
        output_attention_mask=None,
        **kwargs,         
    ):
        
        args_for_super = {
            "input_ids": None,
            "input_attention_mask": None,
            "input_embeds_enc": input_embeds_enc,
            "output_ids": output_ids,
            "output_embeds_enc": output_embeds_enc,
            "output_embeds_dec": output_embeds_dec,
            "output_attention_mask": output_attention_mask,
        }
        
        return super().prepare_inputs(**args_for_super)
        
    
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
    ):
        return {
            "pixel_values": input_embeds,
            "decoder_inputs_embeds": output_embeds_dec,
            "decoder_attention_mask": output_attention_mask,            
        }