from typing import Any, Dict
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper
import torch

class AudioInputAutoRegWrapper(AutoRegWrapper):
    
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
        
        
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
        **kwargs
    ):
        return {
            "inputs": input_embeds,
            "attention_mask": input_attention_mask,
            "decoder_inputs_embeds": output_embeds_dec,
            "decoder_attention_mask": output_attention_mask,            
        }