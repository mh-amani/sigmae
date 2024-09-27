from .auto_reg_wrapper import AbstractAutoRegWrapper
import torch
from typing import Dict, Any
import numpy as np

class IdentityAutoRegWrapper(AbstractAutoRegWrapper):
    

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
        
        # self.output_prepending_embeds_dec = self.config.get("output_prepending_embeds_dec", None)
        # if self.output_prepending_embeds_dec is not None:
        #     self.output_prepending_embeds_dec.requires_grad_(True)
        # assert self.output_prepending_embeds_dec is not None, "output_prepending_embeds_dec should be provided"
    
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
        assert input_embeds_enc is not None, "input_embeds_enc should be provided"
        # assert everything else is None
        for arg in [input_ids, output_ids, input_attention_mask, output_attention_mask, output_embeds_enc, output_embeds_dec]:
            assert arg is None, f"{arg} should be None"
        
       
        return {
            "input": input_embeds_enc,
        }
    
    def forward(
        self,
        input_embeds,
        **kwargs
    ):
        return self.model(input_embeds)
    
  