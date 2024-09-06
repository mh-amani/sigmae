from .abstract_bottleneck import AbstractBottleneck
from torch import torch
import math

class IdentityBottleneck(AbstractBottleneck):
    DISCRETE_BOTTLENECK: bool = False

    def discretize(self, x, **kwargs) -> dict:
        
        #logit = x # idx is the output of the linear layer in this case
        vector_encoder = self.encoder_embedding_from_id(x)
        vector_decoder = self.decoder_embedding_from_id(x)
        
        quantization_loss = torch.tensor(0.0).to(x)
                
        return {
            "id": x,
            "score": None,
            "logit": None,
            "vector_encoder": vector_encoder, 
            "vector_decoder": vector_decoder,
            "quantization_loss": quantization_loss
        }