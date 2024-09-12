from .abstract_bottleneck import AbstractBottleneck
from torch import torch
import math

class IdentityBottleneck(AbstractBottleneck):
    DISCRETE_BOTTLENECK: bool = False
    def __init__(self, configs):
        super().__init__(configs)
        assert self.unembedding_dim is not None, "unembedding_dim must be provided"
        self._initialize_output_head()
        
    def _initialize_output_head(self):
        # let's add an activation function to the output head...
        self.output_head = self._instantiate_embedding(self.vocab_size, self.unembedding_dim)
        self.output_head.requires_grad_(self.config.get('output_head_trainable', True))
        torch.nn.init.normal_(self.output_head.weight, mean=0, std=1/math.sqrt(self.unembedding_dim * self.vocab_size))
        
    def discretize(self, x, **kwargs) -> dict:
        
        vector_encoder = self.encoder_embedding_from_id(x)
        vector_decoder = self.decoder_embedding_from_id(x)
        
        id = self.output_head(x)
    
        quantization_loss = torch.tensor(0.0).to(x)
        return {
            "id": id,
            "score": None,
            "logit": x,
            "vector_encoder": vector_encoder, 
            "vector_decoder": vector_decoder,
            "quantization_loss": quantization_loss
        }