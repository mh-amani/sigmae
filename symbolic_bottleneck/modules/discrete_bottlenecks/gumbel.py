from .abstract_bottleneck import AbstractBottleneck
import torch
from torch.nn.functional import gumbel_softmax, softmax
import numpy as np

class GumbelDiscreteBottleneck(AbstractBottleneck):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        # a probability based discretizer requires the following assertions to hold
        assert self.linear_head.in_features == self.decoder_embedding_dim
        # assert self.linear_head.out_features == self.vocab_size
        assert self.encoder_embedding_dim == self.decoder_embedding_dim

    def discretize(self, x, **kwargs) -> dict:
        logits = x / self.temperature
        score = softmax(logits, dim=-1)
        gb_score = gumbel_softmax(logits, hard=False, dim=-1)
        one_hot_idx = gumbel_softmax(logits, hard=True, dim=-1)
        idx = torch.argmax(one_hot_idx, dim=-1)
        quantize_vector = np.random.binomial(1, self.quantize_vector_prob)
        if quantize_vector:
            # idx @ self.encoder_embedding.weight - self.encoder_embedding(idx.argmax(dim=-1)) == 0.0
            quantized_vector_encoder = one_hot_idx @ self.encoder_embedding.weight
            quantized_vector_decoder = one_hot_idx @ self.decoder_embedding.weight
        elif not quantize_vector:
            quantized_vector_encoder = torch.matmul(gb_score, self.encoder_embedding.weight)
            quantized_vector_decoder = torch.matmul(gb_score, self.decoder_embedding.weight)

        quantization_loss = torch.tensor(0.0).to(x.device)

        return {"id": idx, "score": score, "logit": logits, "vector_encoder": quantized_vector_encoder, 
                "vector_decoder": quantized_vector_decoder, "quantization_loss": quantization_loss}








# old code in initialization
        # self.hard = kwargs['hard'] # if True, use argmax in forward pass, else use gumbel softmax. the backwardpass is the same in both cases

        # # supervised setting that work
        # self.dictionary_std = 1/math.sqrt(self.dictionary_dim)
        # self.input_std = 1/math.sqrt(self.dictionary_dim)
        # self.out_std = 1/math.sqrt(self.output_dim)

        # self.dictionary = torch.nn.Embedding(self.vocab_size, self.dictionary_dim)
        # torch.nn.init.normal_(self.dictionary.weight, mean=0, std=self.dictionary_std)

        # self.output_embedding = torch.nn.Linear(self.output_dim, self.vocab_size, bias=False)
        # torch.nn.init.normal_(self.output_embedding.weight, mean=0, std=self.out_std)
        # # self.output_embedding = lambda x: x
        
        # self.encoder_embedding = torch.nn.Linear(self.dictionary_dim, self.input_dim, bias=False)
        # torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=self.input_std)
        # # self.encoder_embedding = lambda x: x
        # self.decoder_embedding = torch.nn.Linear(self.dictionary_dim, self.output_dim, bias=False)
        # torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=self.input_std)
        # # self.output_embedding = lambda x: x

        # self.logit_std = math.sqrt(self.output_dim * self.out_std**2)
        # self.logit_init = math.log(self.dictionary_dim)