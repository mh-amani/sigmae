from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from torch.nn import LayerNorm,BatchNorm1d
import math
from dataclasses import dataclass

# discretizer_config = {
#         '_target_': 'blocks.modules.discrete_bottlenecks.abstract_discrete_layer.AbstractDiscreteLayer',
#         'config': {
#             'dimensions': None,
#             'quantize_vector': True, 'temperature': 1.0,
#             'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
#             'linear_head_trainable': False, 
#             'encoder_embedding': None, 'decoder_embedding': None, 
#             'linear_head': None,
#         }
#     }


class AbstractDiscreteLayer(nn.Module): 
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self._initialize_dimensions(config)
        self._initialize_encoder_embedding()
        self._initialize_decoder_embedding()
        self._initialize_linear_head()

    def _initialize_dimensions(self, config):
        dimensions = config.get('dimensions', {})
        if dimensions is None:
            dimensions = {}
        self.vocab_size = dimensions.get('vocab_size', None)
        self.decoder_embedding_dim = dimensions.get('decoder_embedding_dim', None)
        self.unembedding_dim = dimensions.get('unembedding_dim', None)
        self.encoder_embedding_dim = dimensions.get('encoder_embedding_dim', None)

        # Other configuration parameters
        self.linear_head_scale = config.get('linear_head_scale', 1.0)
        self.quantize_vector = config.get('quantize_vector', True)
        self.temperature = config.get('temperature', 1.0)
        self.label_smoothing_scale = config.get('label_smoothing_scale', 0.001)

    def _initialize_encoder_embedding(self):
        if self.config.get('encoder_embedding') is not None:
            if self.encoder_embedding_dim is not None or self.vocab_size is not None:
                raise ValueError('encoder_embedding cannot be provided along with encoder_embedding_dim or vocab_size')
            self.encoder_embedding = self.config['encoder_embedding'].requires_grad_(self.config['encoder_embedding_trainable'])
            self.encoder_embedding_dim = self.encoder_embedding.weight.shape[1]
            self.vocab_size = self.encoder_embedding.weight.shape[0]
        elif self.encoder_embedding_dim is not None and self.vocab_size is not None:
            self.encoder_embedding = nn.Embedding(self.vocab_size, self.encoder_embedding_dim)
            self.encoder_embedding.requires_grad_(self.config['encoder_embedding_trainable'])
            torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=1/math.sqrt(self.encoder_embedding_dim * self.vocab_size))
        else:
            raise ValueError('Either encoder_embedding or both encoder_embedding_dim and vocab_size must be provided')

    def _initialize_decoder_embedding(self):
        if self.config.get('decoder_embedding') is not None:
            self.decoder_embedding = self.config['decoder_embedding'].requires_grad_(self.config['decoder_embedding_trainable'])
            self.decoder_embedding_dim = self.decoder_embedding.weight.shape[1]
            assert self.vocab_size == self.decoder_embedding.weight.shape[0]
        elif self.decoder_embedding_dim is not None and self.vocab_size is not None:
            self.decoder_embedding = nn.Embedding(self.vocab_size, self.decoder_embedding_dim)
            self.decoder_embedding.requires_grad_(self.config['decoder_embedding_trainable'])
            torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=1/math.sqrt(self.decoder_embedding_dim * self.vocab_size))
        else:
            raise ValueError('Either decoder_embedding or both decoder_embedding_dim and vocab_size must be provided')

    def _initialize_linear_head(self):
        if self.config.get('linear_head') is not None:
            self.linear_head = self.config['linear_head'].requires_grad_(self.config['linear_head_trainable'])
        elif self.decoder_embedding_dim is not None and self.unembedding_dim is not None:
            self.linear_head = nn.Linear(self.decoder_embedding_dim, self.unembedding_dim)
            self.linear_head.requires_grad_(self.config['linear_head_trainable'])
            torch.nn.init.normal_(self.linear_head.weight, mean=0, std=1/math.sqrt(self.unembedding_dim * self.decoder_embedding_dim))
        else:
            raise ValueError('Either linear_head or both decoder_embedding_dim and unembedding_dim must be provided')

    def forward(self, x,**kwargs):
        continous_vector = self.linear_head(x)
        continous_vector = continous_vector * self.linear_head_scale
        # scores are between 0 and 1, and sum to 1 over the vocab dimension.
        discrete_output  = self.discretize(continous_vector,**kwargs)
        # discrete_output usually has id, score, logit, quantized_vector, quantization_loss
        return discrete_output

    def encoder_embedding_from_id(self, x):
        embeds = self.encoder_embedding(x)
        return embeds
    
    def decoder_embedding_from_id(self, x):
        embeds = self.decoder_embedding(x)
        return embeds

    @abstractmethod
    def discretize(self, x,**kwargs) -> dict:
        # the function that takes the output of the decoder and returns a discrete representation
        raise NotImplementedError




########################################################################################################################
    # code snippets and other useful stuff for debugging and checking stuff
    # def get_normalization_method(self,layer,norm_args,output_dimension):
        
    #     if norm_args.type is None:
    #         layer = lambda x: x
    #         return layer
        
    #     method_type =  norm_args['type']
    #     method_args = norm_args['args']
        
    #     if method_type == 'layer norm':
    #         return LayerNorm(normalized_shape= output_dimension,**method_args)
        
    #     elif method_type == 'batch norm':
    #         return BatchNorm1d(num_features = output_dimension,**method_args)
        
    #     else:
    #         raise ValueError('normalization method not supported: {}'.format(method_type))        



############################ unused code for when we had a unique dictionary instead of the two embeddings ################################
        # self.output_embedding = lambda x: x
        # self.decoder_embedding = lambda x: x

        # # initialize the dictionary
        # if self.configs.get('dictionary_weight', None) is not None:
        #     self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        #     self.dictionary.weight = nn.Parameter(self.configs['dictionary_weight'], 
        #             requires_grad=self.configs['dictionary_trainable'])
        # else:
        #     self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        #     self.dictionary.weight = nn.Parameter(torch.eye(self.vocab_size), requires_grad=True)
        
        # self.bottleneck_normalization_args = self.configs.get('bottleneck_normalization', None)
        # #weight norm must be done at initialization
        # if  self.bottleneck_normalization_args.get("type",None) == 'weight norm':
        #     self.encoder_embedding = nn.utils.weight_norm(self.encoder_embedding, dim=-1)
        #     self.decoder_embedding = nn.utils.weight_norm(self.decoder_embedding, dim=-1)
        #     self.encoder_embedding_normalization = lambda x: x  #weight norm is done at initialization
        #     self.decoder_embedding_normalization = lambda x: x  #weight norm is done at initialization
        # #otherwise, it's normalization is none in the Forward pass
        # else:
        #     self.encoder_embedding_normalization = self.get_normalization_method(self.encoder_embedding, norm_args=self.bottleneck_normalization_args,output_dimension = self.input_dim)
        #     self.decoder_embedding_normalization = self.get_normalization_method(self.decoder_embedding, norm_args=self.bottleneck_normalization_args,output_dimension = self.output_dim)
    


########################################################################################################################
# @dataclass
# class AbstractDiscreteLayerConfig:
#     dimensions = {
#         'embedding_dim': 1024,
#         'unembedding_dim': 250054,
#         'vocab_size': 250054,
#         'dictionary_embedding_dim': 250054 #if probability based quantization is used, this should be the same as vocab_size, if distance based quantization is used, this should be the same as the embedding_dim
#     }
#     average_eos_mask_in_backprop: bool = True
#     soft_average: bool = False
#     temperature: float = 1.0
#     label_smoothing_scale: float = 0.001
#     encoder_embedding_weight = None
#     encoder_embedding_trainable = True
#     decoder_embedding_weight = None
#     decoder_embedding_trainable = True
#     # dictionary_weight = torch.eye(250054)
#     dictionary_trainable = True
#     linear_head_weight = None
#     linear_head_trainable = True
#     # bottleneck_normalization = 'None' # can be 'None', 'layer norm', 'batch norm', 'weight norm'