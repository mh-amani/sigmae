from .abstract_bottleneck import AbstractBottleneck
from torch import torch
import math
from transformers.models.speecht5.modeling_speecht5 import SpeechT5SpeechDecoderPostnet
import inspect
class AudioIdentityBottleneck(AbstractBottleneck):
    DISCRETE_BOTTLENECK: bool = False
    def __init__(self, config):
        super().__init__(config)
        assert self.unembedding_dim is not None, "unembedding_dim must be provided"
        self.speaker_embedding = None
    #     self._initialize_output_head()
        # self.vocoder = config["vocoder"]
    # def _initialize_output_head(self):
    #     # let's add an activation function to the output head...
    #     self.output_head = self._instantiate_embedding(self.vocab_size, self.unembedding_dim)
    #     self.output_head.requires_grad_(self.config.get('output_head_trainable', True))
    #     torch.nn.init.normal_(self.output_head.weight, mean=0, std=1/math.sqrt(self.unembedding_dim * self.vocab_size))
    
    # def encoder_embedding_from_id(self, x, attention_mask=None, use_vocoder=False):
        
    #     inputs = self.vocoder(x) if use_vocoder else x
        
    #     #inspect signature of the forward method of the encoder_embedding
    #     #for SpeechT5SpeechEncoderPrenet for example (size of attention mask changes with convolutions)
    #     if "attention_mask" in inspect.signature(self.encoder_embedding.forward).parameters.keys():
    #         embeds = self.encoder_embedding(inputs, attention_mask)
    #     else:
    #         embeds = self.encoder_embedding(inputs)
            
    #     return embeds
    
    def set_speaker_embeddings(self, speaker_embeddings):
        self.speaker_embeddings = speaker_embeddings
    
    def forward(self, x, **kwargs):
        
        additional_outputs = {}
        
        if isinstance(self.linear_head, SpeechT5SpeechDecoderPostnet): 
            #feat_out, postnet(feat_out), probout --> torch layers called
            #spectrum, decoder_hidden_states, prob_pause --> names of outputs
            #outputs_before_postnet,outputs_after_postnet,logits --> Alternate names for the same thing
            continous_vector, outputs_after_postnet, logits = self.linear_head(x)
            continous_vector = continous_vector
            prob = torch.nn.functional.softmax(logits, dim=-1)
            sampled_eos = logits.sigmoid().sum(dim=-1) >= 0.5 #torch.bernoulli(prob[..., -1])

            p_eos = prob[..., -1]
            p_not_eos = prob[..., 0]
            additional_outputs = {
                "sampled_eos": sampled_eos,
                "p_eos": p_eos,
                "p_not_eos": p_not_eos,
                "outputs_after_postnet": outputs_after_postnet,
                "ouputs_before_postnet": continous_vector,
                "logits": logits
            }
            
        else:
            continous_vector = self.linear_head(x)
            # for now, we are not using p_eos here
            
        continous_vector = continous_vector * self.linear_head_scale

        # scores are between 0 and 1, and sum to 1 over the vocab dimension.
        discrete_output  = self.discretize(continous_vector,**kwargs)
        # discrete_output usually has id, score, logit, quantized_vector, quantization_loss
        
        output = {**discrete_output, **additional_outputs}
        # print(x)
        # breakpoint()
        return output
    
    
    def decoder_embedding_from_id(self, x, attention_mask=None, past_preds=None):
        #inspect signature of the forward method of the encoder_embedding
        #for SpeechT5SpeechEncoderPrenet for example (size of attention mask changes with convolutions)
        passed_full_hidden_states = past_preds is not None
        if passed_full_hidden_states:
            hidden_states = torch.cat([past_preds[...,-1,:], x[..., -1,:].unsqueeze(-2)], dim=1).to(x.device)
        else:
            hidden_states = x[..., -1,:].unsqueeze(-2)
        
        if "attention_mask" in inspect.signature(self.decoder_embedding.forward).parameters.keys():
            embeds = self.decoder_embedding(hidden_states, attention_mask=attention_mask, speaker_embeddings=self.speaker_embeddings)
        else:
            embeds = self.decoder_embedding(hidden_states, speaker_embeddings=self.speaker_embeddings)
        
        if passed_full_hidden_states:
            embeds = embeds[:, -1:]    
        
        return embeds
    
    def discretize(self, x, **kwargs) -> dict:

        # vector_encoder,_ = self.encoder_embedding_from_id(x, use_vocoder=True)
        vector_decoder = self.decoder_embedding_from_id(x, attention_mask = kwargs.get('attention_mask', None), past_preds = kwargs.get('past_preds', None))
        
        quantization_loss = torch.tensor(0.0).to(x)
                
        return {
            "id": x,
            "score": None,
            "logit": None,
            "vector_encoder": vector_decoder, 
            "vector_decoder": vector_decoder,
            "quantization_loss": quantization_loss
        }