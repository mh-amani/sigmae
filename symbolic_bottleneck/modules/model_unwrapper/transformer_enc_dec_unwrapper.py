import torch 
from symbolic_bottleneck.utils import instantiate_from_config
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
from transformers import Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
from copy import deepcopy

def EncoderDecoderUnwrapperFromConfig(model_config, discretizer_enc_config, discretizer_dec_config):
    """
    Unwraps the encoder-decoder model to get the encoder and decoder weights.
    Args:   
        model_config: The configuration object for the model.
        discretizer_enc_config: The configuration object for the encoder discretizer.
        discretizer_dec_config: The configuration object for the decoder discretizer.
    Returns:
        model: The encoder-decoder model.
        vector_model: The encoder-decoder model without embedding and head, pure transfomer.
        discretizer_enc: The encoder weights.
        discretizer_dec: The decoder weights.
    """
    
    model = instantiate_from_config(model_config)
    
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()

    vocab_size = None
    embed_dim = None
    if hasattr(vector_model.config, 'vocab_size'): 
        vocab_size = vector_model.config.vocab_size
    if hasattr(vector_model.config, 'd_model'):    
        embed_dim = vector_model.config.d_model
    if isinstance(vector_model, VisionEncoderDecoderModel):
        vocab_size = vector_model.config.decoder.vocab_size
        embed_dim = vector_model.config.decoder.n_embd
    
    if vocab_size is None or embed_dim is None:
        raise ValueError("The vocab_size or embed_dim is not retrievable from the model config - You need to implement the unwrapping for this type of model")
        
    dimensions = {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 
                  'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size}
    disc_config = {'dimensions': dimensions, 'encoder_embedding': encoder_embedding,
                     'decoder_embedding': decoder_embedding, 'linear_head': linear_head}
    if discretizer_enc_config is not None:
        discretizer_enc_config['config'].update(disc_config)
    discretizer_dec_config['config'].update(disc_config) 
    discretizer_enc = instantiate_from_config(discretizer_enc_config) if discretizer_enc_config is not None else None
    discretizer_dec = instantiate_from_config(discretizer_dec_config)

    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}


def EncoderDecoderUnwrapper(enc_dec_model):
    """
    Unwraps the encoder-decoder model to get the encoder and decoder weights.
    Args:
        enc_dec_model: The encoder-decoder model.
    Returns:
        vector_model: The encoder-decoder model without embedding and head, pure transfomer.
        encoder_embedding_weight: The encoder weights.
        decoder_embedding_weight: The decoder weights.
        linearhead_weight: The linear head weights.
    """

    encoder_embedding = UnWrapEmbeddings( enc_dec_model.get_encoder())
    decoder_embedding = UnWrapEmbeddings(enc_dec_model.get_decoder())
    linear_head = UnWrapLinearHead(enc_dec_model)
    vector_model = UnWrapVectorModel(enc_dec_model)
    return {'vector_model': vector_model, 'encoder_embedding': encoder_embedding, 
        'decoder_embedding': decoder_embedding, 'linear_head': linear_head} 
    
def UnWrapVectorModel(model):
    vector_model = model.base_model
    return vector_model
    
    
def UnWrapLinearHead(model):
    linear_head = model.get_output_embeddings()
    #TODO: Is this necessary ? We should check if we can't replace this with a deepcopy or the embedding itself (a one-liner)
    if isinstance(linear_head, torch.nn.Linear):
        linear_head_weight = linear_head.weight.clone()
        linear_head = torch.nn.Linear(linear_head_weight.shape[1], linear_head_weight.shape[0])
        linear_head.weight.data = linear_head_weight
        return linear_head
    
    else:
        #puposely raise an error to make sure we don't forget to implement the unwrapping for this type of embeddings and avoid silent errors
        raise ValueError(f"The linear head of type {type(linear_head)} is not supported - You need to implement the unwrapping for this type of embeddings")
    
def UnWrapEmbeddings(model):
    def UnWrapVitEmbeddings(embeddings):
        # TODO: Is this really necessary ? We should check if we can just remove the deepcopy and it still works
        return deepcopy(embeddings)

    def UnWrapDiscreteEmbeddings(embeddings, embed_scale):
        # TODO: Is this necessary ? We should check if we can't replace this with a deepcopy or the embedding itself (a one-liner)
        embedding_weight = embeddings.weight.clone()
        embedding = torch.nn.Embedding(embedding_weight.shape[0], embedding_weight.shape[1])
        embedding.weight.data = embedding_weight
        embedding.weight.data = embed_scale * embedding.weight.data
        return embedding
    
    #TODO: handling this sperately but we should look into how to make this better in the future
    if isinstance(model, Wav2Vec2PreTrainedModel):
        embeddings = model.feature_extractor 
    else:
        #some models don't have the get_input_embeddings method, so we need to handle this case
        try:
            embeddings = model.get_input_embeddings()
        except:
            #TODO: This is crazy but has to be done. For example, for some reason MBartEncoder has the embed_tokens attribute instead of get_input_embeddings
            if hasattr(model, 'embed_tokens'):
                embeddings = model.embed_tokens
            else:
                raise ValueError(
                    f"Model of type {type(model)} does not have the 'get_input_embeddings' and its input embeddings are not called 'embed tokens' - You need to implement the unwrapping for this type of embeddings"
                )

    if isinstance(embeddings, torch.nn.Embedding):
        embed_scale = model.embed_scale if hasattr(model, 'embed_scale') else 1.0
        return UnWrapDiscreteEmbeddings(embeddings, embed_scale)
    elif isinstance(embeddings, ViTPatchEmbeddings) or isinstance(embeddings, Wav2Vec2FeatureEncoder):
        return UnWrapVitEmbeddings(embeddings)
    else:
        #puposely raise an error to make sure we don't forget to implement the unwrapping for this type of embeddings and avoid silent errors
        raise ValueError(f"input embeddings of type {type(embeddings)} are not supported - You need to implement the unwrapping for this type of embeddings")


#TODO: OLD EncoderDecoderUnwrapper. I can probably remove this
# def EncoderDecoderUnwrapper(enc_dec_model):
#     """
#     Unwraps the encoder-decoder model of a text to text model to get the encoder and decoder weights.
#     Args:
#         enc_dec_model: The encoder-decoder model.
#     Returns:
#         vector_model: The encoder-decoder model without embedding and head, pure transfomer.
#         encoder_embedding_weight: The encoder weights.
#         decoder_embedding_weight: The decoder weights.
#         linearhead_weight: The linear head weights.
#     """
#     # Get the encoder and decoder weights
#     encoder_embedding_weight = enc_dec_model.get_encoder().get_input_embeddings().weight.clone()
#     encoder_embedding = torch.nn.Embedding(encoder_embedding_weight.shape[0], encoder_embedding_weight.shape[1])
#     encoder_embedding.weight.data = encoder_embedding_weight
#     try:
#         encoder_embedding.weight.data = enc_dec_model.model.encoder.embed_scale * encoder_embedding.weight.data
#     except:
#         pass

#     decoder_embedding_weight = enc_dec_model.get_decoder().embed_tokens.weight.clone()
#     decoder_embedding = torch.nn.Embedding(decoder_embedding_weight.shape[0], decoder_embedding_weight.shape[1])
#     decoder_embedding.weight.data = decoder_embedding_weight
#     try:
#         decoder_embedding.weight.data = enc_dec_model.model.decoder.embed_scale * decoder_embedding.weight.data
#     except:
#         pass

#     try:
#         linear_head_weight = enc_dec_model.lm_head.weight.clone()
#         linear_head = torch.nn.Linear(linear_head_weight.shape[1], linear_head_weight.shape[0])
#         linear_head.weight.data = linear_head_weight
#     # linearhead_bias = enc_dec_model.lm_head.bias
#     # linearhead_final_logit_bias = enc_dec_model.final_logits_bias
#     # linear_head = enc_dec_model.get_output_embeddings()
#     except:
#         linear_head = None  

#     try:
#         vector_model = enc_dec_model.model
#     except:
#         vector_model = enc_dec_model
#     return {'vector_model': vector_model, 'encoder_embedding': encoder_embedding, 
#         'decoder_embedding': decoder_embedding, 'linear_head': linear_head} 







########################################################################################################################
# code snippets and other useful stuff for debugging and checking stuff

# model.model.encoder.embed_tokens(input_enfr['input_ids']) == input_vector_embeddings
# model.model.encoder.embed_positions(input_enfr['input_ids'])
# model.model.decoder.embed_tokens(labels_enfr['input_ids'])*32 - output_vector_embeddings

# # print the weights and shapes
    # print('-'*50)
    # print('vector model:', vector_model)  
    # print('encoder weight shape:', encoder_embedding_weight.shape)
    # print('decoder weight shape:', decoder_embedding_weight.shape)
    # print('linear head weight shape:', linearhead_weight.shape)
    # print('linear head weight:', linearhead_weight)
    # print('-'*50)


############################################################
# hard to make the forwardpass work, the lm_head is not removed properly

# def create_headless_transformer_model(base_model_class: PreTrainedModel, config):
#     """
#     Creates a modified version of a given transformer model class, without embedding and head layers.
    
#     Args:
#         base_model_class (PreTrainedModel): A class of transformer model from Hugging Face's transformers.
#         config: The configuration object for the model.

#     Returns:
#         A modified model class with no embedding and head layers.
#     """

#     class HeadlessTransformerModel(base_model_class):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             # Removing embedding layers and head might depend on the specific model architecture
#             self.model.encoder.embed_tokens = None
#             self.model.decoder.embed_tokens = None
#             self.lm_head = None
        
#         def forward(self, *args, **kwargs):
#             """
#             The forward pass will need to be adjusted depending on what the default model expects
#             and what is now missing (like embeddings and heads).
#             """
#             # Typical use-case would involve calling original model's forward method,
#             # But you must handle the lack of embeddings and heads appropriately.
#             outputs = super().forward(*args, **kwargs)
#             return outputs
    
#     # Return a new class instance with removed components
#     return HeadlessTransformerModel(config)

############################################################
# # vector_model.lm_head = lambda x: x
    # Replace the lm_head with an Identity module
    # vector_model.lm_head = torch.nn.Identity()

    # things that fail to work, especially the forward pass breaks...
    # Remove the embedding and head from the model
    # vector_model = enc_dec_model
    # vector_model.get_encoder().embed_tokens = None
    # vector_model.get_decoder().embed_tokens = None
    # vector_model.lm_head = None
    # vector_model = create_headless_transformer_model(enc_dec_model.__class__, enc_dec_model.config)

    # encoder_model = enc_dec_model.get_encoder()
    # decoder_model = enc_dec_model.get_decoder()
    # vector_model = torch.nn.ModuleDict({'encoder': encoder_model, 'decoder': decoder_model})
    # vector_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
    # vector_model.get_encoder().embed_tokens = None
    # vector_model.get_decoder().embed_tokens = None


############################################################
    # an example for the encoder-decoder MBART model:
    # Initialize the model
    # from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # # tokenizing the input and output sequences (teacher forced generation)
    # tokenizer_enfr = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
    
    # # Unwrap the model to get the encoder and decoder weights, so you can initialize discretizers
    # unwrapped_model = EncoderDecoderUnwrapper(model)
    # vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias = unwrapped_model.values()
    # # initialize the discretizer and connect the two models via the blocks module
    # discretizer_config = {'dimensions': {'decoder_embedding_dim': 1024, 'vocab_size': tokenizer_enfr.vocab_size, 'encoder_embedding_dim': 1024, 'unembedding_dim': tokenizer_enfr.vocab_size}, 
    #                             'quantize_vector': True, 'temperature': 1.0,
    #                             'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False, 
    #                             'encoder_embedding_weight': encoder_embedding_weight, 'decoder_embedding_weight': decoder_embedding_weight, 
    #                             'linear_head_weight': linearhead_weight, 'linear_head_bias': linearhead_bias}
                        
    # en_discretizer = SoftmaxDiscreteBottleneck(discretizer_config)
    # fr_discretizer = SoftmaxDiscreteBottleneck(discretizer_config)