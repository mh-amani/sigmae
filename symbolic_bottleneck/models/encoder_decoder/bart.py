from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapperFromConfig
from transformers import BartConfig

# to find out what token should be prepended when generating output labels, we pass as text_target the input text
    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008

# example config for bart
config_bart = {
        '_target_': 'transformers.BartForConditionalGeneration',
        'config': BartConfig(d_model=128, encoder_layers=3, decoder_layers=3, 
                             vocab_size=23, max_position_embeddings=40,
                             encoder_attention_heads=2, decoder_attention_heads=2)
        }
    

# example config for mbart
config_mbart = {
        '_target_': 'transformers.MBartForConditionalGeneration',
        'init_attribute': 'from_pretrained',
        'pretrained_model_name_or_path': "facebook/mbart-large-50-many-to-many-mmt"
        }

discretizer_config = {
    '_target_': 'symbolic_bottleneck.modules.discrete_bottlenecks.softmax.SoftmaxDiscreteBottleneck',
    'config':{ 
        'dimensions': None,
        'quantize_vector': True, 'temperature': 1.0,
        'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
        'linear_head_trainable': False, 
        'encoder_embedding': None, 'decoder_embedding': None, 
        'linear_head': None,
        }
    }

def UnWrappedBart(config_bart, discretizer_config):
    
    model, vector_model, discretizer, _ = EncoderDecoderUnwrapperFromConfig(config_bart, discretizer_config, discretizer_config).values()
    
    return model, vector_model, discretizer


def UnWrappedMBart(config_mbart, discretizer_config):
    
    model, vector_model, discretizer, _ = EncoderDecoderUnwrapperFromConfig(config_mbart, discretizer_config, discretizer_config).values()
    
    return model, vector_model, discretizer

def transform_xy_outputs_to_y_inputs_bart_bart(xy_outputs):
    # since bart output has a eos <\s> token prepended in its output, we remove it for feeding to the next model
    return {'output_attention_mask': xy_outputs['output_attention_mask'][:, 1:],
            'quantized_vector_encoder': xy_outputs['quantized_vector_encoder'][:, 1:]}

def return_transformation_xy_outputs_from_y_inputs():
    return transform_xy_outputs_to_y_inputs_bart_bart