from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapperFromConfig

# tokenizer.pad_token_id = 0, is appended to inputs to the decode of T5
from transformers import AutoTokenizer


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


tokenizer_t5 = AutoTokenizer.from_pretrained("google-t5/t5-small")

config_t5 = {
    '_target_': 'transformers.T5ForConditionalGeneration',
    'init_attribute': 'from_pretrained',
    'pretrained_model_name_or_path': "google-t5/t5-small", 
    'output_hidden_states': "True",
}

autoreg_wrapper_config_t5 = {'device': 'cpu',
    'use_past_key_values': False, 'use_last_step_states': False,
    'max_lengths': {'input': 30, 'output': 30,},
    'control_token_ids': { 'input_pad_token_id': tokenizer_t5.pad_token_id,
                            'output_eos_token_id': tokenizer_t5.eos_token_id, 
                            'output_pad_token_id': tokenizer_t5.pad_token_id,
                            'output_unknown_token_id': tokenizer_t5.unk_token_id,},
    'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
    'output_prepending_ids': [tokenizer_t5.pad_token_id,], # 0
    'hidden_state_key': 'decoder_hidden_states',
    }

def UnWrappedT5(config=config_t5, discretizer_config=discretizer_config, tokenizer=tokenizer_t5):
    
    model, vector_model, discretizer, _ = EncoderDecoderUnwrapperFromConfig(config, discretizer_config, discretizer_config).values()
    # for t5 you need to scale the output weights
    discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)
    
    return model, vector_model, discretizer, tokenizer



##############################################################


tokenizer_mt5 = AutoTokenizer.from_pretrained("google/mt5-small")

config_mt5 = {
    '_target_': 'transformers.MT5ForConditionalGeneration',
    'init_attribute': 'from_pretrained',
    'pretrained_model_name_or_path': "google/mt5-small", 
    'output_hidden_states': "True",
}

autoreg_wrapper_config_mt5 = {'device': 'cpu',
    'use_past_key_values': False, 'use_last_step_states': False,
    'max_lengths': {'input': 30, 'output': 30,},
    'control_token_ids': { 'input_pad_token_id': tokenizer_mt5.pad_token_id,
                            'output_eos_token_id': tokenizer_mt5.eos_token_id, 
                            'output_pad_token_id': tokenizer_mt5.pad_token_id,
                            'output_unknown_token_id': tokenizer_mt5.unk_token_id,},
    'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
    'output_prepending_ids': [tokenizer_mt5.pad_token_id,], # 0
    'hidden_state_key': 'decoder_hidden_states',
    }


def UnWrappedMT5(config=config_mt5, discretizer_config=discretizer_config):
    return UnWrappedT5(config, discretizer_config, tokenizer_mt5)

