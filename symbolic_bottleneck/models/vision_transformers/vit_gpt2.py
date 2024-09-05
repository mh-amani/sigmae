from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from transformers import ViTImageProcessor
from symbolic_bottleneck.utils import instantiate_from_config

# to find out what token should be prepended when generating output labels, we pass as text_target the input text
    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008

# example config for bart
config_vit_gpt2 = {
        '_target_': 'transformers.VisionEncoderDecoderModel',
        "init_attribute": "from_pretrained",
        "pretrained_model_name_or_path": "nlpconnect/vit-gpt2-image-captioning" 
    }
    
discretizer_dec_config = {
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



def UnWrappedVITGPT2(config_vit_gpt2 ,discretizer_dec_config):
    
    model = instantiate_from_config(config_vit_gpt2)
    
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
  
    return model, vector_model ,encoder_embedding, decoder_embedding, linear_head

