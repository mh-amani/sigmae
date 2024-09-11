import torch
torch.random.manual_seed(42)

from symbolic_bottleneck.models.audio_encoder_decoder.wav2vec_gpt import UnWrappedWav2Vec, config_wav2vec_gpt, discretizer_dec_config
from symbolic_bottleneck.auto_reg_wrapper import AudioInputAutoRegWrapper,OutContAutoRegWrapper
from symbolic_bottleneck.utils import instantiate_from_config
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import requests
from datasets import load_dataset

# Check the unwrapped models
def UnwrappedAudioEncDecTest():

    model, vector_model, encoder_embedding, decoder_embedding, linear_head = UnWrappedWav2Vec(config_wav2vec_gpt, discretizer_dec_config )
    model.eval()
    vector_model.eval()
    
    audio_processor = processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
    tokenizer = audio_processor.tokenizer
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
    decoder_input_ids = tokenizer(ds[0]["text"], return_tensors="pt")["input_ids"]
    output_vector_embeddings = decoder_embedding(decoder_input_ids)
     
    output_model = model(input_values=input_values, decoder_inputs_embeds=output_vector_embeddings, 
                         return_dict=True, output_hidden_states=True)

    output_vector_model = vector_model.forward(input_values=input_values, decoder_inputs_embeds=output_vector_embeddings,
                                            return_dict=True, output_hidden_states=True)
    discretized_output = torch.nn.functional.softmax(linear_head(output_vector_model["decoder_hidden_states"][-1]), dim=-1)

    print("input pixels:", input_values)
    print("output token batch:", decoder_input_ids)
    print("decoded output original model:", output_model.logits.argmax(dim=-1))
    print("decoded output decomposed model:", discretized_output.argmax(dim=-1))
    print("text output original model:", tokenizer.batch_decode(output_model.logits.argmax(dim=-1), skip_special_tokens=False))
    print("text output decomposed model:", tokenizer.batch_decode(discretized_output.argmax(dim=-1), skip_special_tokens=False))
    print("logits mean and std:", output_model.logits.mean(), output_model.logits.std())
    print("logits are the same:", torch.allclose(discretized_output, output_model.logits, atol=1e-1))


def UnwrappedMBartTest():
    pass



# Check the wrapped models
def AutoRegWrappedEncDecTest():
    from copy import deepcopy
    audio2txt_discretizer_config = deepcopy(discretizer_dec_config)
    txt2_im_discretizer_config = {
        '_target_': 'symbolic_bottleneck.modules.discrete_bottlenecks.identity.IdentityBottleneck',
        'config':{ 
            'dimensions': None,
            'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
            'linear_head_trainable': False, 
            'encoder_embedding': None, 'decoder_embedding': None, 
            'linear_head': None,
            }
    }
    audio2txt_model, audio2txt_vector_model, _, audio2txt_decoder_embedding, audio2txt_linear_head = UnWrappedWav2Vec(config_wav2vec_gpt, discretizer_dec_config )

    audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
    audio2txt_tokenizer = audio_processor.tokenizer

    from transformers import BartModel
    from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
    
    txt2audio_model = BartModel.from_pretrained("facebook/bart-base")
    txt2audio_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import UnWrapEmbeddings, UnWrapVectorModel
    txt2audio_encoder_embedding = UnWrapEmbeddings( txt2audio_model.get_encoder())
    txt2audio_decoder_embedding = UnWrapEmbeddings(txt2audio_model.get_decoder())
    txt2audio_vector_model = UnWrapVectorModel(txt2audio_model)
    
    
    audio2txt_decoder_start_token_id = audio2txt_model.generation_config.decoder_start_token_id
    
    audio2txt_vocab_size = audio2txt_vector_model.config.decoder.vocab_size
    audio2txt_embed_dim = audio2txt_vector_model.config.decoder.d_model
    
    txt2audio_unembedding_dim = 1 #size of a image patch
    txt2audio_embed_dim = txt2audio_vector_model.config.d_model
    txt2audio_hidden_size = txt2audio_vector_model.config.hidden_size
    
    txt2audio_dimensions = {'decoder_embedding_dim': txt2audio_embed_dim, "vocab_size": txt2audio_hidden_size,
                    'encoder_embedding_dim': audio2txt_embed_dim, 'unembedding_dim': txt2audio_unembedding_dim}
    
    audio2txt_dimensions = {'decoder_embedding_dim': audio2txt_embed_dim, 'vocab_size': audio2txt_vocab_size, 
                    'encoder_embedding_dim': txt2audio_embed_dim }
    
    output_prepending_embeds_dec = txt2audio_decoder_embedding(torch.tensor([txt2audio_tokenizer.bos_token_id]))
    output_prepending_embeds_enc = audio2txt_decoder_embedding(torch.tensor([audio2txt_decoder_start_token_id]))

    txt2audio_disc_config = {'dimensions': txt2audio_dimensions, 'encoder_embedding': None,
                          'decoder_embedding': None, 'linear_head': None}
    
    audio2txt_disc_config = {'dimensions': audio2txt_dimensions, 'encoder_embedding': None,
                     'decoder_embedding': audio2txt_decoder_embedding, 'linear_head': audio2txt_linear_head}
    
    
    
    audio2txt_discretizer_config["config"].update(audio2txt_disc_config)
    txt2_im_discretizer_config["config"].update(txt2audio_disc_config)
    audio2txt_discretizer = instantiate_from_config(audio2txt_discretizer_config)
    txt2audio_discretizer = instantiate_from_config(txt2_im_discretizer_config)
    
    
    audio2txt_auto_reg_wrapper_config = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 45, 'output': 45,},
            'control_token_ids': { 'input_pad_token_id': audio2txt_tokenizer.pad_token_id,
                                    'output_eos_token_id': audio2txt_tokenizer.eos_token_id, 
                                    'output_pad_token_id': audio2txt_tokenizer.pad_token_id,
                                    'output_unknown_token_id': audio2txt_tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_ids': [audio2txt_decoder_start_token_id]
        }
    
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    
    
    txt2audio_auto_reg_wrapper_config = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 45, 'output': 17,},
            'control_token_ids': { 'input_pad_token_id': audio2txt_tokenizer.pad_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            "output_prepending_embeds_enc": output_prepending_embeds_enc, "output_prepending_embeds_dec": output_prepending_embeds_dec
    }
                                      
    
    # # autoreg-wrapped models
    ##################################### !!!!!!!!!!!!!!!!!!!!!!!  CHANGE ONE OF THE DISCRETIZER #####################################
    audio2text_autoreg_wrapped_model = AudioInputAutoRegWrapper(
        audio2txt_vector_model,
        txt2audio_discretizer,
        audio2txt_discretizer,
        audio2txt_auto_reg_wrapper_config
    )
    
    txt2audio_autoreg_wrapped_model = OutContAutoRegWrapper(
        txt2audio_vector_model,
        audio2txt_discretizer,
        txt2audio_discretizer,
        txt2audio_auto_reg_wrapper_config
    )
    
    input_values = audio_processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
    output_audio2txt = audio2text_autoreg_wrapped_model(
        input_embeds_enc=input_values,
        teacher_force_output=False
    )
    
    print('--'*20)
    print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    print('decoded output:', audio2txt_tokenizer.batch_decode(output_audio2txt['id'], skip_special_tokens=False))

    input_txt = 'two dogs are sitting in the grass with flowers'
    input_ids_txt = audio2txt_tokenizer(input_txt, return_tensors="pt")['input_ids']
    
    output_txt2audio = txt2audio_autoreg_wrapped_model(input_ids=input_ids_txt, teacher_force_output=False)
    print('--'*20)
    print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    print("decoded audio", output_txt2audio["id"])
    print("shape of the audio", output_txt2audio["id"].shape)
    
def main():
    UnwrappedAudioEncDecTest()
    AutoRegWrappedEncDecTest()

if __name__ == "__main__":
    main()