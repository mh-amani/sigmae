import torch
torch.random.manual_seed(42)

from symbolic_bottleneck.models.encoder_decoder.bart import UnWrappedBart, UnWrappedMBart, config_bart, discretizer_config, config_mbart
from symbolic_bottleneck.auto_reg_wrapper import AudioInputAutoRegWrapper, AudioOutAutoRegWrapper
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5ForTextToSpeech, SpeechT5ForSpeechToText, SpeechT5Processor,SpeechT5HifiGan
from symbolic_bottleneck.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from symbolic_bottleneck.utils import instantiate_from_config
from symbolic_bottleneck.symbolic_autoencoder_wrapper import SymbolicAutoEncoderWrapper
from symbolic_bottleneck.models.encoder_decoder.bart import return_transformation_xy_outputs_from_y_inputs
from datasets import load_dataset

text2speech_discretizer_config = {
        '_target_': 'symbolic_bottleneck.modules.discrete_bottlenecks.audio_identity.AudioIdentityBottleneck',
        'config':{ 
            'dimensions': None,
            'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
            'linear_head_trainable': False, 
            'encoder_embedding': None, 'decoder_embedding': None, 
            'linear_head': None,
            }
    }
    
speech2text_discretizer_config = {
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

# Check the wrapped models
def AutoRegWrappedT5SpeechTest():
    
    text2speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    speech2text_model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    #unpacking models
    text2speech_vector_model, text2speech_encoder_embedding, text2speech_decoder_embedding, text2speech_linear_head = EncoderDecoderUnwrapper(text2speech_model).values()
    speech2text_vector_model, speech2text_encoder_embedding, speech2text_decoder_embedding, speech2text_linear_head = EncoderDecoderUnwrapper(speech2text_model).values()
    
    # loading processors    
    text2speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    speech2text_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")

    # initializing the discretizers
    text2speech_dimensions = {
        'decoder_embedding_dim': text2speech_vector_model.config.hidden_size,
        'vocab_size': text2speech_vector_model.config.num_mel_bins,
        'encoder_embedding_dim': text2speech_vector_model.config.hidden_size,
        'unembedding_dim': text2speech_vector_model.config.vocab_size,
    }
    
    
    speech2text_dimensions = {
        'decoder_embedding_dim': speech2text_model.config.hidden_size,
        'vocab_size': speech2text_model.config.vocab_size, 
        'encoder_embedding_dim': speech2text_model.config.hidden_size
    }
    
    speech2text_disc_config = {
        'dimensions': speech2text_dimensions,
        'encoder_embedding': text2speech_encoder_embedding ,
        'decoder_embedding': speech2text_decoder_embedding,
        'linear_head': speech2text_linear_head,
    }
    
    text2speech_disc_config = {
        'dimensions': text2speech_dimensions,
        'encoder_embedding': speech2text_encoder_embedding ,
        'decoder_embedding': text2speech_decoder_embedding ,
        'linear_head': text2speech_linear_head,
        'reduction_factor': text2speech_model.config.reduction_factor,
    }
    
    text2speech_discretizer_config["config"].update(text2speech_disc_config)
    speech2text_discretizer_config["config"].update(speech2text_disc_config)
    

    txt2speech_discretizer = instantiate_from_config(text2speech_discretizer_config)
    speech2text_discretizer = instantiate_from_config(speech2text_discretizer_config)
   
    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    txt2speech_discretizer.set_speaker_embeddings(speaker_embeddings)
    output_prepending_embeds_dec = text2speech_model.speecht5.decoder.prenet(torch.zeros(1, 1, text2speech_model.config.num_mel_bins), speaker_embeddings)

    autoregwrapper_tts_cfg = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 200, 'output': 200,},
            'control_token_ids': {'input_pad_token_id': text2speech_processor.tokenizer.pad_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_embeds_dec': output_prepending_embeds_dec,
            'output_pad_embed_dec': output_prepending_embeds_dec.clone(),
            'output_pad_embed_enc': output_prepending_embeds_dec.clone(),
            "vocoder": vocoder,
            }
    
    autoregwrapper_stt_cfg = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 200, 'output': 200,},
            'control_token_ids': { 'input_pad_token_id': speech2text_processor.tokenizer.pad_token_id,
                                    'output_eos_token_id': speech2text_processor.tokenizer.eos_token_id, 
                                    'output_pad_token_id': speech2text_processor.tokenizer.pad_token_id,
                                    'output_unknown_token_id': speech2text_processor.tokenizer.unk_token_id},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_ids': [speech2text_processor.tokenizer.bos_token_id],
    }
    
    
    
    # autoreg-wrapped models
    tts_autoreg_wrapped_model = AudioOutAutoRegWrapper(
        text2speech_vector_model,
        speech2text_discretizer,
        txt2speech_discretizer,
        autoregwrapper_tts_cfg
    )
    
    stt_autoreg_wrapped_model = AudioInputAutoRegWrapper(
        speech2text_vector_model,
        txt2speech_discretizer,
        speech2text_discretizer,
        autoregwrapper_stt_cfg
    )
    
    
    symoblic_auto_encoder = SymbolicAutoEncoderWrapper(
        stt_autoreg_wrapped_model,
        tts_autoreg_wrapped_model,
        config = None
    )
    
    symoblic_auto_encoder.transform_xy_outputs_to_y_inputs = return_transformation_xy_outputs_from_y_inputs()
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = speech2text_processor(audio = ds[0]["audio"]["array"], return_tensors="pt")
    
    results = symoblic_auto_encoder(
        x_ids=inputs["input_values"],
        x_attention_mask=inputs["attention_mask"],
        teacher_force_z=False
    )
    
    speech=results["id_z"]
    text = results["id_y"]
    print("genenerated text: \n", speech2text_processor.batch_decode(text))
    print("ground truth text: \n", ds[0]["text"])
    
    spectrogram = text2speech_model.generate_speech(results["id_y"], speaker_embeddings)
    speech_2 = vocoder(spectrogram)
    
    speech_3 = tts_autoreg_wrapped_model(input_ids = results["id_y"])["id"]
    
    import soundfile as sf
    sf.write("tts_example.wav", speech.squeeze(0).detach().numpy(), samplerate=16000)
    sf.write("tts_example2.wav", speech_2.squeeze(0).detach().numpy(), samplerate=16000)
    sf.write("tts_example3.wav", speech_3.squeeze(0).detach().numpy(), samplerate=16000)
    breakpoint()
    # # an example input and output sequences
    # en_batch = ['Everything that is lost that is lost.', 'we must imagine Sisyphe happy.']
    # input_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    # input_ids_en = input_en['input_ids']
    # output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, input_attention_mask=None, input_embeds_enc=None,
    #                                             teacher_force_output=False)
    # print('--'*20)
    # print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    # print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # # another example, starting from half of the output instead of the prepending embeddings
    # fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    # output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:5]
    # output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    # output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
    #                                             teacher_force_output=False)
    # print('--'*20)
    # print('auto-regressive forward pass - starting from half of the output')
    # print('decoded input:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    # print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))
    # # output logits stats
    # print('logits mean and std:', output_en_fr['logit'].mean(), output_en_fr['logit'].std())

    # # another example, teacher forcing the output
    # fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    # output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:]
    # output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    # output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
    #                                             teacher_force_output=True)
    # print('--'*20)
    # print('teacher forced forward pass - teacher forcing the output')
    # print('decoded input to decoder:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    # print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # # another example, French to English translation without teacher forcing
    # prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    # auto_reg_wrapper_config['output_prepending_ids'] = prefix_ids_en
    # fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    # input_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']
    # output_fr_en = fren_autoreg_wrapped_model(input_ids=input_ids_fr, teacher_force_output=False)
    # print('--'*20)
    # print('auto-regressive forward pass - French to English translation')
    # print('decoded output:', tokenizer.batch_decode(output_fr_en['id'], skip_special_tokens=False))


def main():
    AutoRegWrappedT5SpeechTest()

if __name__ == "__main__":
    main()