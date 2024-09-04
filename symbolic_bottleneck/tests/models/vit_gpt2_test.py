import torch
torch.random.manual_seed(42)

from symbolic_bottleneck.models.vision_transformers.vit_gpt2 import UnWrappedVITGPT2, discretizer_dec_config, disctetizer_enc_config, config_vit_gpt2
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper

from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image
import requests

# Check the unwrapped models
def UnwrappedViTGPT2Test():

    model, vector_model, encoder_embedding, decoder_embedding, linear_head = UnWrappedVITGPT2(config_vit_gpt2, disctetizer_enc_config, discretizer_dec_config )
    model.eval()
    vector_model.eval()
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    text = "a photo of a cat"
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    decoder_input_ids = tokenizer(text, return_tensors="pt")['input_ids']

    # input_vector_embeddings = encoder_embedding(pixel_values)
    output_vector_embeddings = decoder_embedding(decoder_input_ids)
     
    output_model = model(pixel_values=pixel_values, decoder_inputs_embeds=output_vector_embeddings, 
                         return_dict=True, output_hidden_states=True)

    output_vector_model = vector_model.forward(pixel_values=pixel_values, decoder_inputs_embeds=output_vector_embeddings,
                                            return_dict=True, output_hidden_states=True)
    discretized_output = torch.nn.functional.softmax(linear_head(output_vector_model["decoder_hidden_states"][-1]), dim=-1)

    print("input pixels:", pixel_values)
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
def AutoRegWrappedBartTest():
    pass
    # # an example for the encoder-decoder MBART model:
    # # get the models and the discretizers
    # model, vector_model, encoder_embedding, decoder_embedding, linear_head = UnWrappedVITGPT2(config_vit_gpt2, disctetizer_enc_config, discretizer_dec_config )
    # model.eval()
    # vector_model.eval()
    
    # from transformers import MBart50TokenizerFast
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # if image.mode != "RGB":
    #     image = image.convert(mode="RGB")
    
    # text = "a photo of a cat"
    
    # # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    # prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
    # prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)

    # auto_reg_wrapper_config = {'device': 'cpu',
    #         'use_past_key_values': False, 'use_last_step_states': False,
    #         'max_lengths': {'input': 45, 'output': 45,},
    #         'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
    #                                 'output_eos_token_id': tokenizer.eos_token_id, 
    #                                 'output_pad_token_id': tokenizer.pad_token_id,
    #                                 'output_unknown_token_id': tokenizer.unk_token_id,},
    #         'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
    #         'output_prepending_ids': [2, 250008]
    #         }
    
    # # autoreg-wrapped models
    # enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, discretizer, discretizer, auto_reg_wrapper_config)
    # auto_reg_wrapper_config['output_prepending_ids'] = [2, 250004]
    # fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, discretizer, discretizer, auto_reg_wrapper_config)

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

def AutoRegWrappedMBartTest():
    pass


def main():
    UnwrappedViTGPT2Test()
    AutoRegWrappedBartTest()

if __name__ == "__main__":
    main()