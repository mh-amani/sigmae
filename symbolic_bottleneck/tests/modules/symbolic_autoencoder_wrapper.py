import torch
torch.random.manual_seed(42)

from typing import Any, Dict


from symbolic_bottleneck.symbolic_autoencoder_wrapper import SymbolicAutoEncoderWrapper
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper


def AutoEncodedMbartTest():
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
    prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    config = {'device': 'cpu','use_past_key_values': False, 'use_last_step_states': True,
            'max_lengths': {'input': 30, 'output': 30,},
            'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                    'output_eos_token_id': tokenizer.eos_token_id, 
                                    'output_pad_token_id': tokenizer.pad_token_id,
                                    'output_unknown_token_id': tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': False,},
            }
    
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    from symbolic_bottleneck.models.encoder_decoder.bart import UnWrappedMBart, discretizer_config, config_mbart
    model, vector_model, discretizer = UnWrappedMBart(config_mbart, discretizer_config)
    model.eval()

    en_discretizer = discretizer
    fr_discretizer = discretizer
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, config|{'output_prepending_ids': [2, 250008]})
    fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, config|{'output_prepending_ids': [2, 250004]})
    
    def transform_xy_outputs_to_y_inputs(xy_outputs: Dict[str, Any]) -> Dict[str, Any]:
        # since bart output has a eos <\s> token prepended in its output, we remove it for feeding to the next model
        return {'output_attention_mask': xy_outputs['output_attention_mask'][:, 1:],
                'quantized_vector_encoder': xy_outputs['quantized_vector_encoder'][:, 1:]}
    
    en_fr_en_connected_models = SymbolicAutoEncoderWrapper(enfr_autoreg_wrapped_model, fren_autoreg_wrapped_model, config=None)
    en_fr_en_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs
    fr_en_fr_connected_models = SymbolicAutoEncoderWrapper(fren_autoreg_wrapped_model, enfr_autoreg_wrapped_model, config=None)
    fr_en_fr_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs
    
    # Trying different inputs and check the outputs; making sure the model is working as expected
    
    # an example input and output sequences
    sequence_en_1 = "Everything not saved will be lost."
    sequence_en_2 = "one must imagine Sisyphus happy."
    sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu."
    sequence_fr_2 = "il faut imaginer Sisyphe heureux."
    en_batch = [sequence_en_1, sequence_en_2]
    fr_batch = [sequence_fr_1, sequence_fr_2]
    input_ids_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)['input_ids']
    input_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']
    output_ids_en = torch.cat((prefix_ids_en.repeat(input_ids_en.shape[0], 1), input_ids_en[:, 1:]), dim=1)
    output_ids_fr = torch.cat((prefix_ids_fr.repeat(input_ids_fr.shape[0], 1), input_ids_fr[:, 1:]), dim=1)

    print('en-fr-en output, no teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=None, teacher_force_z=False)    
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, no teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=None, teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    t = 6
    print('en-fr-en output, few input tokens on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_en[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, few input tokens on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_fr[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('en-fr-en output, teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    # #comparing the output of the model with the output of the out of the box HF model, making sure the outputs are the same
    # en_batch_hf = ['Everything that is lost that is lost.', 'we must imagine Sisyphe happy.']
    # hfmodel_input_en_ids = tokenizer(en_batch_hf, return_tensors="pt", padding=True)['input_ids']
    # model = unwrapped_model['model']
    # output_fren_hf_model = model(input_ids=hfmodel_input_en_ids, decoder_input_ids=output_ids_fr) # attention_mask=model_input_fr_ids.ne(tokenizer.pad_token_id),
    # print('hf model en-fr teacherforcing on fr:')
    # print('y input:', tokenizer.batch_decode(hfmodel_input_en_ids, skip_special_tokens=False))
    # print('z output:', tokenizer.batch_decode(output_fren_hf_model['logits'].argmax(dim=-1), skip_special_tokens=False))
    # print('-'*50)
    
    # fr_batch_hf = ["Tout ce qui n'est pas sauvé sera perdu.", "on doit imaginer Sisyphus heureux."]
    # hfmodel_input_fr_ids = tokenizer(text_target=fr_batch_hf, return_tensors="pt", padding=True)['input_ids']
    # model = unwrapped_model['model']
    # output_enfr_hf_model = model(input_ids=hfmodel_input_fr_ids, decoder_input_ids=output_ids_en, output_attentions=True)
    # # attention_mask=model_input_en_ids.ne(tokenizer.pad_token_id),
    # print('hf model fr-en teacherforcing on en:')
    # print('y input:', tokenizer.batch_decode(hfmodel_input_fr_ids, skip_special_tokens=False))
    # print('z output:', tokenizer.batch_decode(output_enfr_hf_model['logits'].argmax(dim=-1), skip_special_tokens=False))
    # print('-'*50)


def AutoEncodedT5Test():
    from symbolic_bottleneck.models.encoder_decoder.t5 import UnWrappedT5, config_t5, discretizer_config, autoreg_wrapper_config, tokenizer
    from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper

    model, vector_model, discretizer = UnWrappedT5(config_t5, discretizer_config)
    model.eval()

    en_discretizer = discretizer
    fr_discretizer = discretizer

    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, autoreg_wrapper_config)
    fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, autoreg_wrapper_config)
    
    def prepending_attention_and_embeds_to_output_ids(prepending_sentence):
        prepending_tokens = tokenizer(prepending_sentence, return_tensors="pt")['input_ids'][..., :-1]
        
        def transform_xy_outputs_to_y_inputs(xy_outputs: Dict[str, Any]) -> Dict[str, Any]:
            # since T5 output has a eos <\s> token prepended in its output, we remove it for feeding to the next model
            bsize = xy_outputs['output_attention_mask'].shape[0]
            prepending_embedding = discretizer.encoder_embedding_from_id(prepending_tokens).repeat(bsize, 1, 1)
            prepend_attention_mask = torch.ones_like(prepending_tokens).repeat(bsize, 1)
            return {'output_attention_mask': torch.cat((prepend_attention_mask, xy_outputs['output_attention_mask'][..., 1:],), dim=1),
                    'quantized_vector_encoder': torch.cat((prepending_embedding, xy_outputs['quantized_vector_encoder'][..., 1:, :],), dim=1)}
        
        return transform_xy_outputs_to_y_inputs
    
    transform_xy_outputs_to_y_inputs_fr_en_fr = prepending_attention_and_embeds_to_output_ids("translate English to French: ")
    transform_xy_outputs_to_y_inputs_en_fr_en = prepending_attention_and_embeds_to_output_ids("translate French to English: ")
        
    fr_en_fr_connected_models = SymbolicAutoEncoderWrapper(fren_autoreg_wrapped_model, enfr_autoreg_wrapped_model, config=None)
    fr_en_fr_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs_fr_en_fr
        
    en_fr_en_connected_models = SymbolicAutoEncoderWrapper(enfr_autoreg_wrapped_model, fren_autoreg_wrapped_model, config=None)
    en_fr_en_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs_en_fr_en
    
    # Trying different inputs and check the outputs; making sure the model is working as expected

    # an example input and output sequences
    en_batch = ["Everything not saved will be lost.", "one must imagine Sisyphus happy."]
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "il faut imaginer Sisyphe heureux."]
    print('en_batch:', en_batch)
    print('fr_batch:', fr_batch)
    input_en = ['translate English to French: ' + x for x in en_batch]
    input_fr = ['translate French to English: ' + x for x in fr_batch]
    input_ids_en = tokenizer(text=input_en, return_tensors="pt", padding=True)['input_ids']
    input_ids_fr = tokenizer(text=input_fr, return_tensors="pt", padding=True)['input_ids']
    output_ids_en = tokenizer(en_batch, return_tensors="pt", padding=True)['input_ids']
    output_ids_fr = tokenizer(fr_batch, return_tensors="pt", padding=True)['input_ids']

    print('en-fr-en output, no teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=None, teacher_force_z=False)    
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, no teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=None, teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    t = 6
    print('en-fr-en output, few input tokens on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_en[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, few input tokens on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_fr[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('en-fr-en output, teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)



def main(): 
    AutoEncodedMbartTest()
    # AutoEncodedT5Test()


if __name__ == "__main__":
    main()