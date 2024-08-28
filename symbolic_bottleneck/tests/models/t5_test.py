import torch
torch.random.manual_seed(42)

from symbolic_bottleneck.models.encoder_decoder.t5 import UnWrappedT5,UnWrappedMT5, config_mt5, config_t5, discretizer_config, autoreg_wrapper_config_mt5, autoreg_wrapper_config_t5
from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper

config_t5['pretrained_model_name_or_path'] =  "google-t5/t5-small"
model, vector_model, discretizer, tokenizer = UnWrappedT5(config_t5, discretizer_config)
autoreg_wrapper_config = autoreg_wrapper_config_t5


def UnwrappedT5TestOne():       
    # for t5 you need to scale the output weights
    # discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)

    # input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    # labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

    
    input_ids = tokenizer("translate German to English: Das Haus ist wunderbar.", return_tensors="pt").input_ids
    labels = tokenizer("The house is wonderful.", return_tensors="pt").input_ids

    # the forward function automatically creates the correct decoder_input_ids
    logits = model(input_ids=input_ids, labels=labels).logits
    print(tokenizer.batch_decode(logits.argmax(dim=-1)))
    
    labels_with_pad = torch.cat((torch.ones((labels.shape[0], 1), dtype=torch.long).to(labels.device)* \
                                    tokenizer.pad_token_id, labels), dim=1)
    # using vectorized model
    input_vector_embeddings = discretizer.encoder_embedding_from_id(input_ids)
    input_attention_mask = torch.ones_like(input_ids)
    output_vector_embeddings = discretizer.decoder_embedding_from_id(labels_with_pad)
    output_attention_mask = torch.ones_like(labels_with_pad)
    # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels)
    # output_attention_mask = torch.ones_like(labels)
    output_vector_model = vector_model(inputs_embeds=input_vector_embeddings, attention_mask=input_attention_mask,
                                        decoder_inputs_embeds=output_vector_embeddings, decoder_attention_mask=output_attention_mask
                                        ,use_cache= None, return_dict=True, output_hidden_states=True)
    discretized_output = discretizer(output_vector_model['decoder_hidden_states'][-1])
    print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))



def UnwrappedT5TestTwo():    
    
    model_enfr = model
    model_fren = model
    vector_model_enfr = vector_model
    vector_model_fren = vector_model
    en_discretizer = discretizer
    fr_discretizer = discretizer

    # en_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)
    # fr_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)

    en_batch = ["Everything not saved will be lost.", "one must imagine Sisyphus happy."]
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "il faut imaginer Sisyphe heureux."]
    print('en_batch:', en_batch)
    print('fr_batch:', fr_batch)
    enfr_input_batch = ['translate English to French: ' + x for x in en_batch]
    fren_input_batch = ['translate French to English: ' + x for x in fr_batch]

    enfr_input_tokenized = tokenizer(text=enfr_input_batch, return_tensors="pt", padding=True)
    fren_input_tokenized = tokenizer(text=fren_input_batch, return_tensors="pt", padding=True)
    fr_tokenized = tokenizer(text=fr_batch, return_tensors="pt", padding=True)
    en_tokenized = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    
    prefix_ids = torch.tensor(tokenizer.pad_token_id).unsqueeze(0) 
    enfr_decoder_tokenized = torch.cat((prefix_ids.repeat(fr_tokenized['input_ids'].shape[0], 1), fr_tokenized['input_ids'][:, 1:]), axis=1)
    fren_decoder_tokenized = torch.cat((prefix_ids.repeat(en_tokenized['input_ids'].shape[0], 1), en_tokenized['input_ids'][:, 1:]), axis=1)

    # print('en_tokenized:', en_tokenized)
    # print('fr_tokenized:', fr_tokenized)
    # print('labels_enfr:', enfr_decoder_tokenized)
    # print('labels_fren:', fren_decoder_tokenized)
    
    # original model

    output_model_enfr = model_enfr(**enfr_input_tokenized, decoder_input_ids=enfr_decoder_tokenized, return_dict=True, output_hidden_states=True)
    output_model_fren = model_fren(**fren_input_tokenized, decoder_input_ids=fren_decoder_tokenized, return_dict=True, output_hidden_states=True)
    print('decoded output original model en fr:', tokenizer.batch_decode(output_model_enfr.logits.argmax(dim=-1), skip_special_tokens=False))
    print('decoded output original model fr en:', tokenizer.batch_decode(output_model_fren.logits.argmax(dim=-1), skip_special_tokens=False))

    # unwrapped model
    # forward pass of one model
    input_en_vector_embeddings = en_discretizer.encoder_embedding_from_id(enfr_input_tokenized['input_ids'])
    output_fr_vector_embeddings = fr_discretizer.decoder_embedding_from_id(enfr_decoder_tokenized)
    input_fr_vector_embeddings = fr_discretizer.encoder_embedding_from_id(fren_input_tokenized['input_ids'])
    output_en_vector_embeddings = en_discretizer.decoder_embedding_from_id(fren_decoder_tokenized)
    
    output_vector_model_enfr = vector_model_enfr.forward(inputs_embeds=input_en_vector_embeddings, decoder_inputs_embeds=output_fr_vector_embeddings,
                                            attention_mask=enfr_input_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    output_vector_model_fren = vector_model_fren.forward(inputs_embeds=input_fr_vector_embeddings, decoder_inputs_embeds=output_en_vector_embeddings,
                                            attention_mask=fren_input_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    discretized_output_enfr = fr_discretizer(output_vector_model_enfr['decoder_hidden_states'][-1])
    discretized_output_fren = en_discretizer(output_vector_model_fren['decoder_hidden_states'][-1])

    # print the output of the discretizer discretized_output['id'], decoded with the tokenizer
    print('decoded output decomposed model en fr:', tokenizer.batch_decode(discretized_output_enfr['id'], skip_special_tokens=False))
    print('decoded output decomposed model fr en:', tokenizer.batch_decode(discretized_output_fren['id'], skip_special_tokens=False))

    # check the logits being the same:
    print('logits are the same:', torch.allclose(discretized_output_enfr['logit'], output_model_enfr.logits, atol=1e-1))
    print('logits are the same:', torch.allclose(discretized_output_fren['logit'], output_model_fren.logits, atol=1e-1))
    

def AutoRegWrappedT5Test():
    
    # an example for the encoder-decoder pretrained T5 model:
    
    input_ids = tokenizer("translate English to French: Well, to continue being alive is also an art.", return_tensors="pt").input_ids
    output_ids = tokenizer("Alors, continuer à vivre est aussi un art.", return_tensors="pt").input_ids
    # the forward function automatically creates the correct decoder_input_ids
    logits = model(input_ids=input_ids, labels=output_ids).logits
    print(tokenizer.batch_decode(logits.argmax(dim=-1)))
    
    autoreg_wrapped_t5 = AutoRegWrapper(vector_model, discretizer, discretizer, autoreg_wrapper_config)

    print('--'*20)
    print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    output = autoreg_wrapped_t5(input_ids=input_ids, input_attention_mask=None, input_embeds_enc=None,
                                                teacher_force_output=False)
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))

    print('--'*20)
    print('teacher forced forward pass - teacher forcing the output')
    output = autoreg_wrapped_t5(input_ids=input_ids, output_ids=output_ids, teacher_force_output=True)
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))

    print('--'*20)
    print('auto-regressive forward pass - starting from half of the output')
    t = 6
    output_ids = torch.cat((torch.ones((output_ids.shape[0], 1), dtype=torch.long).to(output_ids.device)*\
                                     tokenizer.pad_token_id, output_ids), dim=1)[:,:t]
    output = autoreg_wrapped_t5(input_ids=input_ids, output_ids=output_ids, teacher_force_output=False)
    print('decoded input:', tokenizer.batch_decode(output_ids[:, :t], skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))


    # an example with vector model directly to double check.
    # output_ids_with_pad = torch.cat((torch.ones((output_ids.shape[0], 1), dtype=torch.long).to(output_ids.device)*\
    #                                  tokenizer.pad_token_id, output_ids), dim=1)
    # # using vectorized model
    # input_vector_embeddings = input_discretizer.encoder_embedding_from_id(input_ids)
    # input_attention_mask = torch.ones_like(input_ids)
    # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(output_ids_with_pad)
    # output_attention_mask = torch.ones_like(output_ids_with_pad)
    # # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels)
    # # output_attention_mask = torch.ones_like(labels)
    # output_vector_model = vector_model(inputs_embeds=input_vector_embeddings, attention_mask=input_attention_mask,
    #                                    decoder_inputs_embeds=output_vector_embeddings, decoder_attention_mask=output_attention_mask
    #                                    ,use_cache= None,)
    # discretized_output = output_discretizer(output_vector_model['last_hidden_state'])
    # print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))




def main():
    UnwrappedT5TestOne()
    UnwrappedT5TestTwo()
    AutoRegWrappedT5Test()

if __name__ == "__main__":
    main()
