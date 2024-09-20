from typing import Any, Dict
import torch
import numpy as np
from symbolic_bottleneck.auto_reg_wrapper import AbstractAutoRegWrapper

class AutoRegWrapper(AbstractAutoRegWrapper):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    
    REQUIRED_CONFIG_KEYS = [
        "control_token_ids.output_pad_token_id",
        "control_token_ids.input_pad_token_id",
        "control_token_ids.output_unknown_token_id",
        "control_token_ids.output_eos_token_id",
        "soft_average",
        "output_prepending_ids",
    ]
    
    NO_INPUT_ATTENTION_MASK: bool = False
    
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ) -> None:

        super().__init__(
            vector_model=vector_model,
            input_discretizer=input_discretizer,
            output_discretizer=output_discretizer,
            config=config,
        )
        
        self.soft_average = self.config['soft_average']
        
        output_prepending_ids = self.config['output_prepending_ids']
        # output_prepending_embeds_enc = self.config.get('output_prepending_embeds_enc', None)
        # output_prepending_embeds_dec = self.config.get('output_prepending_embeds_dec', None)

        if output_prepending_ids is None: #and (output_prepending_embeds_enc is None or output_prepending_embeds_dec is None):
            raise ValueError("output_prepending_ids are not provided")
        #TODO: This can probably be removed BUT IS USEFUL FOR MODELS WHO HAVE CONTINUOUS OUTPUTS FOR DECODERS
        # elif output_prepending_ids is None and output_prepending_embeds_dec is not None:
        #     self.output_prepending_ids = self.control_token_ids['pad_token_id_y'] * np.ones(output_prepending_embeds_dec.shape[:2], dtype=np.long)
        #     self.output_prepending_embeds_enc = output_prepending_embeds_enc.numpy()
        #     self.output_prepending_embeds_dec = output_prepending_embeds_dec.numpy()
        else:
            self.output_prepending_ids = output_prepending_ids
            self.output_prepending_embeds_enc = self.output_discretizer.encoder_embedding_from_id(torch.from_numpy(np.array(self.output_prepending_ids)).to(self.device)).clone().detach().cpu().numpy()
            self.output_prepending_embeds_dec = self.output_discretizer.decoder_embedding_from_id(torch.from_numpy(np.array(self.output_prepending_ids)).to(self.device)).clone().detach().cpu().numpy()

    def prepare_inputs(
        self,
        input_ids=None,
        input_attention_mask=None,
        input_embeds_enc=None,
        output_ids=None,
        output_embeds_enc=None,
        output_embeds_dec=None,
        output_attention_mask=None,
    ):
                
        assert (input_ids is not None) != (input_embeds_enc is not None), "Either input_ids or input_embeds should be provided"
        if not self.NO_INPUT_ATTENTION_MASK :
            assert (input_embeds_enc is not None and input_attention_mask is not None) or (input_embeds_enc is None and input_attention_mask is None), "input_embeds and input_attention_mask should be provided together or not at all. HINT: if you're model does not accept input_attention_mask, set the class variable NO_INPUT_ATTENTION_MASK to True"
        assert (output_ids is None)  or (output_embeds_enc is None), "Either output_ids or output_embeds or neither should be provided, but not both"
        assert (output_embeds_enc is not None and output_embeds_dec is not None and output_attention_mask is not None) or (output_embeds_enc is None and output_embeds_dec is None and output_attention_mask is None), "output_embeds and output_attention_mask should be provided together or not at all"

        if input_ids is not None:
            input_embeds_enc = self.input_discretizer.encoder_embedding_from_id(input_ids)
            input_attention_mask = torch.logical_not(torch.eq(input_ids, self.control_token_ids['input_pad_token_id']))
        
        if output_ids is not None:
            # setting the output encoder and decoder embeddings and attention mask, starting generation from these embeddings
            output_embeds_enc = self.output_discretizer.encoder_embedding_from_id(output_ids)
            output_embeds_dec = self.output_discretizer.decoder_embedding_from_id(output_ids)
            output_attention_mask = torch.logical_not(torch.eq(output_ids, self.control_token_ids['output_pad_token_id']))
        elif output_embeds_enc is not None:
            # setting the output ids to unk tokens, as we don't have acces to the starting ids
            output_ids = self.control_token_ids['output_unknown_token_id'] * torch.ones(input_embeds_enc.shape[:2], dtype=torch.long).to(input_embeds_enc.device)
        else:
            # no output starting point is provided, so we start from the prepending embeddings
            output_ids = torch.from_numpy(np.array(self.output_prepending_ids)).repeat(input_embeds_enc.shape[0], 1).to(input_embeds_enc.device)
            output_embeds_enc = torch.tensor(self.output_prepending_embeds_enc).repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_embeds_dec = torch.tensor(self.output_prepending_embeds_dec).repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_attention_mask = torch.ones(output_embeds_enc.shape[:2], dtype=torch.bool).to(output_embeds_enc.device)
        
        return {
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "input_embeds": input_embeds_enc,
            # "output_ids": output_ids,
            "output_embeds_enc": output_embeds_enc,
            "output_embeds_dec": output_embeds_dec,
            "output_attention_mask": output_attention_mask
        }
        
    def discretize_output(self, hidden_state, teacher_forced = False, output_ids=None, output_attention_mask=None):
        if teacher_forced:
            discretized_output = self.output_discretizer(
                hidden_state,
                supervision=True,
                target_ids=output_ids,
                target_attention_mask=output_attention_mask, # this is irrelavent. does nothing.
                average_probs=self.soft_average['word_embeds_with_scores_forward'] # this is irrelavent. does nothing.
            )
            
        else:
            discretized_output = self.output_discretizer(
                hidden_state,
                supervision=False,
                average_probs=self.soft_average['word_embeds_with_scores_forward']
            )
        
        return discretized_output
        
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
        **kwargs
    ):
        return {
            "inputs_embeds": input_embeds,
            "attention_mask": input_attention_mask,
            "decoder_inputs_embeds": output_embeds_dec,
            "decoder_attention_mask": output_attention_mask,            
        }
        
    def teacher_forced_model_forward(
        self,
        input_embeds,
        input_attention_mask,
        output_embeds_dec,
        output_attention_mask,
        # output_ids
    ):
         
        outputs = super().teacher_forced_model_forward(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
            # output_ids = output_ids
        )

        outputs['score_list'] = []
        outputs['eos_flag'] = torch.any(torch.eq(outputs['id'], self.control_token_ids['output_eos_token_id']), dim=1).reshape(-1, 1)
        outputs['p_not_eos'] = 1 - outputs['score'][:, :, self.control_token_ids['output_eos_token_id']]
        outputs['output_attention_mask'] = output_attention_mask
        
        return outputs    
        
        
    def return_output_dict(self, outputs) -> Dict[str, Any]:
        return {
            'id': outputs['id'],
            'score': outputs['score'],
            'score_list': outputs['score_list'],
            'logit': outputs['logit'],
            'vector_encoder': outputs['vector_encoder'],
            'vector_decoder': outputs['vector_decoder'],
            'output_attention_mask': outputs['output_attention_mask'],
            'eos_flag': outputs['eos_flag'],
            'p_not_eos': outputs['p_not_eos'],
            'quantization_loss': outputs['quantization_loss']
        }
        
    def prepare_seq_forward_params(
        self,
        input_embeds = None,
        input_attention_mask = None, 
        output_embeds_enc = None,
        output_embeds_dec = None,
        output_attention_mask = None,
        max_output_length = None,
        preprend_length = None,
    ) -> Dict[str, Any]:
        
        # initialize tensors
        quantization_loss = 0
        onehot_score_pad = torch.nn.functional.one_hot(torch.tensor(self.control_token_ids['output_pad_token_id']), num_classes=self.output_discretizer.vocab_size).to(output_embeds_enc.device).float()
        pad_embed_enc = self.output_discretizer.encoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
        pad_embed_dec = self.output_discretizer.decoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
       
        ids = torch.ones(input_embeds.shape[0], max_output_length-preprend_length).to(input_embeds).int() * self.control_token_ids['output_pad_token_id']
        scores = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, self.output_discretizer.vocab_size).to(input_embeds) * onehot_score_pad
        logits = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, self.output_discretizer.vocab_size).to(input_embeds)
        # scores = torch.empty(input_embeds.shape[0], max_output_length-preprend_length, discretizer.vocab_size).to(input_embeds).fill_(0.0)
        logit_list = []
        scores_list = [] # this is only for visualizing the gradients. cause score matrix is a clone of scores, so gradaient propogation throught time doesn't showup in it. it is visible here.
        p_not_eoss = [torch.ones(input_embeds.shape[0], 1, requires_grad=True).to(input_embeds)]
        eos_flags = torch.zeros(input_embeds.shape[0], 1, dtype=torch.bool).to(input_embeds)
        output_embeds_encs = output_embeds_enc.requires_grad_(True)
        output_embeds_decs = output_embeds_dec.requires_grad_(True)
        output_attention_masks = output_attention_mask.float().requires_grad_(True)
        
        # scores.requires_grad = True
        # p_not_eoss.requires_grad = True
        # output_embeds_encs.requires_grad = True
        # output_embeds_decs.requires_grad = True
        # output_attention_masks.requires_grad = True
        
        return {
            "quantization_loss": quantization_loss,
            "onehot_score_pad": onehot_score_pad,
            "pad_embed_enc": pad_embed_enc,
            "pad_embed_dec": pad_embed_dec,
            "ids": ids,
            "scores": scores,
            "logits": logits,
            "logit_list": logit_list,
            "scores_list": scores_list,
            "p_not_eoss": p_not_eoss,
            "eos_flags": eos_flags,
            "output_embeds_encs": output_embeds_encs,
            "output_embeds_decs": output_embeds_decs,
            "output_attention_masks": output_attention_masks,
            "input_embeds": input_embeds,
            "input_attention_mask": input_attention_mask,
            "output_attention_mask": output_attention_mask,
        }
        
    def should_continue_forward(self, eos_flags):
        return not torch.all(eos_flags)
    
    def prepare_one_step_seq_forward_params(
        self,
        step,
        preprend_length,
        input_embeds,
        input_attention_mask,
        output_embeds_decs,
        output_attention_masks,
        **kwargs
    ):
    
        return {
            "input_embeds": input_embeds,
            "input_attention_mask": input_attention_mask,
            "output_embeds_dec": output_embeds_decs[:, :step + preprend_length],
            "output_attention_mask": output_attention_masks[:, :step + preprend_length] > 0, # this is to make sure that the attention mask is binary, otherwise we had NAN in the loss! for longish sequences
        }
        
    def one_step_sequential_forward(
        self,
        input_embeds,
        input_attention_mask, 
        output_embeds_dec,
        output_attention_mask,
        last_step_states={},
    ):
        
        eos_token_id = self.control_token_ids['output_eos_token_id']
       
        seq_forward_output = super().one_step_sequential_forward(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
            last_step_states = last_step_states,
        )
               
        current_eos_flag = (torch.eq(seq_forward_output['id'][:, -1], eos_token_id))

        p_eos = seq_forward_output['score'][:, :, eos_token_id]

        return({
            'id': seq_forward_output['id'],
            'score': seq_forward_output['score'],
            'logit': seq_forward_output['logit'], 
            'vector_encoder': seq_forward_output['vector_encoder'],
            'vector_decoder': seq_forward_output['vector_decoder'], 
            'quantization_loss': seq_forward_output['quantization_loss'],
            'eos_flag': current_eos_flag,
            'p_eos': p_eos, 
            'past_key_values': seq_forward_output["past_key_values"],
            'encoder_last_hidden_state': seq_forward_output["encoder_last_hidden_state"], 
            'hidden_state': seq_forward_output["hidden_state"],
            'encoder_attentions': seq_forward_output["encoder_attentions"]
        })
        
    def post_one_step_seq_forward(
        self,
        current_output,
        step,
        ids,
        scores_list,
        scores,
        logits,
        logit_list,
        p_not_eoss,
        output_attention_masks,
        eos_flags,
        quantization_loss,
        output_embeds_encs,
        output_embeds_decs,
        **kwargs,
    ):
        
        ids[:, step] = current_output['id'].reshape(-1)
        scores_list.append(current_output['score'])
        scores[:, step] = current_output['score'][:, 0]
        logits[:, step] = current_output['logit'][:, 0]
        logit_list.append(current_output['logit'])
        p_not_eoss.append( (1 - current_output['p_eos']) * p_not_eoss[step])
        output_attention_masks = torch.cat((output_attention_masks, self.attention_mask(eos_flags, p_not_eoss[step])[:, 0].reshape(-1, 1)), dim=1)
        eos_flags = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
        quantization_loss += (current_output['quantization_loss'] * torch.logical_not(eos_flags).float())
        quantization_loss = quantization_loss * (current_output['quantization_loss'] * torch.logical_not(eos_flags))
        output_embeds_encs = torch.cat((output_embeds_encs, current_output['vector_encoder']), dim=1)
        output_embeds_decs = torch.cat((output_embeds_decs, current_output['vector_decoder']), dim=1)
        
        return {
            "ids": ids,
            "scores_list": scores_list,
            "scores": scores,
            "logits": logits,
            "logit_list": logit_list,
            "p_not_eoss": p_not_eoss,
            "output_attention_masks": output_attention_masks,
            "eos_flags": eos_flags,
            "quantization_loss": quantization_loss,
            "output_embeds_encs": output_embeds_encs,
            "output_embeds_decs": output_embeds_decs,
            **kwargs
        }
    
    def return_seq_forward_output_dict(
        self,
        step,
        preprend_length,
        ids,
        scores,
        logits,
        p_not_eoss,
        output_embeds_encs,
        output_embeds_decs,
        output_attention_masks,
        onehot_score_pad,
        pad_embed_enc,
        pad_embed_dec,
        scores_list,
        eos_flags,
        quantization_loss,
        **kwargs
    ):
        # cut the tensors to the actual length, remove 0s and 1s.
        ids = ids[:, :step]
        scores = scores[:, :step]
        logits = logits[:, :step, :]
        p_not_eoss = p_not_eoss[1:]
        output_embeds_encs = output_embeds_encs[:, :step + preprend_length]
        output_embeds_decs = output_embeds_decs[:, :step + preprend_length]
        output_attention_masks = output_attention_masks[:, :step + preprend_length]

        # enforce pad tokens and embeds where attention mask is zero
        binary_attention_mask = output_attention_masks > 0

        # pad out the random tokens after first eos token
        ids = ids * binary_attention_mask[:, -ids.shape[1]:] + self.control_token_ids['output_pad_token_id'] * \
                    torch.logical_not(binary_attention_mask)[:, -ids.shape[1]:]
        # similary, pad out scores, but with backpropagatable 0-1 scores (via p_not_eos)
        scores = scores * output_attention_masks[:, -ids.shape[1]:].unsqueeze(-1) + \
            (1 - output_attention_masks[:, -ids.shape[1]:]).unsqueeze(-1) * onehot_score_pad
        # p_not_eoss = p_not_eoss * output_attention_masks[:, -ids.shape[1]:] + \
        #     (1 - output_attention_masks[:, -ids.shape[1]:]) * p_not_eoss[:, -1].reshape(-1, 1)
        output_embeds_encs = output_embeds_encs * output_attention_masks.unsqueeze(-1) + \
            pad_embed_enc * torch.logical_not(output_attention_masks).unsqueeze(-1)
        output_embeds_decs = output_embeds_decs * output_attention_masks.unsqueeze(-1) + \
            pad_embed_dec * torch.logical_not(output_attention_masks).unsqueeze(-1)

        
        return {
            'id': ids,
            'score': scores,
            'score_list': scores_list,
            'logit': logits,
            'vector_encoder': output_embeds_encs,
            'vector_decoder': output_embeds_decs,
            'quantization_loss': quantization_loss,
            'output_attention_mask': output_attention_masks,
            'eos_flag': eos_flags,
            'p_not_eos': p_not_eoss
        }
    
    def attention_mask(self, eos_flags, p_not_eos):
        true_attention_mask = torch.logical_not(eos_flags)
        if self.config['soft_average']['p_eos_forward']:
            return p_not_eos
        elif self.config['soft_average']['p_eos_backward']:
            return true_attention_mask + (p_not_eos - p_not_eos.detach())
        else:
            return true_attention_mask
        # what about eos_flags * p_not_eos?
