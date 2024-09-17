from transformers import SpeechT5Model, SpeechT5PreTrainedModel, SpeechT5Config
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet, SpeechT5DecoderWithSpeechPrenet
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput

class SpeechT5ModelWrapper(SpeechT5PreTrainedModel):
    
    def __init__(
        self,
        model: SpeechT5Model,
    ):
        super().__init__(model.config)
        self.encoder = model.encoder
        self.decoder = model.decoder
        
    def forward(
            self,
            input_values: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_values: Optional[torch.Tensor] = None,
            decoder_inputs_embeds: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            speaker_embeddings: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            
            assert (input_values is None and inputs_embeds is not None) or \
                (inputs_embeds is None and input_values is not None), \
                    "You have to specify either input_values or inputs_embeds" 
            assert (inputs_embeds is None) or (inputs_embeds is not None and attention_mask is not None), \
                "You have to specify attention_mask if inputs_embeds are not None"
            
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            #case input_values is not None
            else:
                hidden_states, attention_mask = self.encoder.prenet(input_values, attention_mask)
            
            encoder_outputs = self.encoder.wrapped_encoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # downsample encoder attention mask (only for encoders with speech input)
        if attention_mask is not None and isinstance(self.encoder, SpeechT5EncoderWithSpeechPrenet):
            encoder_attention_mask = self.encoder.prenet._get_feature_vector_attention_mask(
                encoder_outputs[0].shape[1], attention_mask
            )
        else:
            encoder_attention_mask = attention_mask

        if isinstance(self.decoder, SpeechT5DecoderWithSpeechPrenet):
            decoder_args = {"speaker_embeddings": speaker_embeddings}
        else:
            decoder_args = {}
            
        assert (decoder_input_values is None and decoder_inputs_embeds is not None) or \
            (decoder_inputs_embeds is None and input_values is not None), \
                "You have to specify either input_values or inputs_embeds" 
        assert (decoder_inputs_embeds is None) or \
            (decoder_inputs_embeds is not None and decoder_attention_mask is not None), \
            "You have to specify attention_mask if inputs_embeds are not None"
        
        if decoder_inputs_embeds is not None:
            decoder_hidden_states = decoder_inputs_embeds
        #case input_values is not None
        else:
            decoder_hidden_states, decoder_attention_mask = \
                self.decoder.prenet(decoder_input_values, decoder_attention_mask)
     
        decoder_outputs = self.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )