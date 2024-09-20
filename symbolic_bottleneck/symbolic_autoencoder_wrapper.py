from typing import Any, Dict
import torch
from torch.nn import Module

from symbolic_bottleneck.auto_reg_wrapper import AutoRegWrapper


class SymbolicAutoEncoderWrapper(Module):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    def __init__(
        self,
        model_x_to_y: Module,
        model_y_to_z: Module,
        config,
    ) -> None:

        super().__init__()
        self.config = config
        self.model_x_to_y = model_x_to_y
        self.model_y_to_z = model_y_to_z

    def forward(self, x_ids: torch.Tensor=None, x_attention_mask: torch.Tensor=None, x_embeds_enc: torch.Tensor=None,
                      y_prepending_ids: torch.Tensor=None, y_prepending_embeds_enc: torch.Tensor=None, y_prepending_embeds_dec: torch.Tensor=None,
                      z_ids: torch.Tensor=None, z_attention_mask: torch.Tensor=None, z_embeds_dec: torch.Tensor=None,
                      teacher_force_z: bool=True ,max_y_length=None, max_z_length=None, forced_z_length= None) -> Dict[str, Any]:
        """Perform a forward pass through the models x -> y -> z

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if max_y_length is None:
            max_y_length = self.model_x_to_y.max_lengths['output']
        if max_z_length is None:
            max_z_length = self.model_y_to_z.max_lengths['output']        
        xy_outputs = self.model_x_to_y(input_ids=x_ids, input_attention_mask=x_attention_mask, input_embeds_enc=x_embeds_enc,
                output_ids=y_prepending_ids, output_embeds_enc=y_prepending_embeds_enc, output_embeds_dec=y_prepending_embeds_dec,
                teacher_force_output=False, max_output_length=max_y_length)

        y_inputs = self.transform_xy_outputs_to_y_inputs(xy_outputs)
        y_inputs['vector_encoder'] = y_inputs['vector_encoder'] * y_inputs['output_attention_mask'].unsqueeze(-1) + \
            (1 - y_inputs['output_attention_mask']).detach().unsqueeze(-1) * y_inputs['vector_encoder'] # I don't even know what I'm doing here


        yz_outputs = self.model_y_to_z(
            input_embeds_enc=y_inputs['vector_encoder'], input_attention_mask=y_inputs['output_attention_mask']>0, 
            output_ids=z_ids, output_embeds_dec=z_embeds_dec, output_attention_mask=z_attention_mask,
            teacher_force_output=teacher_force_z, max_output_length=max_z_length, forced_z_length=forced_z_length)
            
        quantization_loss = xy_outputs['quantization_loss'] + yz_outputs['quantization_loss']
        
        
        return { 
                'id_y': xy_outputs['id'], 'id_z': yz_outputs['id'], 
                'score_y': xy_outputs['score'], 'score_z': yz_outputs['score'],
                'logit_y': xy_outputs['logit'], 'logit_z': yz_outputs['logit'],
                'vector_encoder': yz_outputs['vector_encoder'], 'vector_decoder': yz_outputs['vector_decoder'],
                'y_attention_mask': xy_outputs['output_attention_mask'], 'z_attention_mask': yz_outputs['output_attention_mask'],
                'quantization_loss': quantization_loss, "y_cross_attentions": xy_outputs.get('cross_attentions', None),
                "z_cross_attentions": yz_outputs.get('cross_attentions', None)
            }
                # 'xy_outputs': xy_outputs, 'yz_outputs': yz_outputs,} # remove later... aligator

    def transform_xy_outputs_to_y_inputs(self, xy_outputs: Dict[str, Any]) -> Dict[str, Any]:
        # you need to define how in a x -> y -> z model, the output of x -> y is transformed to the input of y -> z models.
        # for examples of this function, see the tests in symbolic_bottleneck/tests/modules/symbolic_autoencoder_wrapper.py
        NotImplementedError
    

