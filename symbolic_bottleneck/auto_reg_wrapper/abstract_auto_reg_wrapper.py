from torch.nn import Module
import inspect
from typing import Any, Dict
from abc import abstractmethod
class AbstractAutoRegWrapper(Module):
    
    REQUIRED_CONFIG_KEYS = [
        "max_lengths",
        "device",
        "use_last_step_states",
        "use_past_key_values",
        "control_token_ids",
    ]
    
    
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ):
        super().__init__()
        
        self._validate_config(config)
        
        self.config = config
        self.model = vector_model
        self.input_discretizer = input_discretizer
        self.output_discretizer = output_discretizer
        
        self.device = self.config["device"]
        self.max_lengths = self.config["max_lengths"]
        
        self.hidden_state_key = self.config.get('hidden_state_key', 'decoder_hidden_states')
        
        if self.input_discretizer is not None:
            self.input_discretizer.to(self.device)
        if self.output_discretizer is not None:
            self.output_discretizer.to(self.device)
            
        self.control_token_ids = self.config['control_token_ids']

    @classmethod
    def _validate_config(cls, config):
        def check_for_invalid_key_names(config):
            for key in config.keys():
                if "." in key:
                    raise ValueError(f"Invalid key name: {key}. You should not use '.' in key names.")
                if isinstance(config[key], dict):
                    check_for_invalid_key_names(config[key])
        
        check_for_invalid_key_names(config)
        
        # Gather all required keys from the current class and its parent classes
        required_keys = set()

        for base_class in cls.__mro__:  # cls.__mro__ gives the method resolution order (MRO)
            if hasattr(base_class, "REQUIRED_CONFIG_KEYS"):
                required_keys.update(base_class.REQUIRED_CONFIG_KEYS)

        if not required_keys:
            raise ValueError(
                "REQUIRED_CONFIG_KEYS should be defined for each AutoRegWrapper class."
            )
            
        # Validate the config
        for key in required_keys:
            nested_keys = key.split(".")
            tmp = config
            for nested_key in nested_keys:
                if nested_key not in tmp:
                    raise ValueError(f"Config is missing key: {nested_key}")
                tmp = tmp[nested_key]
                
    def fetch_args_for_fn(self, fn, exclude = [] ,**kwargs):
        input_names = inspect.signature(fn).parameters.keys()
        inputs = {}
        for name in input_names:
            if name in kwargs and name not in exclude:
                inputs[name] = kwargs[name]
        return inputs
    
    @abstractmethod
    def prepare_inputs(self, **kwargs) -> Dict[str, Any]: 
        raise NotImplementedError
    
    @abstractmethod
    def discretize_output(self, hidden_state, teacher_forced = False):
        raise NotImplementedError
    
    @abstractmethod
    def prepare_args_for_model_forward(
        self,
        input_embeds,
        input_attention_mask, 
        output_embeds,
        output_attention_mask,
    ) -> Dict[str, Any]:
        
        raise NotImplementedError
    
    def teacher_forced_model_forward(self, **kwargs):
        args_for_seq_forward = self.prepare_args_for_model_forward(**kwargs)
        
        model_outputs = self.model.forward(
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            **args_for_seq_forward
        )
        
        inputs_to_discretizer = self.fetch_args_for_fn(
            self.discretize_output,
            exclude= ["hidden_state", "teacher_forced"],
            **{**model_outputs, **kwargs}
        )
        
        outputs = self.discretize_output(model_outputs[self.hidden_state_key][-1], teacher_forced = True ,**inputs_to_discretizer)
        
        return outputs
    
    @abstractmethod
    def return_output_dict(self, outputs) -> Dict[str, Any]:
        raise NotImplementedError
    
    def forward(self, teacher_force_output: bool = False, max_output_length = None, **kwargs) -> Dict[str, Any]:
        
        if max_output_length is None:
            max_output_length = self.max_lengths['output']
        
        self.current_max_output_length = max_output_length
        
        inputs_for_forward = self.prepare_inputs(**kwargs)
        
        if "max_output_length" not in inputs_for_forward:
            inputs_for_forward["max_output_length"] = max_output_length
        
        if not teacher_force_output:
            inputs = self.fetch_args_for_fn(
                self.sequential_forward,
                exclude= [],
                **inputs_for_forward
            )
            outputs = self.sequential_forward(**inputs) 
        
        else:
            inputs = self.fetch_args_for_fn(self.teacher_forced_model_forward, **inputs_for_forward)
            outputs = self.teacher_forced_model_forward(**inputs)
        
        return self.return_output_dict(outputs)

    @abstractmethod
    def prepare_seq_forward_params(
        self,
        input_embeds = None,
        input_attention_mask = None, 
        output_embeds_enc = None,
        output_embeds_dec = None,
        output_attention_mask = None,
        max_output_length = None
    ) -> Dict[str, Any]:
        
        raise NotImplementedError

    def should_continue_forward(self, **kwargs) -> bool:
        return True
    
    def one_step_sequential_forward(
        self,
        input_embeds, 
        output_embeds,
        input_attention_mask=None,
        output_attention_mask=None,
        last_step_states={},
    ) -> Dict[str, Any]:
    
    
         # If you're using cached key values or encoder outputs or what not, you should pass them here. and you should check
        # the model source code to see if the gradients backpropagate through these values from next step or not.
        args_for_seq_forward = self.prepare_args_for_model_forward(
            input_embeds=input_embeds,
            input_attention_mask=input_attention_mask,
            output_embeds=output_embeds,
            output_attention_mask=output_attention_mask,
        )
        
        args_for_seq_forward = {**last_step_states, **args_for_seq_forward}
        if "use_cache" not in args_for_seq_forward:
            args_for_seq_forward['use_cache'] = self.config['use_last_step_states']
        
        output = self.model(output_hidden_states = True, output_attentions= True, return_dict = True , **args_for_seq_forward)

        # output_embed = output['decoder_hidden_states'][-1]
        output_embed = output[self.hidden_state_key][-1][:, -1:, :] # does the same thing as above size: (batch_size, 1, hidden_size)
        # output of the encoder to be used in generation, I don't get why the key values are needed though, only query values are useful
        # maybe using this is blocking the gradient flow, I should check this
        encoder_last_hidden_state = output.encoder_last_hidden_state
        past_key_values = output.past_key_values
        hidden_state = output.encoder_hidden_states
        encoder_attentions = output.encoder_attentions
                
        discretizer_output = self.discretize_output(output_embed, teacher_forced=False)
        
        return {
            'id': discretizer_output['id'],
            'score': discretizer_output['score'],
            'logit': discretizer_output['logit'], 
            'vector_encoder': discretizer_output['vector_encoder'],
            'vector_decoder': discretizer_output['vector_decoder'], 
            'quantization_loss': discretizer_output['quantization_loss'],
            'past_key_values': past_key_values,
            'encoder_last_hidden_state': encoder_last_hidden_state, 
            'hidden_state': hidden_state,
            'encoder_attentions': encoder_attentions
        }
    
    @abstractmethod
    def prepare_one_step_seq_forward_params(self, step, preprend_length, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def post_one_step_seq_forward(self, current_output):
        raise NotImplementedError
    
    @abstractmethod
    def return_seq_forward_output_dict(self, step, preprend_length, **kwargs):
        raise NotImplementedError
        
    def sequential_forward(
        self,
        input_embeds,
        max_output_length,
        output_embeds_enc,
        output_embeds_dec,
        input_attention_mask = None, 
        output_attention_mask = None,
    ):
        
        preprend_length = output_embeds_enc.shape[1]
          
        seq_forward_params = self.prepare_seq_forward_params(
            input_embeds = input_embeds,
            input_attention_mask = input_attention_mask, 
            output_embeds_enc = output_embeds_enc,
            output_embeds_dec = output_embeds_dec,
            output_attention_mask = output_attention_mask,
            max_output_length = max_output_length,
            preprend_length = preprend_length
        )
        
        step = 0
        
        while step + preprend_length < self.current_max_output_length and self.should_continue_forward(**self.fetch_args_for_fn(self.should_continue_forward, **seq_forward_params)):
            
        
            if self.config['use_last_step_states'] and step > 0:
                last_step_states={'encoder_outputs':(current_output['encoder_last_hidden_state'], current_output['hidden_state'], 
                                                    current_output['encoder_attentions'])}
            else:
                last_step_states = {}
            # use the last hidden state of the encoder as the input to the decoder
            if self.config['use_past_key_values'] and step > 0:
                last_step_states['past_key_values'] = current_output['past_key_values'] # used to be torch.logical_not(eos_flag) for gpt2-gpt2,
            
            inputs_for_for_one_step_seq_forward = self.prepare_one_step_seq_forward_params(step, preprend_length, **seq_forward_params)
            
            current_output = \
                self.one_step_sequential_forward(last_step_states= last_step_states, **inputs_for_for_one_step_seq_forward)
                
            seq_forward_params = self.post_one_step_seq_forward(current_output, step ,**seq_forward_params)
            
            step += 1


        return_arguments = self.return_seq_forward_output_dict(step, preprend_length, **seq_forward_params)

        return return_arguments
    
