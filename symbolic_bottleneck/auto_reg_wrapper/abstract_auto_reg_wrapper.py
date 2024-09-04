from torch.nn import Module
import inspect
from typing import Any, Dict

class AbstractAutoRegWrapper(Module):
    
    REQUIRED_CONFIG_KEYS = ["max_lengths", "device"]
    
    
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
        
        if self.input_discretizer is not None:
            self.input_discretizer.to(self.device)
        if self.output_discretizer is not None:
            self.output_discretizer.to(self.device)

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
                
    def _fetch_kwargs_for_fn(self, fn, exclude = [] ,**kwargs):
        input_names = inspect.signature(fn).parameters.keys()
        inputs = {}
        for name in input_names:
            if name in kwargs and name not in exclude:
                inputs[name] = kwargs[name]
        return inputs
    
    def prepare_inputs_for_forward(self, **kwargs) -> Dict[str, Any]: 
        raise NotImplementedError
    
    def teacher_forced_model_forward(self, **kwargs):
        raise NotImplementedError
    
    def return_output_dict(self, outputs) -> Dict[str, Any]:
        raise NotImplementedError
    
    def forward(self, teacher_force_output: bool = False, max_output_length = None, **kwargs) -> Dict[str, Any]:
        
        if max_output_length is None:
            max_output_length = self.max_lengths['output']
        
        inputs_for_forward = self.prepare_inputs_for_forward(**kwargs)
        
        if "max_output_length" not in inputs_for_forward:
            inputs_for_forward["max_output_length"] = max_output_length
        
        if not teacher_force_output:
            inputs = self._fetch_kwargs_for_fn(
                self.sequential_forward,
                exclude= [ "model" , "discretizer"],
                **inputs_for_forward
            )
            outputs = self.sequential_forward(self.model, self.output_discretizer, **inputs) 
        
        else:
            inputs = self._fetch_kwargs_for_fn(self.teacher_forced_model_forward, **inputs_for_forward)
            outputs = self.teacher_forced_model_forward(**inputs)
        
        return self.return_output_dict(outputs)

    def sequential_forward(self, model, discretizer, *args, **kwargs):
        raise NotImplementedError