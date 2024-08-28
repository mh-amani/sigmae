import importlib

def instantiate_from_config(config):
    # Extract the class path and the parameters from the config
    config = config.copy()
    class_path = config.pop('_target_')
    module_name, class_name = class_path.rsplit('.', 1)

    # Dynamically import the module and get the class
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    init_attribute = config.get('init_attribute', None)

    if init_attribute:
        config.pop('init_attribute')
        attributed_cls = getattr(cls, init_attribute)
    else:
        attributed_cls = cls

    if config.get('config', None):
        instance = attributed_cls(config['config'])
    else:
        instance = attributed_cls(**config)

    

    # Instantiate the class with the remaining config parameters
    # if 'pretrained_model_name' in config:
    #     pretrained_model_name = config['pretrained_model_name']
    #     # Instantiate the pre-trained model
    #     instance = cls.from_pretrained(pretrained_model_name,)
    # else:
    #     # Instantiate the class with the remaining config parameters    
    
            
    return instance
