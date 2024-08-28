from symbolic_bottleneck.utils import instantiate_from_config

config_mbart = {
    '_target_': 'transformers.MBartForConditionalGeneration',
    'init_attribute': 'from_pretrained',
    'pretrained_model_name_or_path': "facebook/mbart-large-50-many-to-many-mmt"
}

# config_mbart = {
#     '_target_': 'transformers.PreTrainedModel',
#     'pretrained_model_name': "facebook/mbart-large-50-many-to-many-mmt"
# }

model = instantiate_from_config(config_mbart)