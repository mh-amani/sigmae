

import hydra
from omegaconf import DictConfig
import numpy as np
import os
import torch
from src.utils import hydra_custom_resolvers
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
import hydra



def run_inference(config: DictConfig):
    # assert config.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    # config.output_dir = general_helpers.get_absolute_path(config.output_dir)
    # log.info(f"Output directory: {config.output_dir}")

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    # print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.ckpt_path
    
    print(f"Instantiating data module <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data, _recursive_=False)

    print(f"Instantiating model <{config.model._target_}>")
    config.model._target_ = config.model._target_+'.load_from_checkpoint'
    model: LightningModule = hydra.utils.instantiate(config.model, checkpoint_path=ckpt_path, _recursive_=False)
    model.eval()

    # print(f"Instantiating trainer <{config.trainer._target_}>")
    # trainer = hydra.utils.instantiate(config.trainer)


    # print("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
   
    return model, datamodule

def send_batch_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

path = "/dlabscratch1/amani/sigmae/logs/train/runs/2024-09-12_13-07-22/"

batch_size = 32
model_path = path + "checkpoints/last.ckpt"
configs_path = path + ".hydra/"
config_name = "config.yaml"
with hydra.initialize_config_dir(config_dir=configs_path, version_base="1.2"):
    config = hydra.compose(config_name=config_name, 
                           overrides=[f"data.batch_size={batch_size}", 
                                    f"ckpt_path={model_path}"
                                    ])

model, datamodule = run_inference(config)
datamodule.processor_z = model.processor_z
datamodule.processor_x = None # for image text
datamodule.setup('test', processor_z=model.processor_z)

test_datamodule = datamodule.test_dataloader()
batch = next(iter(test_datamodule))
batch = send_batch_to_device(batch, model.device)
output = model.model_step(batch, 'test')



# # constructing forward pass
# def forward(model, datamodule, id, dataset='val'):
#     model_x_to_z = model.model_x_to_z
#     model_z_to_x = model.model_z_to_x
#     disc_x = model.disc_x
#     disc_z = model.disc_z

#     if dataset=='val':
#         batch = datamodule.data_val[id]
#     elif dataset=='train':
#         batch = datamodule.data_train[id]
#     elif dataset=='test':
#         batch = datamodule.data_test[id]
    
#     collated_batch = model.collator.collate_fn([batch])
#     collated_batch_device = datamodule.transfer_batch_to_device(batch=collated_batch, device=model.device, dataloader_idx=0)

#     forward_from_batch(model, batch, collated_batch_device)


# def correct_val_predictions(model, datamodule):
#     correct_z_ids = []
#     correct_x_ids = []

#     val_dataloader = datamodule.val_dataloader()
#     for batch in val_dataloader:
        
#         collated_batch = batch
#         collated_batch_device = datamodule.transfer_batch_to_device(batch=collated_batch, device=model.device, dataloader_idx=0)

#         loss, losses, outputs = model.forward(batch=collated_batch_device, stage='val')

#         x_ids = collated_batch_device['x_ids']
#         x_hat_ids = outputs['zxz']['x_hat_ids']
#         x_ids, x_hat_ids = pad_label_label(collated_batch_device['x_ids'][:, 1:], x_hat_ids, pad_token_id=model.pad_token_id)
#         x_pad_mask = torch.logical_not(torch.eq(x_ids, model.pad_token_id))
#         x_hat_ids = x_hat_ids * x_pad_mask

#         z_ids = collated_batch_device['z_ids']
#         z_hat_ids = outputs['xzx']['z_hat_ids']
#         z_ids, z_hat_ids = pad_label_label(z_ids[:, 1:], z_hat_ids, pad_token_id=model.pad_token_id)
#         z_pad_mask = torch.logical_not(torch.eq(z_ids, model.pad_token_id))
#         z_hat_ids = z_hat_ids * z_pad_mask

#         z_pred_flag = torch.eq(z_hat_ids, z_ids).all(axis=-1)
#         z_pred_flag_ids = list(torch.where(z_pred_flag)[0].cpu().numpy())

#         x_pred_flag = torch.eq(x_hat_ids, x_ids).all(axis=-1)
#         x_pred_flag_ids = list(torch.where(x_pred_flag)[0].cpu().numpy())

#         correct_z_ids.extend(batch['id'][z_pred_flag_ids])
#         correct_x_ids.extend(batch['id'][x_pred_flag_ids])

#     return correct_z_ids, correct_x_ids

# def forward_from_batch(model, batch, collated_batch_device):
    
#     model_x_to_z = model.model_x_to_z
#     model_z_to_x = model.model_z_to_x
#     disc_x = model.disc_x
#     disc_z = model.disc_z

#     loss, losses, outputs = model.forward(batch=collated_batch_device, stage='val')
#     x_hat_ids = outputs['zxz']['x_hat_ids']
#     z_hat_ids = outputs['xzx']['z_hat_ids']
#     x_ids, x_hat_ids = pad_label_label(collated_batch_device['x_ids'][:, 1:], x_hat_ids, pad_token_id=model.pad_token_id)
#     z_ids, z_hat_ids = pad_label_label(collated_batch_device['z_ids'][:, 1:], z_hat_ids, pad_token_id=model.pad_token_id)
#     x_hat_scores = torch.round(outputs['zxz']['x_hat_scores'][0],  decimals=3)
#     z_hat_scores = torch.round(outputs['xzx']['z_hat_scores'][0],  decimals=3)
    

#     print('x= '.rjust(20), batch['x'])
#     print('x_ids= '.rjust(20), x_ids.cpu().numpy())
#     print('x_hat_text= '.rjust(20), model.collator.tokenizer_x.decode(x_hat_ids.cpu().numpy()[0]))
#     print('x_hat_ids= '.rjust(20), x_hat_ids.cpu().numpy())
#     print('equalities= '.rjust(20), (x_hat_ids == x_ids).cpu().numpy())
#     print('x_hat_scores= '.rjust(20), x_hat_scores.max(dim=-1).values.cpu().detach().numpy())
#     print('_____________________________________________________________________________________')
#     print('z= '.rjust(20), batch['z'])
#     print('z_ids= '.rjust(20), z_ids.cpu().numpy())
#     print('z_hat_text= '.rjust(20), model.collator.tokenizer_z.decode(z_hat_ids.cpu().numpy()[0]))
#     print('z_hat_ids= '.rjust(20), z_hat_ids.cpu().numpy())
#     print('equalities= '.rjust(20), (z_hat_ids == z_ids).cpu().numpy())
#     print('z_hat_scores= '.rjust(20), z_hat_scores.max(dim=-1).values.cpu().detach().numpy())

# def pred_from_sample(model, datamodule, input, print_x_or_z):
    
#     unknown_buffer = '[unk]'
#     if print_x_or_z == 'x':
#         x = unknown_buffer
#         z = input
#         data_type = np.array([0, 1])
#     elif print_x_or_z == 'z':
#         x = input
#         z = unknown_buffer
#         data_type = np.array([1, 0])

#     sample = {'id':0 , 'x':x, 'z':z, 'data_type': data_type}
#     # sample = {'id': 5,
#     #         'x': 'run around left and walk around left twice',
#     #         'z': 'I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK',
#     #         'data_type': np.array([ True,  True])}

#     collated_sample = model.collator.collate_fn([sample])
#     collated_samples_device = datamodule.transfer_batch_to_device(batch=collated_sample, device=model.device, dataloader_idx=0)
#     # print(collated_samples_device)

#     model_x_to_z = model.model_x_to_z
#     model_z_to_x = model.model_z_to_x
#     disc_x = model.disc_x
#     disc_z = model.disc_z

#     loss, losses, outputs = model.forward(batch=collated_samples_device, stage='val')
    
#     if print_x_or_z == 'x':
#         x_hat_ids = outputs['zxz']['x_hat_ids']
#         x_ids, x_hat_ids = pad_label_label(collated_samples_device['x_ids'][:, 1:], x_hat_ids, pad_token_id=model.pad_token_id)
#         x_hat_scores = torch.round(outputs['zxz']['x_hat_scores'][0],  decimals=3)
#         print('input z = '.rjust(20), sample['z'])
#         print('x_hat_text= '.rjust(20), model.collator.tokenizer_x.decode(x_hat_ids.cpu().numpy()[0]))
#         print('x_hat_scores= '.rjust(20), x_hat_scores.max(dim=-1).values.cpu().detach().numpy())
#         print(x_hat_scores[-1].cpu().detach().numpy())
    
#     elif print_x_or_z == 'z':
#         z_hat_ids = outputs['xzx']['z_hat_ids']
#         z_ids, z_hat_ids = pad_label_label(collated_samples_device['z_ids'][:, 1:], z_hat_ids, pad_token_id=model.pad_token_id)
#         z_hat_scores = torch.round(outputs['xzx']['z_hat_scores'][0],  decimals=3)
#         print('input x = '.rjust(20), sample['x'])
#         print('z_hat_text= '.rjust(20), model.collator.tokenizer_z.decode(z_hat_ids.cpu().numpy()[0]))
#         print('z_hat_scores= '.rjust(20), z_hat_scores.max(dim=-1).values.cpu().detach().numpy())
#         print(z_hat_scores[-1].cpu().detach().numpy())

# def print_neat_matrix(matrix):
#     for row in matrix:
#         print("[", end=" ")
#         for value in row:
#             formatted_value = f"{value:.4f}"
#             padding = max(0, 8 - len(formatted_value))
#             print(f"{' ' * padding}{formatted_value}, ", end="")
#         print("]")

# def pad_label_label(label, pred, pad_token_id):
#     pass
