from datasets import load_from_disk
from torchvision import transforms
import torch


data_path = '/dlabscratch1/amani/data/vqgan_cifar10/vqgan_cifar10_tokens'
model_path = '/dlabscratch1/amani/data/vqgan_cifar10/vqgan_cifar10.pth'

# Load the dataset from disk
hf_datasets = load_from_disk(data_path)

# Define transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256
    transforms.ToTensor(),
])

# load the vae model
from dalle_pytorch import VQGanVAE

# Define a custom VQGAN VAE class to handle the forward pass and loss computation
class CustomVQGanVAE(VQGanVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # Encode the images
        z = self.model.encoder(x)
        
        # Quantize the latents
        quant_z, commit_loss, indices = self.model.quantize(z)
        indices = indices[-1].view(x.shape[0], z.shape[2], z.shape[3])
        # torch.round(quant_z[0][:, :, 0].T - self.model.quantize.embedding.weight[738], decimals=2) # almost zero
        # Decode the quantized latents
        recon_images = self.model.decoder(quant_z)
        
        # Compute reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(recon_images, x)
        
        # Total loss combines reconstruction and commitment losses
        loss = recon_loss + commit_loss.mean()
        
        return {
            'reconstructions': recon_images,
            'loss': loss,
            'commitment_loss': commit_loss.mean(),
            'code_indices': indices
        }

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize the pre-trained VQGAN VAE
vae = CustomVQGanVAE().to(device)
vae.load_state_dict(torch.load(model_path))

vae.eval()

batch = hf_datasets['train'][0:32]
images = batch['image']
images = torch.stack([transform(image) for image in images])
batch['pixel_values'] = images
batch = batch['pixel_values'].to(device)
output = vae(batch)
print(output['loss'])

print(torch.tensor(hf_datasets['train'][0:32]['tokens']) - output['code_indices'].cpu())
print(not(torch.all(torch.tensor(hf_datasets['train'][0:32]['tokens']) - output['code_indices'].cpu())))