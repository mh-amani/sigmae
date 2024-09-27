import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import os

# Import the VQGanVAE from Lucidrains' DALLE-pytorch repository
from dalle_pytorch import VQGanVAE


bsize=32

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
        recon_loss = nn.functional.mse_loss(recon_images, x)
        
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

# Define transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=bsize, num_workers=4,)

# Initialize the pre-trained VQGAN VAE
vae = CustomVQGanVAE().to(device)
vae.train()

# Define the optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass: compute the loss and get reconstructions
        outputs = vae(images)
        loss = outputs['loss']
        recon_images = outputs['reconstructions']
        code_indices = outputs['code_indices']

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}')

# Visualization of original and reconstructed images
saving_path = './data/vqgan_cifar10'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

vae.eval()
with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images.to(device)
    outputs = vae(images)
    recon_images = outputs['reconstructions']
    code_indices = outputs['code_indices']

    # Move images to CPU and clamp to [0, 1] range for visualization
    images = images.cpu().clamp(0.0, 1.0)
    recon_images = recon_images.cpu().clamp(0.0, 1.0)

    # Create grids of images
    grid_original = make_grid(images[:32], nrow=8)
    grid_recon = make_grid(recon_images[:32], nrow=8)

    # Plot original and reconstructed images
    plt.figure(figsize=(15, 7))

    # Original images
    plt.subplot(1, 2, 1)
    plt.title('Original Images')
    plt.imshow(grid_original.permute(1, 2, 0))
    plt.axis('off')

    # Reconstructed images
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Images')
    plt.imshow(grid_recon.permute(1, 2, 0))
    plt.axis('off')

    plt.show()
    plt.savefig(f'{saving_path}/reconstructions.png')





# Save the model
torch.save(vae.state_dict(), f'{saving_path}/vqgan_cifar10.pth')
print('Model saved successfully! Path:', f'{saving_path}/vqgan_cifar10.pth')


# Import Hugging Face Datasets library
from datasets import Dataset, Features, Array3D, ClassLabel, Value, Array2D

# Prepare the datasets and dataloaders
# Reuse the transform used during training
transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256x256
    transforms.ToTensor(),
])

# Training dataset and dataloader (already defined earlier)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=bsize, num_workers=4)

# Validation and Test datasets
val_size = 5000
test_size = 5000
test_val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the test set into validation and test sets
val_dataset, test_dataset = torch.utils.data.random_split(test_val_dataset, [val_size, test_size])

val_loader = DataLoader(val_dataset, batch_size=bsize, num_workers=4,)
test_loader = DataLoader(test_dataset, batch_size=bsize, num_workers=4,)
# Obtain spatial dimensions (H, W) of code indices
vae.eval()
with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images.to(device)
    outputs = vae(images)
    code_indices = outputs['code_indices']  # Shape: [batch_size, H, W]
    H, W = code_indices.shape[1], code_indices.shape[2]
    print(f"Code indices spatial dimensions: H={H}, W={W}")



from datasets import Features, Array3D, ClassLabel, Value, Array2D, Dataset, DatasetInfo, concatenate_datasets
from datasets import Features, ClassLabel, Array2D, Image

features = Features({
    'image': Image(),  # Use Image feature type
    'label': ClassLabel(names=train_dataset.classes),
    'tokens': Array2D(dtype='int64', shape=(H, W)),
})


import io
from PIL import Image as PILImage

def process_dataset_to_hf_streaming(dataloader):
    vae.eval()
    dataset_splits = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.tolist()

            # Obtain code indices (tokens) from the model
            outputs = vae(images)
            code_indices = outputs['code_indices']
            batch_tokens = code_indices.cpu().numpy().astype('int64')  # Ensure correct dtype

            # Convert images to bytes
            images_np = images.cpu().numpy()
            images_bytes = []
            for img_array in images_np:
                # Convert tensor to PIL Image
                img = PILImage.fromarray((img_array.transpose(1, 2, 0) * 255).astype('uint8'))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                images_bytes.append({'bytes': buf.getvalue()})

            # Prepare batch data
            batch_data = {
                'image': images_bytes,
                'label': labels,
                'tokens': batch_tokens.tolist()
            }

            # Create a Hugging Face Dataset from batch data
            batch_dataset = Dataset.from_dict(batch_data, features=features)
            dataset_splits.append(batch_dataset)

            # Optional: Print progress
            if batch_idx % 100 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")

    # Concatenate all batch datasets
    hf_dataset = concatenate_datasets(dataset_splits)
    return hf_dataset

# Process and save the datasets
train_hf_dataset = process_dataset_to_hf_streaming(train_loader)
val_hf_dataset = process_dataset_to_hf_streaming(val_loader)
test_hf_dataset =  process_dataset_to_hf_streaming(test_loader)

from datasets import DatasetDict

# Combine the datasets into a single DatasetDict
hf_datasets = DatasetDict({
    'train': train_hf_dataset,
    'val': val_hf_dataset,
    'test': test_hf_dataset
})
# Save the dataset to disk
save_path = f'{saving_path}/vqgan_cifar10_tokens'
if not os.path.exists(save_path):
    os.makedirs(save_path)

hf_datasets.save_to_disk(save_path)
print('Datasets saved successfully! Path:', save_path)
    