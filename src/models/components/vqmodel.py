import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from diffusers import VQModel

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

class CustomVQModel(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        encoder_outputs = self.encode(x)
        latents = encoder_outputs['latents']  # Shape: [batch_size, latent_channels, H, W]

        # Quantize the latents
        quantized_latents, commit_loss, code_indices = self.quantize(latents)

        # Handle the extra dimension in code_indices
        code_indices = code_indices[-1]

        # Reshape code_indices to [batch_size, H, W]
        H, W = latents.shape[2], latents.shape[3]
        code_indices = code_indices.view(x.shape[0], H, W)

        # Ensure quantized_latents have correct shape for decoding
        quantized_latents = quantized_latents.view(x.shape[0], self.config.latent_channels, H, W)

        # Decode the quantized latents
        post_quant_latents = self.post_quant_conv(quantized_latents)
        recon_images = self.decoder(
            post_quant_latents,
            quantized_latents if self.config.norm_type == "spatial" else None
        )

        return {
            'reconstructions': recon_images,
            'commitment_loss': commit_loss.mean(),
            'code_indices': code_indices
        }

# Initialize the model
vqvae = CustomVQModel(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(64, 128),
    layers_per_block=2,
    act_fn='silu',
    latent_channels=64,
    num_vq_embeddings=256,  # Size of the codebook
    vq_embed_dim=64,        # Dimensionality of the embeddings
    scaling_factor=1.0,
)
vqvae.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    vqvae.train()
    total_loss = 0
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = vqvae(images)
        recon_images = outputs['reconstructions']
        commit_loss = outputs['commitment_loss']
        code_indices = outputs['code_indices']

        # Compute reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_images, images)

        # Total loss combines reconstruction loss and commitment loss
        loss = recon_loss + commit_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Optionally, access and use 'code_indices' as the code matrix
        if (i + 1) % 100 == 0:
            print(f'Code indices shape: {code_indices.shape}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(train_loader):.4f}')

# Visualization of original and reconstructed images
vqvae.eval()
with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images.to(device)
    outputs = vqvae(images)
    recon_images = outputs['reconstructions']

    # Access code matrix for visualization
    indices = outputs['code_indices'][0].cpu().numpy()  # First image's code indices

    # Move images to CPU for visualization
    images = images.cpu()
    recon_images = recon_images.cpu()

    # Create grids of images
    grid_original = make_grid(images[:64], nrow=8)
    grid_recon = make_grid(recon_images[:64], nrow=8)

    # Plot original images
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.title('Original Images')
    plt.imshow(grid_original.permute(1, 2, 0))
    plt.axis('off')

    # Plot reconstructed images
    plt.subplot(1, 3, 2)
    plt.title('Reconstructed Images')
    plt.imshow(grid_recon.permute(1, 2, 0))
    plt.axis('off')

    # Visualize code matrix for the first image
    plt.subplot(1, 3, 3)
    plt.title('Code Matrix of First Image')
    plt.imshow(indices, cmap='viridis')
    plt.axis('off')

    plt.show()
    plt.savefig('reconstruction.png')
