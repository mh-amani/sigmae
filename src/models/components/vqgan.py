import torch
from dalle_pytorch import VQGanVAE

# Initialize VQGanVAE
vae = VQGanVAE()

text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)

u = vae.model.encoder(torch.randn(4, 3, 256, 256)) # torch.Size([4, 256, 16, 16])

# Prepare dummy data (replace with your actual data)
batch_size = 4
text_tokens = torch.randint(0, 10000, (batch_size, 256))    # Dummy text tokens
images = torch.randn(batch_size, 3, 256, 256)               # Dummy images

# Get image tokens
image_tokens = vae.get_codebook_indices(images)

# Training step
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

optimizer.zero_grad()
loss = vae(
    text=text_tokens,
    images=images,           # Alternatively, use image_tokens=image_tokens
    return_loss=True
)
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')
