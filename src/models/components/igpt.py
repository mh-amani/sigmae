from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
import torch
import matplotlib.pyplot as plt
import numpy as np

image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# unconditional generation of 8 images
batch_size = 4
context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token
context = context.to(device)
output = model.generate(
    input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
)

clusters = image_processor.clusters
height = image_processor.size["height"]
width = image_processor.size["width"]

samples = output[:, 1:].cpu().detach().numpy()
samples_img = [
    np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
]  # convert color cluster tokens back to pixels
f, axes = plt.subplots(1, batch_size, dpi=300)

for img, ax in zip(samples_img, axes):
    ax.axis("off")
    ax.imshow(img)

print("Generated images:")

