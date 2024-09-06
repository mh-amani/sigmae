import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
#Check if accelerator library is available if not throw an error
try:
    import accelerate
except ImportError:
    raise ImportError("Please install accelerate library to run this script")

#################### DATASET ####################
normalization_mean = 0.5
normalization_std = 0.5

device = "cuda:1" if torch.cuda.is_available() else "cpu"
save_name = "gauss_cifar"
ds_to_use = save_name
num_epochs = 3
do_train = True
sliding_window = 20
use_one_digit_per_class = False
gaussian_noise = True

normalization_mnist = transforms.Normalize((normalization_mean,), (normalization_std,))
normalization_cifar = transforms.Normalize((normalization_mean, normalization_mean, normalization_mean), (normalization_std, normalization_std, normalization_std))

normalization = normalization_cifar if "cifar" in ds_to_use  else normalization_mnist
# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
cifar_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
# Create a DataLoader
if "cifar" in ds_to_use:
    dataloader = DataLoader(cifar_dataset, batch_size=32, shuffle=True)
else:
    dataloader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
    

from transformers import AutoModel, GPT2Tokenizer, PreTrainedTokenizer

#################### MODEL ####################
# Load the GPT-2 tokenizer and model
patch_width = 8 if "cifar" in ds_to_use else 7
patch_height = 8 if "cifar" in ds_to_use else 7
channels = 3 if "cifar" in ds_to_use else 1
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", device_map = device)
model_hidden_size = model.config.n_embd
image_patch_head = torch.nn.Linear(model_hidden_size, patch_width * patch_height *channels).to(device)
img_embed_head = torch.nn.Linear(patch_width * patch_height * channels, model_hidden_size).to(device)
image_width = 32 if "cifar" in ds_to_use else 28
image_height = 32 if "cifar" in ds_to_use else 28
num_patches_to_predict = image_width*image_height//(patch_width*patch_height)

class DigitEmbedding(torch.nn.Module):
    def __init__(self, n_digits,embedding_dim):
        super(DigitEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(n_digits + 1, embedding_dim) #bos token
        self.n_digits = n_digits
        self.bos_token_id = n_digits
        
    def forward(self, x):
        return self.embedding(x)

class MyModel(torch.nn.Module):
    def __init__(self, model, digit_embedding ,image_patch_head, img_embed_head,gaussian_noise):
        super(MyModel, self).__init__()
        self.model = model
        self.image_patch_head = image_patch_head
        self.img_embed_head = img_embed_head
        self.digit_embedding = digit_embedding
        self.gaussian_noise = gaussian_noise
        if gaussian_noise:
            self.mean_estimator = torch.nn.Linear(model.config.n_embd, model.config.n_embd)
            self.std_estimator = torch.nn.Linear(model.config.n_embd, model.config.n_embd)
            self.init_mean_std()
            
    def init_mean_std(self):
        # Estimate the standard deviation based on the number of input features
        std_mean = 1.0 / torch.sqrt(torch.tensor(self.mean_estimator.in_features, dtype=torch.float32))
        std_std = 1.0 / torch.sqrt(torch.tensor(self.std_estimator.in_features, dtype=torch.float32))
        # Initialize weights with a normal distribution, then take the absolute value to ensure positivity
        with torch.no_grad():
            self.mean_estimator.weight.data.normal_(mean=0.0, std=std_mean)
            self.std_estimator.weight.data.normal_(mean=0.0, std=std_std).abs_()
            
        # Optionally, you can initialize the bias as well
        if self.mean_estimator.bias is not None:
            self.mean_estimator.bias.data.zero_()
        if self.std_estimator.bias is not None:
            self.std_estimator.bias.data.zero_()

    def forward(self, embeds = None, text_id = None ,text_embeds = None, img_embeds = None):
        
        valid_input_condition = (embeds is not None and text_embeds is None and text_id is None) or \
                                (embeds is None and text_embeds is not None and text_id is None) or \
                                (embeds is None and text_embeds is None and text_id is not None)
        
        assert valid_input_condition, "Provide exactly one of embeds, text_id or text_embeds"
        
        if text_id is not None:
            assert text_embeds is None, "text_embeds should be None when text_id is provided"
            text_embeds = self.digit_embedding(text_id)
                
        if img_embeds is not None:
            hidden_img_embeds = self.img_embed_head(img_embeds)
            inputs_embeds = torch.cat((text_embeds, hidden_img_embeds), dim=1)    

        elif img_embeds is None and text_embeds is not None:
            inputs_embeds = text_embeds
        
        elif embeds is not None:
            inputs_embeds = embeds
        
        else:
            raise ValueError("Provide exactly one of embeds, text_id or text_embeds")    
        
        outputs = self.model(inputs_embeds=inputs_embeds)
        
        if self.gaussian_noise:
            num_input_embeds = inputs_embeds.size(1)
            output_embeds_last_hidden_state = outputs.last_hidden_state[..., num_input_embeds:,:]
            
            mean = self.mean_estimator(output_embeds_last_hidden_state)
            std = torch.nn.functional.relu(output_embeds_last_hidden_state)
            epsilon = torch.normal(
                mean=torch.zeros_like(mean),
                std=torch.ones_like(std)
            ).to(mean.device)
            
            sampled_state = torch.cat(
                (
                    outputs.last_hidden_state[...,:num_input_embeds,:],
                    mean + std * epsilon
                ),
                dim=-2
            ).to(outputs.last_hidden_state) 
         
            #check for nan
            if torch.isnan(sampled_state).any():
                breakpoint()
            image_patches = self.image_patch_head(sampled_state)
             
        else:
            image_patches = self.image_patch_head(outputs.last_hidden_state)
        
        img_embed = self.img_embed_head(image_patches)
        return {"image_patches": image_patches, "img_embeds": img_embed}

n_digits = 10
digit_embedding = DigitEmbedding(n_digits, model.config.n_embd).to(device)  # Embedding size matches GPT-2's embedding size

bos_token_id = digit_embedding.bos_token_id
bos_embed = model.wte(torch.tensor(bos_token_id).to(model.wte.weight.device))

model = MyModel(model, digit_embedding ,image_patch_head, img_embed_head,gaussian_noise)

def repatch_image(patches, patch_width, patch_height, image_width, image_height):
    width_patches = image_width//patch_width
    height_patches = image_height//patch_height
    assert width_patches * patch_width == image_width, "Width not divisible by patch width"
    assert height_patches * patch_height == image_height, "Height not divisible by patch height"
    
    image = torch.zeros(channels, image_width, image_height)
    for i, patch in enumerate(patches):
        height_patch = i // width_patches
        width_patch = i % width_patches
        
        start_height = height_patch * patch_height
        end_height = start_height + patch_height
        start_width = width_patch * patch_width
        end_width = start_width + patch_width
        image[..., start_height:end_height, start_width:end_width] = patch
    return image
    

def prepare_image_labels(image, patch_width, patch_height):
    
    image_width = image.size(2)
    image_height = image.size(3)
    chans = image.size(1)
    width_patches = image_width//patch_width
    height_patches = image_height//patch_height
    
    assert width_patches * patch_width == image_width, "Width not divisible by patch width"
    assert height_patches * patch_height == image_height, "Height not divisible by patch height"
    patches = []
    
    for height_patch in range(0, height_patches):
        start_height = height_patch * patch_height
        end_height = start_height + patch_height

        for width_patch in range(width_patches):
            start_width = width_patch * patch_width
            end_width = start_width + patch_width
            patch = image[..., start_height:end_height, start_width:end_width]
            patches.append(patch.reshape(patch.size(0), -1))

    return torch.stack(patches, dim=1)

def prepare_input_sequence(labels):
    if len(labels.size()) < 2:
        labels = labels.unsqueeze(1)
    
    batch_size = labels.size(0)
    seq_len = labels.size(1)
    input_sequence = torch.zeros(batch_size, seq_len + 1, dtype=torch.long).to(labels.device)
    input_sequence[:, 0] = bos_token_id
    input_sequence[:, 1:] = labels
    return input_sequence

from torch.optim import Adam
from tqdm import tqdm
# Define optimizer
lr = 1e-3 # was previously 1e-4
min_lr = 1e-5
min_lr_ratio = min_lr / lr
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.L1Loss() if "cifar" in ds_to_use else torch.nn.MSELoss() 

# Training loop
model.to(device)
model.train()

img_dict = {num: None for num in range(n_digits)}


for batch in dataloader:
    images, labels = batch
    for label, image in zip(labels, images):
        if img_dict[label.item()] is None:
            img_dict[label.item()] = image
    if  all([val is not None for val in img_dict.values()]):
        break

len_og_dataset = len(dataloader)
#Remake datloader with img_dict
dataset = [(img, torch.tensor(label)) for label, img in img_dict.items()]


dl2 = DataLoader(dataset, batch_size=10, shuffle=True)
    

if use_one_digit_per_class:
    dataloader = dl2
    num_epochs = len_og_dataset//len(dataloader) *num_epochs
    
len_dl = len(dataloader)
# Define linear learning rate decay function
def lr_lambda(step):
    return max(1 - step / (num_epochs * len_dl),min_lr_ratio)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
 
if do_train:
 
    aggregate_loss = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            with torch.no_grad():
                images, labels = batch
                image_patches = prepare_image_labels(images.to(device), patch_width, patch_height)
                input_sequence = prepare_input_sequence(labels.to(device))
            
            outputs = model(text_id = input_sequence, img_embeds = image_patches)
            logits = outputs["image_patches"]
            img_embed = outputs["img_embeds"]
            # Calculate loss (using an appropriate loss function)
            input_seq_len = input_sequence.size(1)
            preds = logits[:, input_seq_len-1:-1, ...]
            
            # if random_patch_loss:
            #     random_patches = torch.randint(0, preds.size(1), (num_patches_to_predict,))
            #     predicted_patches = preds[:, random_patches, ...].reshape(-1, channels * patch_width * patch_height)
            #     labs = image_patches[:, random_patches, ...].reshape(-1, patch_width * patch_height * channels)
            
            # else:
            predicted_patches = preds.contiguous().view(-1, channels * patch_width * patch_height)
            labs = image_patches.contiguous().view(-1, patch_width * patch_height * channels)
            
            
            loss = loss_fn(predicted_patches, labs)  # Customize loss_fn as necessary
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if len(aggregate_loss) == sliding_window == 0:
                aggregate_loss.pop(0)
            aggregate_loss.append(loss.item())
            mean_agg_loss = torch.mean(torch.tensor(aggregate_loss)).item()
            #check if mean of aggreagte loss is nan
            if math.isnan(mean_agg_loss):
                breakpoint()
            progress_bar.set_postfix({"loss": mean_agg_loss, "lr": optimizer.param_groups[0]["lr"]})

    #save mode
    torch.save(model.state_dict(),f"{save_name}.pth")
    #save last image
    im = images[0].permute(1, 2, 0).squeeze().numpy()
    im = im * normalization_std + normalization_mean
    plt.imsave(f"{ds_to_use}_last_image.png", im)
    #save last patches
    pred_ptch = [preds[0,i].reshape(channels, patch_width, patch_height) for i in range(preds.size(1))]
    pred_im = repatch_image(pred_ptch, patch_width, patch_height, image_width, image_height)
    pred_im = np.clip(pred_im.permute(1, 2, 0).squeeze().detach().numpy(), 0, 1)
    renorm_pred_im = np.clip(pred_im * normalization_std + normalization_mean, 0, 1)
    plt.imsave(f"{ds_to_use}_last_pred_image.png", pred_im)
    plt.imsave(f"{ds_to_use}_last_pred_image_renorm.png", renorm_pred_im)

# Load model
model.load_state_dict(torch.load(f"{save_name}.pth"))

# Inference
model.eval()
digits = torch.arange(n_digits).to(device)
input_sequences = prepare_input_sequence(digits)


images = []
for digit,seq in enumerate(input_sequences):
    input_sequence = seq.unsqueeze(0)
    embeds = model.digit_embedding(input_sequence).to(device)
    image = []
    for i in tqdm(range(num_patches_to_predict), desc=f"Generating patches of digit {digit}"):
        with torch.no_grad():
            outputs = model(embeds)
            patch = outputs["image_patches"][..., -1,:].reshape(-1, channels ,patch_width, patch_height)
            image.append(patch.to("cpu"))
            next_input = outputs["img_embeds"][..., -1,:]
            embeds = torch.cat((embeds, next_input.unsqueeze(1)), dim=1)
    images.append(repatch_image(image, patch_width, patch_height, image_width, image_height))

#saving non teacher forced images
fig, ax = plt.subplots(1, n_digits, figsize=(20, 20))
for i in range(n_digits):
    im = images[i].permute(1, 2, 0).squeeze().numpy()
    im = np.clip(im * normalization_std + normalization_mean, 0, 1)
    ax[i].imshow(im)
    ax[i].axis("off")
    ax[i].set_title(f"Digit {i}")
plt.show()
#save figure
fig.savefig(f"{ds_to_use}_digits.png")

#save the images
for i in range(n_digits):
    im = images[i].permute(1, 2, 0).squeeze().numpy()
    im = np.clip(im * normalization_std + normalization_mean, 0, 1)
    plt.imsave(f"{ds_to_use}_digit_{i}.png", im)    

teacher_forced_images = []
progress_bar = tqdm(dl2, desc=f"Teacher forcing inference")
aggregate_loss = []
per_patch_loss = []
teacher_forced_images = {i: None for i in range(n_digits)}

for batch in progress_bar:
    with torch.no_grad():
        images, labels = batch
        image_patches = prepare_image_labels(images.to(device), patch_width, patch_height)
        input_sequence = prepare_input_sequence(labels.to(device))
    
        outputs = model(text_id = input_sequence, img_embeds = image_patches)
        logits = outputs["image_patches"]
        img_embed = outputs["img_embeds"]
        # Calculate loss (using an appropriate loss function)
        input_seq_len = input_sequence.size(1)
        preds = logits[:, input_seq_len-1:-1, ...]
        for samp,label in zip(preds,labels):
            tmp = []
            for patch in samp:
                tmp.append(patch.reshape(channels, patch_width, patch_height))
            teacher_forced_images[label.item()] = repatch_image(tmp, patch_width, patch_height, image_width, image_height)
        #total loss
        predicted_patches = preds.contiguous().view(-1, channels * patch_width * patch_height)
        labs = image_patches.contiguous().view(-1, patch_width * patch_height * channels)
        loss = loss_fn(predicted_patches, labs)  # Customize loss_fn as necessary
        aggregate_loss.append(loss.item())
        
        #per patch loss
        for i in range(preds.size(1)):
            pred = preds[:, i, ...]
            lab = image_patches[:, i, ...]
            per_patch_loss.append(loss_fn(pred, lab).item())
        
print(f"Mean loss: {torch.mean(torch.tensor(aggregate_loss)).item()}")
print("Per patch loss:")
for i in range(num_patches_to_predict):
    print(f"    Patch {i}: {per_patch_loss[i]}")

#saving teacher forced images
fig, ax = plt.subplots(1, n_digits, figsize=(20, 20))
for i in range(n_digits):
    im = teacher_forced_images[i].permute(1, 2, 0).squeeze().numpy()
    im = np.clip(im * normalization_std + normalization_mean, 0, 1)
    ax[i].imshow(im)
    ax[i].axis("off")
    ax[i].set_title(f"Digit {i}")
plt.show()
#save figure
fig.savefig(f"tf_{ds_to_use}_digits.png")

#save the images
for i in range(n_digits):
    im = teacher_forced_images[i].permute(1, 2, 0).squeeze().numpy()
    im = np.clip(im * normalization_std + normalization_mean, 0, 1)
    plt.imsave(f"tf_{ds_to_use}_digit_{i}.png", im)
    
#save gt images
fig, ax = plt.subplots(1, n_digits, figsize=(20, 20))
for batch in dl2:
    imgs, labels = batch
    for label,im in zip(labels,imgs):
        im = im.permute(1, 2, 0).squeeze().numpy()
        im = np.clip(im * normalization_std + normalization_mean, 0, 1)
        ax[label.item()].imshow(im)
        ax[label.item()].axis("off")
        ax[label.item()].set_title(f"Digit {label}")
plt.show()
#save figure
fig.savefig(f"gt_{ds_to_use}_digits.png")
breakpoint()