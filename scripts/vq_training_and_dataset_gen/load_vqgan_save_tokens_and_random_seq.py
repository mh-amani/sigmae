from datasets import load_from_disk
import numpy as np

data_path = '/dlabscratch1/amani/sigmae/data/vqgan_cifar10/vqgan_cifar10_tokens'

# Load the dataset from disk
hf_datasets = load_from_disk(data_path)

# Function to create random sequences as the 'X' feature
def _create_random_batch_of_tokens(batchsize, seed=None):
    bos_token = 3
    eos_token = 2
    pad_token = 0
    max_num_tokens = 100
    vocab_size = 20
    
    if seed is not None:
        np.manual_seed(seed)
    
    random_batch_of_tokens = np.random.randint(4, vocab_size, (batchsize, max_num_tokens))

    random_batch_of_tokens[:, 0] = bos_token  # Set beginning-of-sequence token
    ending_index = np.random.randint(1, max_num_tokens, (batchsize,))
    
    for i in range(batchsize):
        random_batch_of_tokens[i, ending_index[i]] = eos_token  # Set end-of-sequence token
        random_batch_of_tokens[i, ending_index[i] + 1:] = pad_token  # Pad the remaining tokens
    
    return random_batch_of_tokens

# Modify the dataset: add 'X' (random sequence) and 'z' (flattened tokens) features
def process_example(example):
    # Flatten the 'token' into 'z'
    flattened_z = np.array(example['tokens']).flatten()
    # add bos
    flattened_z = np.insert(flattened_z, 0, 3)

    # Create a random sequence 'X'
    random_sequence = _create_random_batch_of_tokens(1)[0]  # 1 example at a time

    return {'z': flattened_z, 'x': random_sequence,}

# Apply transformation to the whole dataset and remove unused columns
hf_datasets_new = hf_datasets.map(process_example, remove_columns=['image', 'label'])

# Save the new dataset with the 'X' and 'z' features
new_data_path = '/dlabscratch1/amani/sigmae/data/vqgan_cifar10/vqgan_cifar10_tokens_with_random_seq_20_100'
hf_datasets_new.save_to_disk(new_data_path)

print(f"New dataset saved to: {new_data_path}")
