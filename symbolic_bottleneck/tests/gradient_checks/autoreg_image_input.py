import torch
from transformers import BartForConditionalGeneration, BartConfig
from symbolic_bottleneck.auto_reg_wrapper.image_input_auto_reg_wrapper import ImageInputAutoRegWrapper
from symbolic_bottleneck.models.vision_transformers.vit_gpt2 import UnWrappedVITGPT2, discretizer_dec_config, config_vit_gpt2
import transformers
from symbolic_bottleneck.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck


def return_autoreg_wrapped_model():
    """Initialize and return an autoregressively wrapped Vision-Encoder-Decoder model."""
    torch.manual_seed(42)  # Fix seed for reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize VisionEncoderDecoder model
    model = transformers.VisionEncoderDecoder(
        transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(**config_vit_gpt2)
    ).to(device)

    # Unwrap model components for custom bottleneck
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()

    # Define the discretizer with embeddings
    discretizer = SoftmaxDiscreteBottleneck(
        {**discretizer_dec_config, 'encoder_embedding': encoder_embedding,
         'decoder_embedding': decoder_embedding, 'linear_head': linear_head}
    ).to(device)

    # Wrap the model with autoregressive wrapper
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, discretizer, discretizer, config_vit_gpt2).to(device)

    return enfr_autoreg_wrapped_model


def test_gradients_vector_model_functionality():
    """Test gradients for vector model components."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0  # Force first token to 0 for test purposes

    # Forward pass - get embeddings and compute model outputs
    input_embeds = discretizer.encoder_embedding_from_id(input_ids)
    output_embeds = discretizer.encoder_embedding_from_id(output_ids[:, 0:1])
    output_vector_model = vector_model(
        inputs_embeds=input_embeds, decoder_inputs_embeds=output_embeds
    )['last_hidden_state']

    # Compute loss and gradients
    discrete_output = discretizer(output_vector_model)
    loss = torch.nn.functional.nll_loss(torch.log(discrete_output['score'][0, 1:]), output_ids[:, 1].view(-1))

    # Backward pass - calculate gradients
    input_embeds.retain_grad()
    loss.backward()

    # Print gradients
    print(f"Grad of loss w.r.t input embedding: {input_embeds.grad}")
    print(f"Grad of loss w.r.t linear head: {discretizer.linear_head.weight.grad}")
    print(f"Loss: {loss}")


def test_gradients_autoregged_wrapped_functionality():
    """Test gradients for the autoregressively wrapped model."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0

    # Forward pass through the autoregressively wrapped model
    output = enfr_autoreg_wrapped_model(input_ids, max_output_length=seq_length)
    loss = torch.nn.functional.nll_loss(torch.log(output['score'][:, 2]), output_ids[:, 2].view(-1))

    # Compute gradients
    output['score'].retain_grad()
    loss.backward()

    # Print gradients
    print(f"Gradient of loss w.r.t output scores: {output['score'].grad}")
    print(f"Loss: {loss}")


def test_simple_training_autoregged_wrapped():
    """Run a simple training loop for the autoregressively wrapped model."""
    optimizer = torch.optim.Adam(enfr_autoreg_wrapped_model.parameters(), lr=0.01)
    
    for i in range(n_train_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        output_ids[:, 0] = 0  # Set first token to 0

        # Forward pass
        output = enfr_autoreg_wrapped_model(input_ids, max_output_length=seq_length)
        loss = torch.nn.functional.nll_loss(torch.log(output['score'][:, 1]), output_ids[:, 1].view(-1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for each step
        print(f"Step {i + 1}, Loss: {loss}")


if __name__ == '__main__':
    # Run tests
    test_gradients_autoregged_wrapped_functionality()
    test_simple_training_autoregged_wrapped()
