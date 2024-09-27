import numpy as np
from transformers.image_utils import (
    infer_channel_dimension_format,
    ChannelDimension,
)
from transformers.image_transforms import rescale, resize, to_channel_dimension_format


def unnormalize(images, mean, std, data_format=None):
    """
    Reverse the normalization of images.

    Args:
        images (np.ndarray): The normalized images with shape (batch_size, channels, height, width)
                             or (batch_size, height, width, channels).
        mean (float or List[float]): The mean used for normalization.
        std (float or List[float]): The std used for normalization.
        data_format (ChannelDimension, optional): The channel dimension format.

    Returns:
        np.ndarray: The unnormalized images.
    """
    if data_format is None:
        data_format = infer_channel_dimension_format(images)

    # if mean and std are single values, convert them to lists of the same length as the number of channels
    channel_dim = 1 if data_format == ChannelDimension.FIRST else -1
    if not isinstance(mean, list):
        mean = [mean] * images.shape[channel_dim]
        std = [std] * images.shape[channel_dim]
    mean = np.array(mean)
    std = np.array(std)

    images = images.copy()

    if data_format == ChannelDimension.FIRST:
        # images shape: (batch_size, channels, height, width)
        mean = mean[:, None, None]
        std = std[:, None, None]
    elif data_format == ChannelDimension.LAST:
        # images shape: (batch_size, height, width, channels)
        mean = mean[None, None, :]
        std = std[None, None, :]
    else:
        raise ValueError("Unsupported data format for unnormalization.")

    images = images * std + mean

    return images

class UnViTImageProcessor:
    def __init__(
        self,
        processor,
        original_height,
        original_width,
        data_format=ChannelDimension.FIRST,
    ):
        self.do_resize = processor.do_resize
        self.resample = processor.resample
        self.do_rescale = processor.do_rescale
        self.rescale_factor = processor.rescale_factor
        self.do_normalize = processor.do_normalize
        self.image_mean = processor.image_mean
        self.image_std = processor.image_std
        self.original_height = original_height
        self.original_width = original_width
        self.data_format = data_format

    def unprocess(self, images):
        if not isinstance(images, np.ndarray):
            raise ValueError("Input images must be a NumPy array.")

        input_data_format = infer_channel_dimension_format(images)

        # Reverse normalization
        if self.do_normalize:
            images = unnormalize(
                images,
                mean=self.image_mean,
                std=self.image_std,
                data_format=input_data_format,
            )

        # Reverse rescaling
        if self.do_rescale:
            images = rescale(
                images,
                scale=1 / self.rescale_factor,
                data_format=input_data_format,
            )

        # Ensure image values are within [0, 255]
        images = np.clip(images, 0, 255)

        # Convert images to uint8 before resizing
        images = images.astype(np.uint8)

        # Reverse resizing
        if self.do_resize:
            batch_size = images.shape[0]
            resized_images = []
            for i in range(batch_size):
                image = images[i]

                # No need to clip and convert here as it's already done
                # Resize the image
                image = resize(
                    image,
                    size=[self.original_height, self.original_width],
                    resample=self.resample,
                    data_format=None,
                    input_data_format=input_data_format,
                )
                resized_images.append(image)

            images = np.stack(resized_images, axis=0)

        # Ensure the images are in the desired data format
        images = to_channel_dimension_format(
            images, self.data_format, input_channel_dim=input_data_format
        )

        return images


# from transformers import ViTImageProcessor
# import numpy as np

# # Instantiate the processor
# processor = ViTImageProcessor()

# # Suppose you have a batch of images as a NumPy array
# # images: NumPy array of shape (batch_size, channels, height, width) or (batch_size, height, width, channels)
# # For this example, let's create dummy data
# batch_size = 4
# channels = 3
# original_height = 224
# original_width = 224
# images = np.random.randint(
#     0, 256, (batch_size, channels, original_height, original_width), dtype=np.uint8
# )

# # Preprocess the images
# processed = processor.preprocess(images, return_tensors="np")
# pixel_values = processed["pixel_values"]  # NumPy array

# # Create the unprocessor with original dimensions
# unprocessor = UnViTImageProcessor(
#     processor,
#     original_height=original_height,
#     original_width=original_width,
#     data_format=ChannelDimension.FIRST,  # or ChannelDimension.LAST
# )

# # Unprocess the images
# recovered_images = unprocessor.unprocess(pixel_values)


# # Check if the recovered images are close to the original images
# print(np.allclose(images, recovered_images, atol=1))
