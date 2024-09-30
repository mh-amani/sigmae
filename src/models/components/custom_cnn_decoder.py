import torch
import torch.nn as nn

class CustomCNNDecoder(nn.Module):
    def __init__(self, in_channels=10, out_channels=3, num_filters=64, num_layers=4):
        super(CustomCNNDecoder, self).__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1  # This maintains spatial dimensions
                )
            )
            layers.append(nn.GroupNorm(num_groups=32, num_channels=num_filters))
            layers.append(nn.SiLU())
            channels = num_filters
        # Final layer to map to desired output channels
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
