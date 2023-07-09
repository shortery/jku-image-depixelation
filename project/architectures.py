import torch

class SimpleCNN(torch.nn.Module):
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        num_hidden_layers: int,
        kernel_size: int = 3,
        activation_function: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        
        super().__init__()

        layers = []
        for i in range(num_hidden_layers):

            # add a CNN layer with appropriate padding to keep the dimensions the same
            conv_layer = torch.nn.Conv2d(
                in_channels=input_channels if i == 0 else hidden_channels,
                out_channels=output_channels if i == num_hidden_layers-1 else hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.append(conv_layer)

            # add an activation function    
            layers.append(activation_function)

        # register all layers    
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    