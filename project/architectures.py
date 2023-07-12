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

            if i != num_hidden_layers-1:
                # add an activation function to hidden layers   
                layers.append(activation_function)

        # register all layers    
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual skip connection around the whole network
        return x[:, :1, :, :] + self.layers(x)
    


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        squeeze_channels: int,
        kernel_size: int = 3,
        activation_function: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        """Block with residual skip connection"""
        
        super().__init__()

        self.conv_layer_1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=squeeze_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation_layer_1 = activation_function

        self.conv_layer_2 = torch.nn.Conv2d(
            in_channels=squeeze_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation_layer_2 = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.conv_layer_1(x)
        x = self.activation_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.activation_layer_2(x)
        return x + skip
        

class ResidualCNN(torch.nn.Module):
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        squeeze_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        activation_function: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        
        super().__init__()

        blocks = []

        # add convolution layer to change the dimension
        self.conv_layer_in = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        for _ in range(num_blocks):
            # add residual block
            res_block = ResidualBlock(
                hidden_channels,
                squeeze_channels,
                kernel_size,
                activation_function
            )
            blocks.append(res_block)

        # add convolution layer to change the dimension
        self.conv_layer_out = torch.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # register all layers
        self.blocks = torch.nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer_in(x)
        x = self.blocks(x)
        x = self.conv_layer_out(x)
        return x
