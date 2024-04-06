import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0), #input 1x64x64 output 10x60x60
            nn.MaxPool2d(kernel_size=3,stride=3,padding=0), #input 10x60x60 output 10x20x20
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0), #input  10x20x20 output 20x16x16
            nn.MaxPool2d(kernel_size=4,stride=3,padding=0), # i 20x16x16 o 20x5x5
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Flatten()
        )  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential(
            nn.Linear(500,100),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(100,15)
        )  # linear and supporting layers here
        # self.loss_criterion = nn.MSELoss(reduction='sum')
        # self.loss_criterion = nn.KLDivLoss()
        self.loss_criterion = nn.CrossEntropyLoss()
        ############################################################################
        # Student code begin
        ############################################################################

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        conv_features = self.conv_layers(x)
        linear_model_input = conv_features
        model_output = self.fc_layers(linear_model_input)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
