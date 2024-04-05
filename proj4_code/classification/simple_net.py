import torch
import torch.nn as nn
import os


class SimpleNet(nn.Module):
    """Simple Network with atleast 2 conv2d layers and two linear layers."""

    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Hints:
        1. Refer to https://pytorch.org/docs/stable/nn.html for layers
        2. Remember to use non-linearities in your network. Network without
        non-linearities is not deep.
        3. You will get 3D tensor for an image input from self.cnn_layers. You need 
        to process it and make it a compatible tensor input for self.fc_layers.
        """
        super().__init__()
        '''
        based on the first part of the simplenet.jpg image, we need 1 input channel
        because it is a grayscale image. output channel needs to be 10 because we need 10 feature maps/filters of size 60x60 each. 
        this is indeed the case, because we use the formula (64-5+1) x (64-5+1), then we get 60x60


        for the second convolutional layer, we get (20-5+1) x(20-5+1) = 16x16
        '''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0), #input 1x64x64 output 10x60x60
            # the output from the previous layer is 60x60, but the output after the pool should be 20x20
            # outputsize = ((inputsize - filtersize) / stride ) + 1
            # therefore in order to get 20x20, stride must be 3 => 60-3 = 57, divide by 3 is 19 + 1 = 20 matches the output

            nn.MaxPool2d(kernel_size=3,stride=3,padding=0), #input 10x60x60 output 10x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0), #input  10x20x20 output 20x16x16
            # here the output would be 16x16
            
            # but we cannot use a 3x3 kernel to get the output, we can use a 4x4 kernel with stride 3 to get output 5x5
            # output = 16-4 / 3 + 1 = 5
            
            nn.MaxPool2d(kernel_size=4,stride=3,padding=0), # i 20x16x16 o 20x5x5
            nn.ReLU(),
            # at the end of this relu, we will have 20 features at 5x5 size each, which if we flatten we willget dimensions 500(5x5x20)
            nn.Flatten()
        )  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential(
            nn.Linear(500,100),
            nn.Linear(100,15)
        )  # linear and supporting layers here
        # self.loss_criterion = nn.MSELoss(reduction='sum')
        # self.loss_criterion = nn.KLDivLoss()
        self.loss_criterion = nn.CrossEntropyLoss()

        self.count=1
        

        ############################################################################
        # Student code begin
        ############################################################################

        # TESTING TO UNDERSTAND TENSORS
        # # here is how we can test the conv_layers by passing it through a dummy input, it will give an output tensor, [1,500] in which the first is the batch size and the second dimension is the flattened feature data.
        # dummy_input = torch.randn(1, 1, 64, 64)  # Batch size of 1, 1 channel, 64x64 image
        # output = self.conv_layers(dummy_input)

        # # Print out the shape of the output tensor
        # print("output size", output.size())
        # print(output)
        # # a tensor is basically a multidimensional array







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
        #The following are helper variables and are OPTIONAL to use. They are meant to help guide you to calculate model_output.

        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape(), , or nn.Flatten()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        
        conv_features = self.conv_layers(x)
        # print("conv_features size",conv_features.size()) #1x500

        # conv_features[0] initially i used this, but this does not seem to be necessary, as the torch outputs nx15, so it has to be 2 dimensional, so we can leave the input as 1x500
        linear_model_input = conv_features
        # print("linear model input", linear_model_input.size()) #1x500

        model_output = self.fc_layers(linear_model_input)


        # print("model output", model_output.size()) #1x15
        ############################################################################
        # Student code end
        ############################################################################
        return model_output
    
    
