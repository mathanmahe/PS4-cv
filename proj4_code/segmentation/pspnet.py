from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from proj4_code.segmentation.resnet import resnet50
from proj4_code.segmentation.ppm import PPM
import math

class PSPNet(nn.Module):
    """
    The final feature map size is 1/8 of the input image.
    Use the dilated network strategy described in
    ResNet-50 has 4 blocks, and those 4 blocks have [3, 4, 6, 3] layers, respectively.
    """

    def __init__(
        self,
        layers: int = 50,
        bins=(1, 2, 3, 6),
        dropout: float = 0.1,
        num_classes: int = 2,
        zoom_factor: int = 8,
        use_ppm: bool = True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained: bool = True,
        deep_base: bool = True,
    ) -> None:
        """
        Args:
            layers: int = 50,
            bins: list of grid dimensions for PPM, e.g. (1,2,3) means to create (1x1), (2x2), and (3x3) grids
            dropout: float representing probability of dropping out data
            num_classes: number of classes
            zoom_factor: scale value used to upsample the model output's (HxW) size to (H * zoom_factor, W * zoom_factor)
            use_ppm: boolean representing whether to use the Pyramid Pooling Module
            criterion: loss function module
            pretrained: boolean representing ...
        """
        # super(PSPNet, self).__init__()
        super().__init__()
        assert layers == 50
        assert 2048 % len(bins) == 0
        print("Number of Classes", num_classes)
        assert num_classes > 0
        assert zoom_factor in [1, 2, 4, 8]
        self.dropout = dropout
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.ppm = None
        self.cls = None
        self.aux = None

        ############################################################################
        # Student code begin
        # Initialize your ResNet backbone, and set the layers                     
        # layer0, layer1, layer2, layer3, layer4. Note: layer0 should be sequential
        ############################################################################
        backbone = resnet50(pretrained=pretrained, deep_base=deep_base)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        ############################################################################
        # Student code end
        ############################################################################
        

        self.__replace_conv_with_dilated_conv()

        ############################################################################
        # Student code begin
        # Initialize the PPM. The reduction_dim should be equal to the            #
        # output number of ResNet feature maps, divided by the number of PPM bins #
        # Afterwards, set fea_dim to the updated feature dimension to be passed   #
        # to the classifier
        ############################################################################
        self.ppm = PPM(in_dim=2048, reduction_dim=(2048 // len(bins)), bins=bins)
        fea_dim = 2048 + len(bins) * (2048 // len(bins))

        ############################################################################
        # Student code end
        ############################################################################

        self.cls = self.__create_classifier(in_feats=fea_dim, out_feats=512, num_classes=num_classes)
        self.aux = self.__create_classifier(in_feats=1024, out_feats=256, num_classes=num_classes)

    def __replace_conv_with_dilated_conv(self):
        """Increase the receptive field by reducing stride and increasing dilation.
        In Layer3, in every `Bottleneck`, we will change the 3x3 `conv2`, we will
        replace the conv layer that had stride=2, dilation=1, and padding=1 with a
        new conv layer, that instead has stride=1, dilation=2, and padding=2.
        In the `downsample` block, we'll also need to hardcode the stride to 1, instead of 2.
        In Layer4, for every `Bottleneck`, we will make the same changes, except we'll
        change the dilation to 4 and padding to 4.
        Hint: you can iterate over each layer's modules using the .named_modules()
        attribute, and check the name to see if it's the one you want to edit.
        Then you can edit the dilation, padding, and stride attributes of the module.
        """
        ############################################################################
        # Student code begin
        ############################################################################
        for name, module in self.layer3.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv2' in name:
                module.stride = (1, 1)
                module.dilation = (2, 2)
                module.padding = (2, 2)

        for name, module in self.layer4.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv2' in name:
                module.stride = (1, 1)
                module.dilation = (4, 4)
                module.padding = (4, 4)

        for name, module in self.layer3.named_modules():
            if 'downsample.0' in name:
                module.stride = (1, 1)

        for name, module in self.layer4.named_modules():
            if 'downsample.0' in name:
                module.stride = (1, 1)
        ############################################################################
        # Student code end
        ############################################################################

    def __create_classifier(self, in_feats: int, out_feats: int, num_classes: int) -> nn.Module:
        """Implement the final PSPNet classifier over the output categories.
        Args:
            in_feats: number of channels in input feature map
            out_feats: number of filters for classifier's conv layer
            num_classes: number of output categories
        Returns:
            cls: A sequential block of 3x3 convolution, 2d Batch norm, ReLU,
                2d dropout, and a final 1x1 conv layer over the number of output classes.
                The 3x3 conv layer's padding should preserve the height and width of the
                feature map. The specified dropout is defined in `self.dropout`.
        """
        cls = None
        ############################################################################
        # Student code begin
        ############################################################################
        return nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feats),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(out_feats, num_classes, kernel_size=1)
        )

        ############################################################################
        # Student code end
        ############################################################################
        return cls

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass of the network.
        Feed the input through the network, upsample the aux output (from layer 3)
        and main output (from layer4) to the ground truth resolution (based on zoom_factor), and then
        compute the loss and auxiliary loss.
        The aux classifier should operate on the output of layer3.
        The PPM should operate on the output of layer4.
        Note that you can return a tensor of dummy values for the auxiliary loss
        if the model is set to inference mode. Note that nn.Module() has a
         `self.training` attribute, which is set to True depending upon whether
        the model is in in training or evaluation mode.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#module
        comments on zoom_factor: 
            Because the final feature map size is 1/8 of the input image, 
            if the input to the network is of shape (N,C,H,W), then
            with a zoom_factor of 1, the output is computed logits 
            has shape (N,num_classes,H/8,W/8), yhat has shape (N,H/8,W/8)
            and the ground truth labels are of shape (N, H/8, W/8).
            If the zoom_factor is 2, the computed logits has shape 
            (N,num_classes,H/4,W/4), yhat has shape (N,H/4,W/4),
            and the ground truth labels is of shape (N,H/4,W/4).
            We will be testing your zoom_factor for values of [1, 2, 4, 8] and assume
            that the ground truth labels will have already beeen upsampled by the zoom_factor.
            When scaling the dimenions (H/8 * zoom_factor, W/8 * zoom_factor), 
            round up to the nearest integer value.
            Use Pytorch's functional interpolate for upsampling the outputs to the correct shape scale.
        Args:
            x: tensor of shape (N,C,H,W) representing batch of normalized input image
            y: tensor of shape (N,H/8 * zoom_factor,W/8 * zoom_factor) representing batch of ground truth labels
                Note: Handle the case when y is None, by setting main_loss and aux_loss
                to None. 
        Returns:
            logits: tensor of shape (N,num_classes,H/8 * zoom_factor,W/8 *zoom_factor) representing class scores at each pixel
            yhat: tensor of shape (N,H/8 * zoom_factor,W/8 * zoom_factor) representing predicted labels at each pixel
            main_loss: loss computed on output of final classifier if y is provided,
               else return None if no ground truth is passed in
            aux_loss:loss computed on output of auxiliary classifier (from intermediate output)
               if y is provided, else return None if no ground truth is passed in
               Note: set a dummy value for aux_loss if self.training == False
        """
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0

        ############################################################################
        # Student code begin
        ############################################################################
        h, w = x_size[2], x_size[3]

        h = int(math.ceil(h / 8 * self.zoom_factor))
        w = int(math.ceil(w / 8 * self.zoom_factor))
        # print("initial", x.size())
        x = self.layer0(x)
        # print("0",x.size())
        x = self.layer1(x)
        # print("1",x.size())
        x = self.layer2(x)
        # print("2",x.size())
        x = self.layer3(x)
        # print("3",x.size())
        
        if self.training and self.use_ppm:
            aux = self.aux(x)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

        x = self.layer4(x)
        # print("4",x.size())
        x = self.ppm(x)
        # print("ppm",x.size())
        x = self.cls(x)
        # print("cls",x.size())
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        logits = x
        yhat = torch.argmax(logits, dim=1)

        if y is not None and self.training:
            main_loss = self.criterion(logits, y)
            aux_loss = self.criterion(aux, y) if self.use_ppm else torch.tensor(0)
            return logits, yhat, main_loss, aux_loss
        else:
            # During inference, return the logits and the predicted class labels (yhat).
            # You can return None for the losses.
            return logits, yhat, None, None

        ############################################################################
        # Student code end
        ############################################################################
        return logits, yhat, main_loss, aux_loss