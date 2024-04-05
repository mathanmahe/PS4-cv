from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from proj4_code.segmentation.resnet import resnet50

class SimpleSegmentationNet(nn.Module):
    """
    ResNet backbone, with no increased dilation and no PPM, and a barebones
    classifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        deep_base: bool = True,
    ) -> None:
        """ """
        # super(SimpleSegmentationNet, self).__init__()
        super().__init__()

        self.criterion = criterion
        self.deep_base = deep_base

        resnet = resnet50(pretrained=pretrained, deep_base=True)
        self.resnet = resnet
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool,
        )

        self.cls = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the network. The loss function has already been set for
        you (self.criterion).

        Args:
            x: tensor of shape (N,C,H,W) representing batch of normalized input
                image
            y: tensor of shape (N,H,W) representing batch of ground truth labels
                Note: Handle the case when y is None, by setting main_loss and aux_loss
                to None.
        Returns:
            logits: tensor of shape (N,num_classes,H,W) representing class scores
                at each pixel
            yhat: tensor of shape (N,H,W) representing predicted labels at each
                pixel
            main_loss: loss computed on output of final classifier
            aux_loss:loss computed on output of auxiliary classifier (from
                intermediate output). Note: aux_loss is set to a dummy value,
                since we are not using an auxiliary classifier here, but we
                keep the same API as PSPNet in the EC section.
        """
        _, _, H, W = x.shape 

        x = self.layer0(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.cls(x)

        

        ############################################################################
        # Student code begin
        # Upsample the output to (H,W) using Pytorch's functional              #
        # `interpolate`, then compute the loss, and the predicted label per    #
        # pixel (yhat).                                                        #
        # Note: x has shape (N, num_classes, 7, 7) after the conv layers       # 
        ############################################################################
        # print("x size", x.size())
        # print("logits size", logits.size())
        
        logits = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        

        yhat = torch.argmax(logits, dim=1)

        main_loss = self.criterion(logits, y) if y is not None else None
        
        aux_loss = torch.Tensor([0]) if y is not None else None # this is set to a dummy value and is not being used

        ############################################################################
        # Student code end
        ############################################################################
        return logits, yhat, main_loss, aux_loss
