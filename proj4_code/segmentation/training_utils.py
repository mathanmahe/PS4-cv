
from typing import Tuple

import torch
from torch import nn

import proj4_code.segmentation.cv2_transforms as transform
from proj4_code.segmentation.pspnet import PSPNet
from proj4_code.segmentation.simple_segmentation_net import SimpleSegmentationNet


def get_model_and_optimizer(args) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Create your model, optimizer and configure the initial learning rates.

    Use the SGD optimizer, use a parameters list, and set the momentum and
    weight decay for each parameter group according to the parameter values in `args`.

    Create 5 param groups for the 0th + 1st,2nd,3rd,4th ResNet layer modules, and
    then add separate groups afterwards for the classifier and/or PPM heads.

    You should set the learning rate for the resnet layers to the base learning
    rate (args.base_lr), and you should set the learning rate for the new PSPNet
    PPM and classifiers to be 10 times the base learning rate.

    Args:
        args: object containing specified hyperparameters, including the "arch"
           parameter that determines whether we should return PSPNet or the
           SimpleSegmentationNet
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.arch == "PSPNet":
        model = PSPNet(
            layers=args.layers,
            num_classes=args.classes,
            zoom_factor=args.zoom_factor,
            criterion=criterion,
            pretrained=args.pretrained,
            use_ppm=args.use_ppm
        )
        modules_orig = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        if args.use_ppm:
            modules_new = [model.ppm, model.cls, model.aux]
        else:
            modules_new = [model.cls, model.aux]
    elif args.arch == "SimpleSegmentationNet":
        model = SimpleSegmentationNet(pretrained=args.pretrained, num_classes=args.classes, criterion=criterion)
        modules_orig = [model.layer0, model.resnet.layer1, model.resnet.layer2, model.resnet.layer3, model.resnet.layer4]
        modules_new = [model.cls]

    params_list = []
    for module in modules_orig:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))

    # different learning rate after 5th param group
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))

    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return model, optimizer



def update_learning_rate(current_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Given an updated current learning rate, set the ResNet modules to this current
    learning rate, and the classifiers/PPM module to 10x the current lr.

    Hint: You can loop over the dictionaries in the optimizer.param_groups list, and set
    a new "lr" entry for each one. They will be in the same order you added them above,
    so if the first N modules should have low learning rate, and the next M modules should
    have a higher learning rate, this should be easy modify in two loops.

    Note: this depends upon how you implemented the param groups above.
    """

    for index in range(0, 5):
        optimizer.param_groups[index]["lr"] = current_lr
    for index in range(5, len(optimizer.param_groups)):
        optimizer.param_groups[index]["lr"] = current_lr * 10

    return optimizer



def get_train_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the training split, with data augmentation. Use the classes
    from `src/vision.cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    then random horizontal flipping, blurring, rotation, scaling (in any order),
    followed by taking a random crop of size (args.train_h, args.train_w), converting
    the Numpy array to a Pytorch tensor, and then normalizing by the
    Imagenet mean and std (provided here).

    Note that your scaling should be confined to the [scale_min,scale_max] params in the
    args. Also, your rotation should be confined to the [rotate_min,rotate_max] params.

    To prevent black artifacts after a rotation or a random crop, specify the paddings
    to be equal to the Imagenet mean to pad any black regions.

    You should set such artifact regions of the ground truth to be ignored.

    Use the classes
    from `proj6_code.cv2_transforms`, imported above as `transform`.

    Args:
        args: object containing specified hyperparameters

    Returns:
        train_transform
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose(
        [
            transform.ResizeShort(args.short_size),
            transform.RandScale([args.scale_min, args.scale_max]),
            transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop(
                [args.train_h, args.train_w], crop_type="rand", padding=mean, ignore_label=args.ignore_label
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform


def get_val_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the val split, with no data augmentation. Use the classes
    from `proj6_code.cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    taking a *center* crop of size (args.train_h, args.train_w) with a padding equal
    to the Imagenet mean, converting the Numpy array to a Pytorch tensor, and then
    normalizing by the Imagenet mean and std (provided here).

    Args:
        args: object containing specified hyperparameters

    Returns:
        val_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose(
        [
            transform.ResizeShort(args.short_size),
            transform.Crop(
                [args.train_h, args.train_w], crop_type="center", padding=mean, ignore_label=args.ignore_label
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
    )

    return val_transform
