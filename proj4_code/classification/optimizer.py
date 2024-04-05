"""
This class contains helper functions which will help get the optimizer
"""

import torch


def compute_quadratic_loss(w: torch.tensor) -> torch.tensor:
    """Computes the quadratic loss w^2 - 10w + 25

    Args:
        w: the value to compute the loss at.

    Useful functions: torch.pow(), torch.square()

    Returns:
        Computed loss value
    """
    assert w.shape == (1,)

    L = None

    ############################################################################
    # Student code begin
    ############################################################################
    L = torch.pow(w,2) - 10*w + 25

    ############################################################################
    # Student code end
    ############################################################################
    return L


def gradient_descent_step(w: torch.tensor, L: torch.tensor, lr: float = 1e-3) -> None:
    """Perform a single step of gradient descent.

    Note: you need to update the input w itself and not return anything

    Args:
        w: input variable.
        L: loss.
        lr (optional): learning rate/step size. Defaults to 1e-3.
    """
    # manually zero out the gradient to prevent accumulation
    if w.grad is not None:
        w.grad.zero_()

    # perform backward on loss (we need to retain graph here otherwise Pytorch will throw it away)
    L.backward(retain_graph=True)

    gradient = w.grad

    step = None
    with torch.no_grad():
        ############################################################################
        # Student code begin
        ############################################################################
        # i tried w = w- lr*gradient and it didnt work because it has to be inplace, 
        # we do not need to use gradient.data
        w -= (lr * gradient)

        ############################################################################
        # Student code end
        ############################################################################


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config on the model.

    Note: config has a minimum of three entries. Feel free to add more entries if you want.
    But do not change the name of the three existing entries

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    """

    optimizer = None

    optimizer_type = config["optimizer_type"]
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]

    ############################################################################
    # Student code begin
    ############################################################################
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    ############################################################################
    # Student code end
    ############################################################################

    return optimizer
