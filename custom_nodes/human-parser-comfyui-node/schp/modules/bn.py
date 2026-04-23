import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

def _toolchain_available() -> bool:
    if shutil.which("c++") is None:
        return False
    if torch.cuda.is_available():
        return shutil.which("nvcc") is not None or os.path.exists("/usr/local/cuda/bin/nvcc")
    return True


if _toolchain_available():
    try:
        from .functions import *  # noqa: F403
        _HAS_INPLACE_ABN = True
    except Exception:  # noqa: BLE001
        ACT_RELU = "relu"
        ACT_LEAKY_RELU = "leaky_relu"
        ACT_ELU = "elu"
        ACT_NONE = "none"
        inplace_abn = None
        inplace_abn_sync = None
        _HAS_INPLACE_ABN = False
else:
    ACT_RELU = "relu"
    ACT_LEAKY_RELU = "leaky_relu"
    ACT_ELU = "elu"
    ACT_NONE = "none"
    inplace_abn = None
    inplace_abn_sync = None
    _HAS_INPLACE_ABN = False


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training, self.momentum, self.eps)

        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__(num_features, eps, momentum, affine, activation, slope)

    def forward(self, x):
        if not _HAS_INPLACE_ABN:
            return super().forward(x)
        x, _, _ = inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var,
                           self.training, self.momentum, self.eps, self.activation, self.slope)
        return x


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization
    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DistributedDataParallel`.
    """

    def forward(self, x):
        if not _HAS_INPLACE_ABN:
            return super().forward(x)
        x, _, _ =  inplace_abn_sync(x, self.weight, self.bias, self.running_mean, self.running_var,
                                   self.training, self.momentum, self.eps, self.activation, self.slope)
        return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


