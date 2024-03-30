import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from sgm_cuda_c import sgm_forward


class SGMCudaFunction(Function):
    @staticmethod
    def forward(
        ctx,
        left: torch.Tensor,
        right: torch.Tensor,
        p1: int,
        p2: int
    ):
        assert left.is_cuda
        assert right.is_cuda
        assert len(left.size()) == 2

        disp = sgm_forward(left, right, p1, p2)
        return disp


sgm_cuda = SGMCudaFunction.apply