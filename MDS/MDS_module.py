import torch
from torch import nn
from torch.autograd import Function
import MDS

class MinimumDensitySampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint, mean_mst_length):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative radius point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        mean_mst_length : torch.Tensor
            (B) the average edge length from expansion penalty module

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        idx = torch.zeros(xyz.shape[0], npoint, requires_grad= False, device='cuda', dtype=torch.int32).contiguous()
        MDS.minimum_density_sampling(xyz, npoint, mean_mst_length, idx)
        return idx

    @staticmethod
    def backward(grad_idx, a=None):
        return None, None, None


minimum_density_sample = MinimumDensitySampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return MDS.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = MDS.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply

