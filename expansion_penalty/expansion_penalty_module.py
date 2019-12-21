# Expansion penalty module (based on minimum spanning tree)
# author: Minghua Liu

# Input:
# xyz: [#batch, #point] 
# primitive_size: int, the number of points of sampled points for each surface elements, which should be no greater than 512
# in each point cloud, the points from the same surface element should be successive
# alpha: float, > 1, only penalize those edges whose length are greater than (alpha * mean_length)

#Output:
# dist: [#batch, #point], if the point u is penalized then dist[u] is the distance between u and its neighbor in the MST, otherwise dist[u] is 0
# assignment: [#batch, #point], if the point u is penalized then assignment[u] is its neighbor in the MST, otherwise assignment[u] is -1
# mean_mst_length: [#batch], the average length of the edeges in each point clouds


import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import expansion_penalty

# GPU tensors only
class expansionPenaltyFunction(Function):
    @staticmethod
    def forward(ctx, xyz, primitive_size, alpha):
        assert(primitive_size <= 512)
        batchsize, n, _ = xyz.size()
        assert(n % primitive_size == 0)
        xyz = xyz.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        neighbor = torch.zeros(batchsize, n * 512,  device='cuda', dtype=torch.int32).contiguous()
        cost = torch.zeros(batchsize, n * 512, device='cuda').contiguous()
        mean_mst_length = torch.zeros(batchsize, device='cuda').contiguous()
        expansion_penalty.forward(xyz, primitive_size, assignment, dist, alpha, neighbor, cost, mean_mst_length)
        ctx.save_for_backward(xyz, assignment)
        return dist, assignment, mean_mst_length / (n / primitive_size)

    @staticmethod
    def backward(ctx, grad_dist, grad_idx, grad_mml):
        xyz, assignment = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_xyz = torch.zeros(xyz.size(), device='cuda').contiguous()
        expansion_penalty.backward(xyz, grad_xyz, grad_dist, assignment)
        return grad_xyz, None, None

class expansionPenaltyModule(nn.Module):
    def __init__(self):
        super(expansionPenaltyModule, self).__init__()

    def forward(self, input, primitive_size, alpha):
        return expansionPenaltyFunction.apply(input, primitive_size, alpha)

def test_expansion_penalty():
    x = torch.rand(20, 8192, 3).cuda()
    print("Input_size: ", x.shape)
    expansion = expansionPenaltyModule()
    start_time = time.perf_counter()
    dis, ass, mean_length = expansion(x, 512, 1.5)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))

#test_expansion_penalty()