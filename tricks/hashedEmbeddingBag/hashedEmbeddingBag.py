from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math

import hashed_embedding_bag

class HashedEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hashed_weights, indices, offsets, mode, embedding_dim):
        if indices.dim() == 2:
            if offsets is not None:
                raise ValueError("if indices is 2D, then offsets has to be None"
                                ", as indices is treated is a mini-batch of"
                                " fixed length sequences. However, found "
                                "offsets of type {}".format(type(offsets)))
            offsets = torch.arange(0, indices.numel(), indices.size(1),
                                dtype=torch.long, device=indices.device)
            indices = indices.reshape(-1)
        elif indices.dim() == 1:
            if offsets is None:
                raise ValueError("offsets has to be a 1D Tensor but got None")
            if offsets.dim() != 1:
                raise ValueError("offsets has to be a 1D Tensor")
        else:
            raise ValueError("indices has to be 1D or 2D Tensor,"
                            " but got Tensor of dimension {}".format(indices.dim()))

        if mode == 'sum':
            mode_enum = 0
        elif mode == 'mean':
            mode_enum = 1
            raise ValueError("mean mode not supported")
        elif mode == 'max':
            mode_enum = 2
            raise ValueError("max mode not supported")

        
        hashed_weights_size = hashed_weights.size(0)
        output, offset2bag, bag_size, max_indices, hashed_idx = \
            hashed_embedding_bag.forward(hashed_weights, indices, offsets, mode_enum, embedding_dim)
        ctx.save_for_backward(indices, offsets, offset2bag, bag_size, max_indices, hashed_idx)
        ctx.mode_enum = mode_enum
        ctx.hashed_weights_size = hashed_weights_size
        return output

    @staticmethod
    def backward(ctx, grad):
        indices, offsets, offset2bag, bag_size, max_indices, hashed_idx = ctx.saved_variables
        hashed_weights_size = ctx.hashed_weights_size
        mode_enum = ctx.mode_enum
        embedding_dim = grad.size(1)
        weight_grad = hashed_embedding_bag.backward(
            grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights_size, False, mode_enum, embedding_dim)
        return weight_grad, None, None, None, None

class HashedEmbeddingBag(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        compression:float = 1. / 64., 
        mode:str = "sum", 
        _weight: Optional[torch.Tensor] = None) -> None:
        super(HashedEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_size = int(num_embeddings * embedding_dim * compression)
        print("Inside HashedEmbeddingBag: ", num_embeddings, embedding_dim, compression, weight_size)
        if _weight is None:
            low = -math.sqrt(1 / self.num_embeddings)
            high = math.sqrt(1 / self.num_embeddings)
            self.hashed_weight = Parameter(torch.rand(weight_size) * (high - low) + low)
            #self.reset_parameters()
            print("Inside HashedEmbeddingBag (after reset): ", num_embeddings, embedding_dim, compression, weight_size, self.hashed_weight.shape)
        else:
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            self.hashed_weight = Parameter(_weight)
            self.weight_size = self.hashed_weight.numel()
        self.mode = mode
    """
    def reset_parameters(self) -> None:
        # init.normal_(self.weight)
        W = np.random.uniform(
                low=-np.sqrt(1 / self.num_embeddings), high=np.sqrt(1 / self.num_embeddings), size=(self.hashed_weight.shape[0], )
            ).astype(np.float32)
        self.hashed_weight.data = torch.tensor(W, requires_grad=True)
    """
    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return HashedEmbeddingBagFunction.apply(
            self.hashed_weight,
            indices,
            offsets,
            self.mode,
            self.embedding_dim
        )
