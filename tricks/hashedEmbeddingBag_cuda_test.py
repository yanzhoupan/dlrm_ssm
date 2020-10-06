# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import torch
import torch.nn as nn


# %%
import hashedEmbeddingBag
import hashed_embedding_bag

# %%
bag_num = 18

num_categories = 100
num_feature = 200

hashed_weight_size = 200


# %%
hashed_weights = torch.rand(hashed_weight_size)
#hashed_weights = torch.arange(start=0, end=hashed_weight_size, dtype=torch.float)
bag_size = torch.randint(low=0, high=7, size=(bag_num,))
indices_num = bag_size.sum().item()

indices = torch.randint(low=0, high=num_categories - 1, size=(indices_num, ))
offsets = torch.cat([torch.zeros(1, dtype=torch.long), bag_size.cumsum(dim=0)[:-1]])


# %%
print(hashed_weights)


# %%
print(bag_size)
print(indices)
print(offsets)
print(indices_num)


# %%
print(indices.shape)


# %%
mode = 0


# %%
device = torch.cuda.current_device()
hashed_weights = hashed_weights.to(device)
indices = indices.to(device)
offsets = offsets.to(device)

output, offset2bag, bag_size, max_indices, hashed_idx =   hashed_embedding_bag.forward(hashed_weights, indices, offsets, mode, num_feature)


# %%
print("output is:")
print(output)


# %%
print("offset2bag is:")
print(offset2bag)


# %%
print("bag_size is:")
print(bag_size)


# %%
print("max_indices is:")
print(max_indices)


# %%
print("hashed_idx is:")
print(hashed_idx)


# %%
print(hashed_idx.shape)


# %%
def toSignedInt(value, bits):
    valueUint8 = value & (2**bits - 1)
    if valueUint8 & 2**(bits-1):
       return valueUint8 - 2**bits
    return valueUint8
"""
def hash_function(a, b):
    tmp1 = toSignedInt(a * 9824516537, 64)
    tmp2 = toSignedInt(b * 57857966300227, 64)
    tmp3 = toSignedInt(tmp1 + tmp2, 64)
    tmp3 %= 117130198221199
    return tmp3
"""
def hash_function(a, b):
    return a + b


# %%
device = torch.device("cpu")
hashed_weights = hashed_weights.to(device)
indices = indices.to(device)
offsets = offsets.to(device)


# %%
output = output.to(device)
offset2bag = offset2bag.to(device)
bag_size = bag_size.to(device)
max_indices = max_indices.to(device)
hashed_idx = hashed_idx.to(device)


# %%
def make_offset2bag(offsets, indices):
    offsets2bag = torch.zeros(indices.size(0) + 1, dtype=indices.dtype, device=offsets.device)
    offsets2bag.index_add_(0, offsets, torch.ones_like(offsets, memory_format=torch.legacy_contiguous_format))
    offsets2bag[0] -= 1
    offsets2bag = offsets2bag.cumsum(0)
    offsets2bag.resize_(indices.size(0))
    return offsets2bag


# %%
expected_offsets2bag = make_offset2bag(offsets, indices)
assert((expected_offsets2bag - offset2bag).abs().sum().item() == 0)


# %%
expected_hashed_index = torch.zeros((indices_num, num_feature), dtype=torch.long)
expected_output = torch.zeros(bag_num, num_feature)
for i in range(indices.size(0)):
    for j in range(num_feature):
        weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
        expected_hashed_index[i, j] = weight_idx
        expected_output[expected_offsets2bag[i].item(), j] += hashed_weights[weight_idx]


# %%
print(expected_output)


# %%
assert(expected_hashed_index.equal(hashed_idx))
assert(expected_output.equal(output))


# %%
output_grad = torch.rand_like(expected_output)
#output_grad = torch.arange(start=0, end=num_feature, dtype=torch.float).unsqueeze(0).repeat(bag_num, 1)
#output_grad = torch.arange(start=0, end=bag_num, dtype=torch.float).unsqueeze(-1).repeat(1, num_feature)

output_grad[:, 0] = 0.5


# %%
expected_weight_grad = torch.zeros_like(hashed_weights)
for i in range(indices.size(0)):
    for j in range(num_feature):
        weight_idx = hashed_embedding_bag.hash(indices[i].item(), j) % hashed_weights.size(0)
        expected_weight_grad[weight_idx] += output_grad[offset2bag[i].item(), j]


# %%
device = torch.cuda.current_device()
hashed_weights = hashed_weights.to(device)
indices = indices.to(device)
offsets = offsets.to(device)
offset2bag = offset2bag.to(device)
bag_size = bag_size.to(device)
max_indices = max_indices.to(device)
hashed_idx = hashed_idx.to(device)


# %%
print(output_grad)


# %%
output_grad = output_grad.to(device)
weight_grad = hashed_embedding_bag.backward(
    output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), False, mode, num_feature)
weight_grad = weight_grad.cpu()


# %%
print(expected_weight_grad)
print(weight_grad)


# %%
assert((weight_grad - expected_weight_grad).sum().item() < 1)


# %%
import hashedEmbeddingBag

emb = hashedEmbeddingBag.HashEmbeddingBag(
    num_categories, num_feature, hashed_weight_size / (num_feature * num_categories), "sum", hashed_weights)


# %%
res = emb(indices, offsets)
res.retain_grad()
tmp = res * torch.rand_like(res)
y = tmp.sum()


# %%
y.backward()


# %%
res.grad


# %%
assert(res.cpu().equal(expected_output))


# %%
output_grad = res.grad
weight_grad = hashed_embedding_bag.backward(
    output_grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights.size(0), False, mode, num_feature)


# %%
print(emb.weight.grad)
print(weight_grad)
assert((emb.weight.grad - weight_grad).sum().abs().item() < 1)


