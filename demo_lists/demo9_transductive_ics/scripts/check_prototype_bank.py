#!/usr/bin/env python3
"""CPU numerical checks for the confidence-free bank control, without images/weights/CUDA."""
import torch
from prototype_bank_eval import bank_mask

f = torch.tensor([[1., 0.]]*6+[[0., 1.]]*6)
mask = torch.tensor([True]*6+[False]*6)
target = torch.tensor([[1., 0.], [0., 1.]])
base = torch.tensor([False, True])
y, count = bank_mask([f], [mask], target, base)
assert y.tolist() == [True, False] and count == [6, 6]
y, _ = bank_mask([f], [torch.zeros(12, dtype=torch.bool)], target, base)
assert torch.equal(y, base)
y, _ = bank_mask([], [], target, base)
assert torch.equal(y, base)
y, _ = bank_mask([f], [mask], target, base, torch.tensor([0, 0]))
assert y.tolist() == [False, False]
y, _ = bank_mask([f, f], [mask, mask], target, base)
assert y.tolist() == [True, False]
print('PASSED: 5 CPU checks: centroid direction, missing class, empty bank, cluster averaging, replication invariance')
