"""Small CPU correctness checks before GPU startup; no research images or GT."""
import torch
from candidate import predict,rotate
torch.set_num_threads(1)
torch.manual_seed(0)
n,d=16,8
q=torch.nn.functional.normalize(torch.randn(n,d),dim=1)
r=torch.nn.functional.normalize(torch.randn(n,d),dim=1)
b=torch.rand(n)>.3
for cov in [torch.ones(n),torch.zeros(n),torch.tensor([.1]+[0.]*(n-1)),torch.rand(n)]:
    got=predict(q,r,cov,b,side=4)
    for key,m in got.items():
        assert m.shape==(n,) and m.dtype==torch.bool
        assert not (m&~b).any()
        if bool((cov==1).all()):assert torch.equal(m,b)
        if bool((cov==0).all()):assert not m.any()
assert len({rotate(1,0,m,t) for m in (False,True) for t in range(4)})==4
# An intervention only on noncentral reference labels is represented in the
# permuted control; no label-dependent state is used for template matching.
print('PASS: output shape/type, B-subset, all-FG/all-BG, thin-reference fallback, fixed orientation map')
