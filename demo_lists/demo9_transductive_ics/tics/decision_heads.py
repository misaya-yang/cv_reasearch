"""A ladder of read-outs that turn pair evidence into a mask. Every rung starts as the host's own mask (its last
layer is zero and the host's logit is added), so a rung can only move away from the host where the data say so.

Rungs, from the least to the most capacity (inputs [B, C, H, W] evidence maps, one of them the host's score):
  cut      a * host + b                      a global re-cut of the host score (2 weights)
  pixel    linear in the C maps              per-patch fusion of the evidence
  mlp      per-patch MLP                     non-linear fusion, no neighbours
  context  per-patch MLP + image statistics  the cut may depend on how the evidence is distributed in this query
  conv     dilated 3x3 convolutions          neighbours and shape
  convctx  conv + image statistics
  unroll   convctx, then steps that see how each patch relates to the current mask through the query's own
           feature neighbours (the query re-estimates its target and background from its own decision)
The number of epochs is chosen on classes held out of the fit (never on the classes that are scored).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

RUNGS = ("cut", "pixel", "mlp", "context", "conv", "convctx", "unroll")
GRAPH = 5  # maps made from the neighbour graph and the current mask


def dihedral(x, y, k, pair=None):
    """One of the 8 symmetries of the square applied to maps x [B, C, H, W] and targets y [B, H, W].
    pair: indices of two channels that describe the horizontal and the vertical direction; a transpose swaps them."""
    if k & 1:
        x, y = x.flip(3), y.flip(2)
    if k & 2:
        x, y = x.flip(2), y.flip(1)
    if k & 4:
        x, y = x.transpose(2, 3), y.transpose(1, 2)
        if pair is not None:
            idx = list(range(x.shape[1]))
            idx[pair[0]], idx[pair[1]] = pair[1], pair[0]
            x = x[:, idx]
    return x, y


def undo(x, k):
    """The inverse of `dihedral` for maps [B, C, H, W] (no channel pair)."""
    if k & 4:
        x = x.transpose(2, 3)
    if k & 2:
        x = x.flip(2)
    if k & 1:
        x = x.flip(3)
    return x


def graph_maps(p, idx, sim, tau=0.05):
    """How each patch relates to the current mask through its feature neighbours in the same image.
    p [B, N] in [0, 1]; idx, sim [B, N, K]. Returns [B, GRAPH, N]: nearest target-like neighbour, nearest
    background-like neighbour, similarity-weighted and plain share of target among the neighbours, the difference."""
    B, N, K = idx.shape
    pn = p.gather(1, idx.reshape(B, -1)).view(B, N, K)
    near_t, near_b = (sim * pn).amax(2), (sim * (1 - pn)).amax(2)
    w = torch.softmax(sim / tau, 2)
    return torch.stack([near_t, near_b, (w * pn).sum(2), pn.mean(2), near_t - near_b], 1)


def agnostic(x, idx, sim):
    """Evidence that does not depend on how the host scores: its mask, the relations to the reference, the layer
    margins, and how each patch relates to the host's mask through the query's neighbour graph. x [B, 24, H, W] in
    the layout of scripts/decision_fit.py (score first, edge maps at 14 and 15, layer margins from 16); returns
    [B, 20, H, W]: the mask, fg, bg, nn_label, cycle, the two edge maps (at 5 and 6), 8 layer margins, GRAPH maps."""
    m = (x[:, :1].float() > 0.5).float()
    g = graph_maps(m.flatten(1), idx.long(), sim.float()).view(len(x), GRAPH, *x.shape[-2:])
    return torch.cat([m, x[:, [4, 5, 6, 7, 14, 15]].float(), x[:, 16:24].float(), g], 1)


def body(rung, d, width):
    if rung == "pixel":
        return nn.Conv2d(d, 1, 1)
    if rung in ("mlp", "context"):
        return nn.Sequential(nn.Conv2d(d, width, 1), nn.ReLU(), nn.Conv2d(width, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))
    layers, c = [], d
    for dil in (1, 2, 4, 1):
        layers += [nn.Conv2d(c, width, 3, padding=dil, dilation=dil), nn.GroupNorm(8, width), nn.ReLU()]
        c = width
    return nn.Sequential(*layers, nn.Dropout2d(0.1), nn.Conv2d(width, 1, 1))


def zero_last(net):
    last = net if isinstance(net, nn.Conv2d) else net[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


class Head(nn.Module):
    def __init__(self, rung, channels, host, width=48, steps=2):
        super().__init__()
        self.rung, self.host = rung, host
        self.scale = nn.Parameter(torch.tensor(8.0))  # host logit = scale * (host - 0.5): its own cut at start
        self.shift = nn.Parameter(torch.zeros(()))
        self.ctx = rung in ("context", "convctx", "unroll")
        d = channels * (4 if self.ctx else 1)  # each map, plus its image mean, its image maximum and its mean over the host mask
        self.body = None if rung == "cut" else zero_last(body("convctx" if rung == "unroll" else rung, d, width))
        self.more = nn.ModuleList([zero_last(body("conv", d + GRAPH + 1, width)) for _ in range(steps - 1)]) if rung == "unroll" else None

    def forward(self, x, raw_host, graph=None, k=0):
        """x: standardised maps; raw_host: the host score in [0, 1], [B, H, W]; graph: (idx, sim) of the query's
        feature neighbours in the untransformed layout, k the symmetry x was put through. Returns the list of logits
        [B, H, W] of every step (one step unless the rung is `unroll`)."""
        base = self.scale * (raw_host - 0.5) + self.shift
        if self.body is None:
            return [base]
        if self.ctx:
            m = (raw_host > 0.5).float()[:, None]
            inside = (x * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp_min(1.0)
            stats = [x.mean((2, 3), keepdim=True), x.amax((2, 3), keepdim=True), inside]
            x = torch.cat([x] + [s.expand_as(x) for s in stats], 1)
        out = [base + self.body(x)[:, 0]]
        for net in self.more or []:
            p = out[-1].sigmoid()[:, None]
            g = graph_maps(undo(p, k).flatten(1), *graph).view(len(p), GRAPH, *p.shape[-2:])
            g = dihedral(g, p[:, 0], k)[0]
            out.append(out[-1] + net(torch.cat([x, g, p], 1))[:, 0])
        return out


def loss_fn(logit, y):
    p = logit.sigmoid()
    soft = 1 - (p * y).sum((1, 2)) / (p + y - p * y).sum((1, 2)).clamp_min(1e-6)
    return F.binary_cross_entropy_with_logits(logit, (y > 0.5).float()) + soft.mean()


def class_miou(prob, y, cls):
    """Class mIoU of the masks prob > 0.5 against soft targets y; cls: [B] class ids."""
    mask = (prob > 0.5).float()
    i, u = (mask * y).sum((1, 2)), (y.sum((1, 2)) + mask.sum((1, 2)) - (mask * y).sum((1, 2)))
    ids = cls.unique()
    return float(torch.stack([i[cls == c].sum() / u[cls == c].sum().clamp_min(1e-6) for c in ids]).mean() * 100)


def fit(rung, X, Y, cls, host, epochs=30, batch=32, lr=2e-3, seeds=2, val_classes=0.2, pair=None, width=48, dev="cpu",
        graph=None, steps=2):
    """Fits one rung. X [n, C, H, W] raw maps (channel `host` in [0, 1]); Y [n, H, W] targets; cls [n] class ids;
    graph (idx [n, N, K], sim [n, N, K]) for `unroll`. The epoch count is the one that is best on `val_classes` of
    the classes, held out of a first fit; the returned models are refitted on everything for that many epochs.
    Returns a function (maps, graph) -> probability and a record."""
    X, Y, cls = X.to(dev), Y.to(dev), cls.to(dev)
    mu, sd = X.float().mean((0, 2, 3), keepdim=True), X.float().std((0, 2, 3), keepdim=True) + 1e-6
    part = lambda g, b: None if g is None else (g[0][b].long(), g[1][b].float())

    def run(idx, n_epochs, seed, watch=None):
        torch.manual_seed(seed)
        net = Head(rung, X.shape[1], host, width, steps).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        total = n_epochs * ((len(idx) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total, 1))
        curve = []
        for _ in range(n_epochs):
            net.train()
            perm = idx[torch.randperm(len(idx), device=dev)]
            for s in range(0, len(perm), batch):
                b = perm[s:s + batch]
                k = int(torch.randint(0, 8, ()))
                xb = X[b].float()
                x, y = dihedral(torch.cat([(xb - mu) / sd, xb[:, host:host + 1]], 1), Y[b].float(), k, pair)
                outs = net(x[:, :-1], x[:, -1], part(graph, b), k)
                loss = sum(loss_fn(o, y) for o in outs) / len(outs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if watch is not None:
                curve.append(class_miou(predict([net], X[watch], part(graph, watch)), Y[watch].float(), cls[watch]))
        return net, curve

    def predict(nets, x, g=None):
        with torch.no_grad():
            out = 0
            for net in nets:
                net.eval()
                for_one = lambda i: net((x[i:i + 64].float() - mu) / sd, x[i:i + 64, host].float(),
                                        None if g is None else (g[0][i:i + 64], g[1][i:i + 64]))[-1].sigmoid()
                out = out + torch.cat([for_one(i) for i in range(0, len(x), 64)]) / len(nets)
        return out

    every = torch.arange(len(X), device=dev)
    record = dict(rung=rung, weights=None, epochs=epochs)
    if val_classes and rung != "cut":
        ids = cls.unique()
        g = torch.Generator().manual_seed(0)
        held = ids[torch.randperm(len(ids), generator=g)[:max(1, int(round(val_classes * len(ids))))].to(dev)]
        is_val = (cls[:, None] == held[None]).any(1)
        _, curve = run(every[~is_val], epochs, 0, watch=every[is_val])
        best = max(range(len(curve)), key=lambda e: curve[e]) + 1
        host_val = class_miou((X[is_val][:, host] > 0.5).float(), Y[is_val].float(), cls[is_val])
        record.update(epochs=best, validation_curve=curve, validation_host=host_val)
    nets = [run(every, record["epochs"], seed)[0] for seed in range(seeds)]
    record["weights"] = sum(p.numel() for p in nets[0].parameters())
    record["export"] = dict(rung=rung, channels=X.shape[1], host=host, width=width, steps=steps, mu=mu.cpu(), sd=sd.cpu(),
                            states=[{k: v.cpu() for k, v in net.state_dict().items()} for net in nets])

    def model(x, g=None):
        g = None if g is None else (g[0].to(dev).long(), g[1].to(dev).float())
        return predict(nets, x.to(dev), g).cpu()
    return model, record


def restore(export, dev="cpu"):
    """A fitted read-out from its saved record: a function (maps [B, C, H, W], graph or None) -> probability [B, H, W]."""
    nets = []
    for state in export["states"]:
        net = Head(export["rung"], export["channels"], export["host"], export["width"], export["steps"]).to(dev)
        net.load_state_dict(state)
        nets.append(net.eval())
    mu, sd, host = export["mu"].to(dev), export["sd"].to(dev), export["host"]

    def model(x, g=None):
        x = x.to(dev).float()
        g = None if g is None else (g[0].to(dev).long(), g[1].to(dev).float())
        with torch.no_grad():
            return sum(net((x - mu) / sd, x[:, host], g)[-1].sigmoid() for net in nets) / len(nets)
    return model
