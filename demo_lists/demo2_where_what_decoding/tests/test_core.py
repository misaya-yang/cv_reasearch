"""CPU-only checks of the decoding rule on synthetic posteriors.  Run: python tests/test_core.py"""
import itertools, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mdecode import gains, value, decode, SoftStats, fit_labelfree


def posteriors(n, C, tail=2.0, sharp=3.0, seed=0):
    """Long-tailed synthetic posteriors: class c has prior ~ (c+1)^-tail; logits = sharp * onehot(z) + noise."""
    g = torch.Generator().manual_seed(seed)
    prior = torch.arange(1, C + 1).double() ** (-tail); prior /= prior.sum()
    z = torch.multinomial(prior, n, replacement=True, generator=g)
    logits = torch.randn(n, C, generator=g, dtype=torch.float64) * 1.5 + sharp * torch.nn.functional.one_hot(z, C) + prior.log()
    return torch.softmax(logits, 1).T.contiguous()          # (C, n)


def sweep_factory(p):
    def sweep(a, b):
        s = SoftStats(p.shape[0]); s.add(p, decode(p, a.double(), b.double())); return s
    return sweep


def expected(metric, p, pred):
    s = SoftStats(p.shape[0]); s.add(p, pred); return float(value(metric, *s.fractions(0.0)))


def test_never_worse_than_argmax_in_expectation():
    for metric in ("miou", "mdice", "fwiou", "macc"):
        for seed in range(10):
            p = posteriors(20000, 8, seed=seed)
            a, b, hist = fit_labelfree(sweep_factory(p), 8, metric=metric, iters=6, smooth=0.0)
            assert max(hist) >= hist[0] - 1e-12
            assert expected(metric, p, decode(p, a.double(), b.double())) >= expected(metric, p, p.argmax(0)) - 1e-12


def test_close_to_brute_force_optimum_on_tiny_problem():
    """9 pixels x 3 classes: enumerate all 3^9 labelings. The first-order rule is exact only for many small
    pixels, so we only require it to recover most of the gap between argmax and the true optimum."""
    gaps = []
    for seed in range(20):
        p = posteriors(9, 3, tail=1.0, sharp=1.0, seed=seed)
        best = max(expected("miou", p, torch.tensor(lab)) for lab in itertools.product(range(3), repeat=9))
        a, b, _ = fit_labelfree(sweep_factory(p), 3, iters=6, smooth=0.0)
        ours = expected("miou", p, decode(p, a.double(), b.double())); base = expected("miou", p, p.argmax(0))
        assert ours <= best + 1e-12 and ours >= base - 1e-12
        if best - base > 1e-9: gaps.append((ours - base) / (best - base))
    print("  fraction of the argmax->optimum gap recovered on tiny problems:", round(sum(gaps) / len(gaps), 3))


def test_local_optimality_single_pixel_flips():
    """At the returned solution, no single-pixel relabeling should improve the expected mIoU by more than a hair."""
    p = posteriors(3000, 5, seed=3)
    a, b, _ = fit_labelfree(sweep_factory(p), 5, iters=8, smooth=0.0)
    pred = decode(p, a.double(), b.double()); base = expected("miou", p, pred); worst = 0.0
    for i in range(0, 3000, 7):
        for c in range(5):
            q = pred.clone(); q[i] = c; worst = max(worst, expected("miou", p, q) - base)
    print("  largest single-pixel improvement left:", f"{worst:.2e}", "of", round(base, 4))
    assert worst < 2e-4


def test_macc_is_prior_normalised_argmax():
    p = posteriors(5000, 6, seed=1)
    a, b, _ = fit_labelfree(sweep_factory(p), 6, metric="macc", iters=2, smooth=0.0)
    assert torch.equal(decode(p, a.double(), b.double()), (p / p.mean(1, keepdim=True)).argmax(0))


def test_real_gain_when_labels_follow_the_posterior():
    """Calibrated world: labels are sampled from p. The realised (not expected) mIoU must go up on average."""
    deltas = []
    for seed in range(10):
        C = 12; p = posteriors(200000, C, seed=seed)
        y = torch.multinomial(p.T, 1, generator=torch.Generator().manual_seed(100 + seed))[:, 0]
        def miou(pred):
            conf = torch.bincount(y * C + pred, minlength=C * C).view(C, C).double()
            return float((conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag())).mean())
        a, b, _ = fit_labelfree(sweep_factory(p), C, iters=5, smooth=0.0)
        deltas.append(miou(decode(p, a.double(), b.double())) - miou(p.argmax(0)))
    print("  realised mIoU gain over argmax, calibrated synthetic world:", [round(100 * d, 2) for d in deltas])
    assert min(deltas) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name); fn()
    print("ALL PASSED")
