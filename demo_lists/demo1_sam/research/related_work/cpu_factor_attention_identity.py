"""Bounded NumPy algebra check; not SAM correctness or GPU evidence."""
import json
from pathlib import Path
import numpy as np

def mm(a, b):
    # Use scalar contraction to bypass local NumPy/Accelerate FPE warnings.
    return np.einsum('ik,kj->ij', a, b, optimize=False)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def run_case(n, d, dh, t, r, seed):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.normal(size=shape) / np.sqrt(max(d, 1))
    (base, s, u, factors) = (rand(n, d), rng.uniform(0.2, 2, (n, 1)), rand(n, r), rand(r, d))
    x = s * base + mm(u, factors)
    (w_k, w_v, w_q) = (rand(d, dh), rand(d, dh), rand(d, dh))
    (b_k, b_v, b_q) = (rand(1, dh), rand(1, dh), rand(1, dh))
    (pe, read_q, write_k) = (rand(n, dh), rand(t, dh), rand(t, dh))
    write_v = rand(t, d)
    scale = np.sqrt(dh)
    (dense_k, dense_v) = (mm(x, w_k) + pe + b_k, mm(x, w_v) + b_v)
    dense_logits = mm(read_q, dense_k.T) / scale
    dense_read = mm(softmax(dense_logits), dense_v)
    logits = (mm(read_q, mm(base, w_k).T) * s.T + mm(mm(read_q, mm(factors, w_k).T), u.T) + mm(read_q, pe.T) + mm(read_q, b_k.T)) / scale
    a = softmax(logits)
    read = mm(a * s.T, mm(base, w_v)) + mm(mm(a, u), mm(factors, w_v)) + b_v
    dense_write_logits = mm(mm(x, w_q) + pe + b_q, write_k.T) / scale
    write_logits = (s * mm(mm(base, w_q), write_k.T) + mm(u, mm(mm(factors, w_q), write_k.T)) + mm(pe, write_k.T) + mm(b_q, write_k.T)) / scale
    (dense_write, write) = (mm(softmax(dense_write_logits), write_v), mm(softmax(write_logits), write_v))
    errors = {'read_logits': float(np.max(np.abs(logits - dense_logits))), 'read_output': float(np.max(np.abs(read - dense_read))), 'write_logits': float(np.max(np.abs(write_logits - dense_write_logits))), 'write_output': float(np.max(np.abs(write - dense_write)))}
    assert all(np.isfinite(v) for v in errors.values())
    assert max(errors.values()) < 1e-12, errors
    return dict(n=n, d=d, dh=dh, tokens=t, rank=r, seed=seed, errors=errors)
if __name__ == '__main__':
    cases = [run_case(*args) for args in [(13, 16, 8, 7, 0, 1), (257, 64, 32, 7, 8, 2), (513, 256, 16, 7, 48, 3), (513, 256, 16, 9, 98, 4), (31, 32, 8, 11, 40, 5)]]
    out = dict(dtype='float64', scope='NumPy algebra only, one attention head, no trained SAM or timing', threshold=1e-12, passed=True, cases=cases)
    path = Path(__file__).with_name('cpu_factor_attention_identity_results.json')
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))
