"""Same class/image-disjoint probe, now retaining query-region relationships.

Only the readout feature map changes relative to cpu_selection_probe.py.
No semantic category vectors or query labels enter the feature map.
"""
import cpu_selection_probe as probe

np = probe.np
base_extract = probe.extract


def extract(t):
    base = base_extract(t)
    area = t["area"].numpy().astype(float)
    evidence = np.column_stack([
        t["back"].float().numpy(), t["cross"].float().numpy(),
        base[:, 4], t["combined"].float().numpy(),
    ])
    blocks = [base]
    for space in ("Po", "Pd"):
        x = t[space].float().numpy().astype(float)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        similarity = x @ x.T
        for temperature in (0.05, 0.1):
            kernel = np.exp((similarity - 1) / temperature)
            np.fill_diagonal(kernel, 0)
            kernel *= np.sqrt(area)[None, :]
            kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-9)
            first = kernel @ evidence
            second = kernel @ first
            blocks.extend([first, second, evidence - first])
    # Contrast to strong/weak foreground witnesses, excluding the region itself.
    x = t["Po"].float().numpy().astype(float)
    sim = x @ x.T
    for key in (t["back"].float().numpy(), t["combined"].float().numpy()):
        order = np.argsort(key)
        for count in (1, 3, 5):
            v = []
            for candidates in (order[-count:], order[:count]):
                scores = sim[:, candidates].copy()
                scores[np.arange(len(area))[:, None] == candidates[None, :]] = -1
                v.append(scores.max(axis=1))
            blocks.append(np.column_stack([v[0], v[1], v[0] - v[1]]))
    return np.column_stack(blocks)


probe.extract = extract
probe.FEATURES += [f"relation_stat_{k}" for k in range(66)]

if __name__ == "__main__":
    probe.main()
