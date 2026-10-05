"""Scoped reuse of the existing native positional basis; no downloads or GT.

Actual INSID3/FoRIS source uses a normalized black image. Do not substitute a
Gaussian/pair-derived basis or change a third-party file. Other processes are
unaffected; this context restores the constructor method in this process.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import torch


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_native_basis(path,dimension=1024,rank=500):
    doc=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(doc,dict) or doc.get('state')!='NATIVE_BASIS_FROZEN' or doc.get('source_input')!='normalized_black_image':
        raise ValueError('Expected explicitly source-faithful frozen native basis')
    u=doc['basis']
    if u.shape!=(dimension,rank) or not torch.isfinite(u).all():raise ValueError('Invalid basis shape/values')
    # This is the original host's numerical SVD output, not a mathematical QR
    # basis created by our method. Actual CUDA FP32 native U500 has spectral
    # Gram defect~6.11e-4. Re-normalizing it would change the strong baseline.
    # Check finite, well-conditioned near-unit columns and record the defect;
    # do not falsely require the CPU fixture's much tighter exact-QR tolerance.
    eigenvalues=torch.linalg.eigvalsh(u.double().T@u.double())
    defect=float((eigenvalues-1).abs().max())
    if defect>=.01:raise ValueError('Native numerical basis is not near-unit/well-conditioned')
    return u,dict(path=str(path),sha256=sha256_file(path),source_input=doc['source_input'],
        native_numerical_gram_spectral_defect=defect,exact_orthogonality_claim=False,
        matrix_preserved_without_QR_or_column_rescaling=True)


@contextmanager
def reuse_native_basis(host_class,path):
    if not path:
        yield None
        return
    u,receipt=load_native_basis(path)
    original=host_class._build_positional_basis
    def frozen(this,device):
        if this.svd_components!=u.shape[1]:raise ValueError('Basis rank mismatch; do not silently change native rank')
        return u.to(device).clone()
    host_class._build_positional_basis=frozen
    try:yield receipt
    finally:host_class._build_positional_basis=original
