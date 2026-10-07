#!/usr/bin/env python3
"""Execute the released basis constructor once with the pinned CPU backbone."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--released-source', type=Path, required=True)
    parser.add_argument('--released-source-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    assert sha(args.released_source) == args.released_source_sha256
    assert not args.out.exists(), 'Never overwrite a prior basis or receipt'
    args.out.mkdir(parents=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import einops
    from ics.cpu100.encoder import _TimmCPUForward
    from ics.astra300.common import array_hash

    manifest = json.loads(args.manifest.read_text())
    row = manifest['rows'][0]
    assert sha(row['feature_pack']) == row['sha256']
    with np.load(row['feature_pack'], allow_pickle=False) as pack:
        producer = json.loads(pack['producer_json'].item())
    start = time.monotonic()
    backbone = _TimmCPUForward(args.model_dir, producer, args.threads)

    class Encoder:
        calls = 0
        input_sha256 = None

        def to(self, device):
            assert str(device) == 'cpu'
            return self

        def get_intermediate_layers(self, x, n=1, reshape=True):
            assert n == 1 and reshape and x.shape == (1, 3, 1024, 1024)
            assert x.dtype == torch.float32 and x.device.type == 'cpu'
            expected = (-torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225]))[None, :, None, None]
            assert torch.equal(x, expected.expand_as(x))
            backbone.check_frozen()
            self.calls += 1
            self.input_sha256 = array_hash(x.numpy())
            tokens = backbone._model.forward_features(x)
            assert tokens.shape == (1, 4101, 1024) and tokens.dtype == torch.float32
            backbone.check_frozen()
            return (tokens[:, 5:].transpose(1, 2).reshape(1, 1024, 64, 64),)

    text = args.released_source.read_text()
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FoRIS')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_build_positional_basis')
    # Compile the untouched released method, including its no_grad decorator.
    namespace = dict(torch=torch, F=F, einops=einops)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(args.released_source), 'exec'), namespace)
    encoder = Encoder()
    host = SimpleNamespace(image_size=1024, svd_components=500, encoder=encoder)
    with torch.inference_mode(), torch.autocast('cpu', enabled=False):
        matrix = namespace['_build_positional_basis'](host, 'cpu').numpy().copy()
    assert encoder.calls == 1 and matrix.shape == (1024, 500) and matrix.dtype == np.float32
    np.savez_compressed(args.out / 'basis.npz', basis=matrix)
    defect = float(torch.linalg.eigvalsh(torch.from_numpy(matrix).double().T @ torch.from_numpy(matrix).double()).sub(1).abs().max())
    receipt = dict(state='NATIVE_BASIS_FROZEN', source_input='normalized_black_image',
                   source='unchanged released FoRIS _build_positional_basis with actual frozen CPU FP32 model',
                   source_sha256=args.released_source_sha256,
                   constructor_sha256=hashlib.sha256(ast.get_source_segment(text, method).encode()).hexdigest(),
                   basis_array_sha256=array_hash(matrix), basis_file_sha256=sha(args.out / 'basis.npz'),
                   basis_shape=[1024, 500], dtype='float32', model_input_side=1024,
                   real_encoder_execution=True, synthetic_test_only=False,
                   actual_encoder_forwards=encoder.calls, normalized_input_array_sha256=encoder.input_sha256,
                   checkpoint_sha256=backbone.binding['model_assets']['checkpoint_sha256'],
                   config_sha256=backbone.binding['model_assets']['config_sha256'],
                   full_model_binding=backbone.binding, gram_spectral_defect=defect,
                   matrix_preserved_without_QR_or_column_rescaling=True,
                   query_GT_read=False, elapsed_seconds=time.monotonic()-start)
    (args.out / 'producer.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key:receipt[key] for key in ('state','actual_encoder_forwards','elapsed_seconds','gram_spectral_defect','basis_file_sha256')}, indent=2))


if __name__ == '__main__':
    main()
