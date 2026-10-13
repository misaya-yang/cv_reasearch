"""Check the entire working tree, including ignored files, for generated bulk assets."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
BULK_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npz', '.npy',
                 '.h5', '.hdf5', '.pkl', '.pickle', '.png', '.jpg', '.jpeg',
                 '.mp4', '.avi', '.so', '.dylib'}
GIB = 1024 ** 3


def inspect(root=ROOT):
    files, byte_count, problems = 0, 0, []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                dirs.remove(name)
                if not path.exists():
                    problems.append(f'Broken link: {path.relative_to(root)}')
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                if not path.exists():
                    problems.append(f'Broken link: {path.relative_to(root)}')
                continue
            size = path.stat().st_size
            files += 1
            byte_count += size
            if path.suffix.lower() in BULK_SUFFIXES:
                problems.append(f'Generated binary in checkout: {path.relative_to(root)}')
            if size >= GIB:
                problems.append(f'Gigabyte file in checkout: {path.relative_to(root)}')
    if (root / 'outputs').exists() or (root / 'outputs').is_symlink():
        problems.append('outputs/ has returned; place run outputs under ../cv_data/a/')
    if byte_count >= GIB:
        problems.append('Working tree is gigabyte-scale again; inspect ignored files too')
    return {'regular_files': files, 'bytes': byte_count, 'excluding': '.git and symlink targets',
            'status': 'PASS' if not problems else 'FAIL', 'problems': problems}


if __name__ == '__main__':
    result = inspect()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(bool(result['problems']))
