"""Bind existing cache episodes to real COCO JPEG bytes, RGB128 and original H/W.

No query labels or prediction arrays are opened. Archives already present on the
server are read by exact manifest members; no dataset download or bulk extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--smoke-manifest', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'rgb128').mkdir()
    source = json.loads(args.manifest.read_text())
    rows = source if isinstance(source, list) else source['episodes']
    query_members = {row['query'] for row in rows}
    images, archives = {}, {}
    try:
        bound = []
        for index, row0 in enumerate(rows):
            row = dict(row0)
            for role in ('support', 'query'):
                member = str(row[role])
                match = re.fullmatch(r'(train2014|val2014)/COCO_\1_(\d{12})\.jpg', member)
                if match is None:
                    raise ValueError('Cannot establish canonical COCO photo identity: ' + member)
                split, photo = match.groups()
                if member not in images:
                    if split not in archives:
                        archives[split] = zipfile.ZipFile(args.archive_root / (split + '.zip'))
                    archive = archives[split]
                    info = archive.getinfo(member)
                    payload = archive.read(member)
                    with Image.open(io.BytesIO(payload)) as image:
                        width, height = image.size
                        metadata = dict(member=member, archive=str(Path(archive.filename).resolve()),
                                        member_crc32=info.CRC, jpeg_sha256=hashlib.sha256(payload).hexdigest(),
                                        original_hw=[height, width])
                        if member in query_members:
                            canonical = image.convert('RGB').resize((1024, 1024), Image.Resampling.BILINEAR)
                            rgb = np.asarray(canonical.resize((128, 128), Image.Resampling.BILINEAR)).copy()
                            path = args.out / 'rgb128' / (photo + '.npz')
                            np.savez_compressed(path, rgb=rgb)
                            metadata.update(rgb_export=str(path.resolve()), rgb_sha256=sha(path))
                    images[member] = metadata
                row[role + '_photo_id'] = 'coco2014:' + photo
                row[role + '_image_hw'] = images[member]['original_hw']
            row['query_rgb_export'] = images[row['query']]['rgb_export']
            for key in ('feature_export', 'packet_export', 'base_field_export'):
                if row.get(key) and not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
            bound.append(row)
            if (index + 1) % 100 == 0:
                print(json.dumps(dict(bound=index + 1, total=len(rows))), flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    original_smoke = json.loads(args.smoke_manifest.read_text())
    smoke_rows = original_smoke if isinstance(original_smoke, list) else original_smoke['episodes']
    identities = [(row['key'], row['support'], row['query']) for row in smoke_rows]
    lookup = {}
    for row in bound:
        lookup.setdefault((row['key'], row['support'], row['query']), row)
    smoke = [lookup[key] for key in identities]
    (args.out / 'rows.json').write_text(json.dumps(bound, indent=2) + '\n')
    (args.out / 'smoke4.json').write_text(json.dumps(smoke, indent=2) + '\n')
    receipt = dict(occurrences=len(bound), unique_images=len(images),
                   source_manifest=str(args.manifest), source_manifest_sha256=sha(args.manifest),
                   source_smoke_sha256=sha(args.smoke_manifest),
                   image_bindings=images, query_GT_opened=False,
                   RGB_producer='Exact manifest JPEG bytes, PIL RGB->1024 bilinear->128 bilinear',
                   query_geometry='Original JPEG header H/W, not canonical-grid geometry',
                   exposure='Reused public600 development episodes; not independent confirmation')
    (args.out / 'binding.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(state='BOUND', occurrences=len(bound), unique_images=len(images),
                          smoke_occurrences=len(smoke), query_GT_opened=False)), flush=True)


if __name__ == '__main__':
    main()
