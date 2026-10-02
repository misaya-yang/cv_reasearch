"""Export bounded evidence snapshots without copying tensors or per-image dumps.

Run from any directory: python3 tools/export_evidence.py
Only stdlib is required. Originals are never modified. Exported JSON explicitly
marks omitted arrays; these snapshots cannot replace raw-data statistical checks.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence' / 'compact'
SKIP = {'assets', '.git', '__pycache__', 'foris_source', 'crf_source'}
MAX_EXPORT = 200_000


def compact(value):
    if isinstance(value, list):
        if len(value) > 64:
            return {'_omitted': 'array exceeds 64 items', 'item_count': len(value)}
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def main():
    manifest = []
    for source in sorted((ROOT / 'demo_lists').rglob('*')):
        rel = source.relative_to(ROOT)
        if not source.is_file() or SKIP.intersection(rel.parts):
            continue
        is_result = ('results' in rel.parts or source.name.endswith(('_report.json', '_results.json'))
                     or (source.name == 'results.json' and 'official_cpu_validation' in rel.parts))
        if not is_result or source.suffix not in {'.json', '.csv', '.log', '.txt'}:
            continue
        # Per-image/prompt artifacts have no independent aggregate interpretation.
        if any(re.fullmatch(r'(?:image|prompt)_\d+.*', part) for part in rel.parts):
            continue
        raw = source.read_bytes()
        item = {'source': rel.as_posix(), 'source_bytes': len(raw),
                'source_sha256': hashlib.sha256(raw).hexdigest()}
        target = OUT / rel
        if source.suffix == '.json':
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                item['status'] = 'skipped: invalid or incomplete JSON'
                manifest.append(item)
                continue
            payload = json.dumps({'export_note': 'Arrays longer than 64 items are omitted; retained values are unchanged. No metrics are recomputed.',
                                  **item, 'data': compact(value)}, ensure_ascii=False, indent=2).encode() + b'\n'
        else:
            payload = raw
        if len(payload) > MAX_EXPORT:
            item['status'] = 'skipped: compact file exceeds 200 KB'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            item.update(status='exported', export=target.relative_to(ROOT).as_posix(),
                        export_bytes=len(payload), export_sha256=hashlib.sha256(payload).hexdigest())
        manifest.append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'evidence' / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {sum(x["status"] == "exported" for x in manifest)} of {len(manifest)} aggregate candidates.')


if __name__ == '__main__':
    main()
