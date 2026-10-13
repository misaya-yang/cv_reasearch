"""Read compact research evidence using the Python standard library only.

No model, raw feature, image, query GT, external archive or network is accessed.
Historical path strings are identifiers only; they are never opened.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import json


def pooled(path):
    totals = defaultdict(lambda: [0, 0])
    counts = defaultdict(int)
    for line in gzip.open(path, 'rt', encoding='utf-8'):
        r = json.loads(line)
        counts[r['dataset']] += 1
        for arm, (intersection, union) in r['iu']['cli1024'].items():
            key = (r['dataset'], arm, str(r['fold']), str(r['class_id']))
            totals[key][0] += intersection
            totals[key][1] += union
    values = defaultdict(list)
    for (dataset, arm, fold, category), (intersection, union) in totals.items():
        values[(dataset, arm)].append(100 * intersection / max(union, 1))
    return counts, {key: sum(v) / len(v) for key, v in values.items()}


def summarize(root):
    folder = root / 'evidence/research'
    n, scores = pooled(folder / 'c1_confirm2136_iu.jsonl.gz')
    c1 = {d: {'n': n[d], 'FoRIS': scores[d, 'FoRIS'],
              'candidate': scores[d, 'candidate'],
              'delta_pp': scores[d, 'candidate'] - scores[d, 'FoRIS']} for d in sorted(n)}
    n, scores = pooled(folder / 'b1_common_k3_iu.jsonl.gz')
    b1 = {}
    for d in sorted(n):
        k1, k3 = scores[d, 'K1/IND0'], scores[d, 'K3/IND0']
        o1, o3 = scores[d, 'K1/IND0/oracle_threshold'], scores[d, 'K3/IND0/oracle_threshold']
        b1[d] = {'n': n[d], 'K1': k1, 'K3': k3, 'natural_delta_pp': k3-k1,
                 'K1_oracle': o1, 'K3_oracle': o3, 'oracle_delta_pp': o3-o1}
    return {'metric': 'CLI1024; pooled I/U per observed (fold,class), equal observed-slot mean',
            'scope': 'Saved integer reaggregation only; no new inference or GT reads; CI not recomputed',
            'C1': c1, 'B1_common_K3': b1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = summarize(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result['metric'])
    print('\nC1 fixed candidate vs FoRIS (pp)')
    print('dataset              n      FoRIS  candidate      delta')
    for d, r in result['C1'].items():
        print(f"{d:18} {r['n']:5d} {r['FoRIS']:10.3f} {r['candidate']:10.3f} {r['delta_pp']:+10.3f}")
    print('\nB1 K3 vs K1, common queries (diagnostic; not a 1-shot result)')
    print('dataset              n   natural delta   oracle delta')
    for d, r in result['B1_common_K3'].items():
        print(f"{d:18} {r['n']:5d} {r['natural_delta_pp']:+15.3f} {r['oracle_delta_pp']:+14.3f}")


if __name__ == '__main__':
    main()
