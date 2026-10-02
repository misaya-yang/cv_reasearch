#!/usr/bin/env python3
"""Evaluator-only exact class-IoU oracle for a FROZEN finite output bank.

Not a reachable method and not an upper bound for arbitrary new algorithms.
For each class use Dinkelbach's finite fractional optimization, maximizing
sumI/sumU, rather than selecting each episode's best I/U and mislabelling it.
"""
import argparse
import itertools
import json
from pathlib import Path


def class_bank_oracle(bank):
    if not bank or any(not values for values in bank):
        raise ValueError('Nonempty frozen per-episode output banks required')
    if any(not (0 <= i <= u and u > 0) for values in bank for i,u in values):
        raise ValueError('Foreground-present class-IoU bank requires 0<=I<=U and U>0')
    ratio=0.;visited=set()
    while True:
        chosen=tuple(max(range(len(v)),key=lambda j:v[j][0]-ratio*v[j][1]) for v in bank)
        intersection=sum(v[j][0] for v,j in zip(bank,chosen))
        union=sum(v[j][1] for v,j in zip(bank,chosen))
        new_ratio=intersection/union
        if chosen in visited or abs(intersection-ratio*union) <= 1e-10*max(union,1):
            return dict(iou=new_ratio,intersection=intersection,union=union,indices=list(chosen),
                        scope='Strict maximum for this saved finite bank/class ratio only; GT-dependent diagnostic')
        visited.add(chosen);ratio=new_ratio


def self_check():
    bank=[[(2,3),(40,100)],[(8,20),(3,4)],[(1,8),(10,20)]]
    brute=max(sum(bank[e][j][0] for e,j in enumerate(js))/sum(bank[e][j][1] for e,j in enumerate(js))
              for js in itertools.product(*[range(len(v)) for v in bank]))
    result=class_bank_oracle(bank)
    assert abs(result['iou']-brute)<1e-12
    per_episode=[max(range(len(v)),key=lambda j:v[j][0]/v[j][1]) for v in bank]
    naive=sum(v[j][0] for v,j in zip(bank,per_episode))/sum(v[j][1] for v,j in zip(bank,per_episode))
    assert result['iou'] >= naive
    print(json.dumps(dict(state='PASSED',finite_bank_exact=True,bruteforce_equal=True,
                         best_class_iou=result['iou'],scope='Synthetic CPU evaluator test, not task gain')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check();return
    if not a.input or not a.out:p.error('--input and --out required')
    doc=json.loads(a.input.read_text())
    if doc.get('state')!='PREDICTIONS_FROZEN':raise ValueError('Bank must be frozen before labels/evaluator optimization')
    groups={}
    for row in doc['records']:groups.setdefault(str(row['class_id']),[]).append(row['iu_bank'])
    classes={c:class_bank_oracle(bank) for c,bank in groups.items()}
    result=dict(state='EVALUATOR_DIAGNOSTIC_COMPLETED',classes=classes,
                class_miou=100*sum(v['iou'] for v in classes.values())/len(classes),
                inference_use_forbidden=True,unrestricted_method_upper_bound=False)
    a.out.write_text(json.dumps(result))


if __name__=='__main__':main()
