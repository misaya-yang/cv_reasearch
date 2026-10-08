#!/usr/bin/env python3
"""Check the fixed primary, both no-drop comparisons and distinct dev/val gates."""
import copy
from candidate_promotion import DATASETS,development,validation


def report(split):
    result=dict(split_role=split,sampling_seed={'dev':1,'val':2}[split],
                prepared_state='FRESH_MANIFEST_FROZEN',n=10000,datasets={})
    for name in DATASETS:
        pairs={b:{'candidate':dict(delta_pp=0.,ci95_pp=[-.1,.1])} for b in ('parent','initial')}
        result['datasets'][name]=dict(primary=dict(n=2000,paired=pairs,
            bootstrap=dict(unit='query photo within fold and class',repetitions=10000)))
    return result


def set_gain(value,name,delta,lower,baseline='parent'):
    value['datasets'][name]['primary']['paired'][baseline]['candidate']=dict(delta_pp=delta,ci95_pp=[lower,delta+1])


def main():
    dev=report('dev')
    for name in ('paco_part','suim','lvis'):set_gain(dev,name,1.2,.2)
    selected=development(dev,'candidate','parent','initial')
    assert selected['passed'] and selected['primary_dataset']=='paco_part'
    largest=copy.deepcopy(dev);set_gain(largest,'lvis',2.,-1.)
    assert development(largest,'candidate','parent','initial')['primary_dataset']=='lvis'
    val=report('val');set_gain(val,'paco_part',.25,.01)
    set_gain(val,'suim',2.,1.)
    accepted=validation(val,selected)
    assert accepted['passed'] and accepted['primary_dataset']=='paco_part'
    # Validation neither switches primary nor repeats the development+1 line.
    set_gain(val,'paco_part',.25,0.)
    assert not validation(val,selected)['passed']
    for baseline in ('parent','initial'):
        bad=copy.deepcopy(dev);set_gain(bad,'coco',-.301,-1.,baseline)
        assert not development(bad,'candidate','parent','initial')['passed']
    boundary=copy.deepcopy(dev);set_gain(boundary,'coco',-.3,-1.)
    assert development(boundary,'candidate','parent','initial')['passed']
    bad=copy.deepcopy(dev)
    for name in ('paco_part','suim','lvis'):set_gain(bad,name,1.,0.)
    assert not development(bad,'candidate','parent','initial')['passed']
    for mutation in ('partial','seed','unit'):
        bad=copy.deepcopy(dev)
        if mutation=='partial':bad['datasets']['lvis']['primary']['n']=1999
        if mutation=='seed':bad['sampling_seed']=0
        if mutation=='unit':bad['datasets']['suim']['primary']['bootstrap']['unit']='episode within fold and class'
        try:development(bad,'candidate','parent','initial');raise AssertionError('Invalid cohort/protocol promoted')
        except ValueError:pass
    print('PASS: fixed primary/ties, both no-drop comparisons, strict positive CI, lower validation gain and invalid protocol rejection')


if __name__=='__main__':main()
