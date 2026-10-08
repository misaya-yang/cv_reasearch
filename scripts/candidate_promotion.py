"""Fixed development/validation decisions from sealed score_runs statistics.

This helper reads no labels, selects no variant and starts no experiment.
Validation receives the primary dataset frozen by development. Passing these
stage rules does not establish the final five-dataset FoRIS result.
"""
import math

DATASETS=('coco','lvis','pascal_part','paco_part','suim')
PRIMARY_ORDER=('paco_part','suim','lvis')


def contrasts(report,candidate,parent,initial,split):
    if (report.get('split_role')!=split or report.get('sampling_seed')!={'dev':1,'val':2}[split] or
        report.get('prepared_state')!='FRESH_MANIFEST_FROZEN' or report.get('n')!=10000 or
        set(report.get('datasets',{}))!=set(DATASETS)):
        raise ValueError('Require all five complete fresh development/validation scores')
    result={}
    for name in DATASETS:
        stats=report['datasets'][name]['primary']
        bootstrap=stats['bootstrap']
        if stats['n']!=2000 or bootstrap['unit']!='query photo within fold and class' or bootstrap['repetitions']!=10000:
            raise ValueError('Require2000 episodes and the fixed grouped paired intervals per dataset')
        result[name]={}
        for label,baseline in (('parent',parent),('initial',initial)):
            item=stats['paired'][baseline][candidate]
            delta=float(item['delta_pp']);lo,hi=map(float,item['ci95_pp'])
            if not all(math.isfinite(x) for x in (delta,lo,hi)) or lo>hi:
                raise ValueError('Invalid paired contrast')
            result[name][label]=dict(delta_pp=delta,ci95_pp=[lo,hi])
    return result


def development(report,candidate,parent='rcg.fine',initial='rcg.fine'):
    values=contrasts(report,candidate,parent,initial,'dev')
    drops=[name for name in DATASETS if any(v['delta_pp']<-.3 for v in values[name].values())]
    eligible=[name for name in PRIMARY_ORDER if values[name]['parent']['delta_pp']>=1 and
              values[name]['parent']['ci95_pp'][0]>0]
    # Largest observed development gain; fixed order resolves exact ties.
    primary=max(PRIMARY_ORDER,key=lambda name:values[name]['parent']['delta_pp']) if eligible and not drops else None
    return dict(stage='dev',passed=primary is not None,candidate=candidate,parent=parent,initial=initial,
                primary_dataset=primary,drop_failures=drops,contrasts=values)


def validation(report,development_decision):
    if development_decision.get('stage')!='dev' or development_decision.get('passed') is not True:
        raise ValueError('Validation requires an already passed frozen development decision')
    candidate,parent,initial=(development_decision[k] for k in ('candidate','parent','initial'))
    primary=development_decision['primary_dataset']
    if primary not in PRIMARY_ORDER:raise ValueError('Invalid frozen primary dataset')
    values=contrasts(report,candidate,parent,initial,'val')
    drops=[name for name in DATASETS if any(v['delta_pp']<-.3 for v in values[name].values())]
    passed=not drops and values[primary]['parent']['ci95_pp'][0]>0
    return dict(stage='val',passed=passed,candidate=candidate,parent=parent,initial=initial,
                primary_dataset=primary,drop_failures=drops,contrasts=values)
