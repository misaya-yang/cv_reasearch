"""Scoped, single-axis full-source control; not a new segmentation method.

The pinned FoRIS function normalizes [1,D,H,W] support features along H.
This context retains its exact code except the verified `ref_m` normalize
dimension. Axis2 is an exact replica control; axis1 is a channel-normalized
correspondence control. All grouping, priors and refinement stay unchanged.
"""
import ast
from contextlib import contextmanager
import inspect
import textwrap
import types


def source_candidate_function(original, axis):
    if axis not in (1,2):raise ValueError('Only declared source or channel axis')
    fn=getattr(original,'__func__',original)
    source=getattr(fn,'_fixture_source',None) or inspect.getsource(fn)
    tree=ast.parse(textwrap.dedent(source))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    if function.name!='_locate_candidates':raise ValueError('Wrong complete source method')
    changed=0
    for node in ast.walk(function):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name) or node.targets[0].id!='ref_m':continue
        call=node.value
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or not isinstance(call.func.value,ast.Name) or call.func.value.id!='F' or call.func.attr!='normalize':raise ValueError('Unexpected native reference-normalization expression')
        if len(call.args)!=1 or ast.unparse(call.args[0])!='ref_feats[0:1, m]':raise ValueError('Unexpected native support shape selection')
        keyword=next((x for x in call.keywords if x.arg=='dim'),None)
        if keyword is None or not isinstance(keyword.value,ast.Constant) or keyword.value.value!=2:raise ValueError('Native source no longer uses declared H normalization')
        keyword.value=ast.Constant(value=axis);changed+=1
    if changed!=1:raise ValueError('Exactly one pinned source assignment required')
    ast.fix_missing_locations(tree)
    scope=dict(fn.__globals__)
    exec(compile(tree,inspect.getsourcefile(fn) or '<source-axis-control>','exec'),scope)
    return scope[function.name]


@contextmanager
def candidate_normalization_axis(host,axis):
    existed='_locate_candidates' in host.__dict__
    previous=host.__dict__.get('_locate_candidates')
    changed=source_candidate_function(host._locate_candidates,axis)
    host._locate_candidates=types.MethodType(changed,host)
    try:yield
    finally:
        if existed:host._locate_candidates=previous
        else:delattr(host,'_locate_candidates')
