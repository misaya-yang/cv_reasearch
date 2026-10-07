#!/usr/bin/env python3
"""QK v3 CPU preparation from bound teacher/raw packets, no query GT access."""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def self_check(out):
    import copy
    import numpy as np
    import torch
    from ics.methods.qk_role_consensus_v3 import extract_pre_rope,predict
    torch.set_num_threads(1);torch.manual_seed(4)
    # Five orthogonal zero-mean hidden states, all realizable inputs to norm1.
    centering=torch.eye(8)-torch.ones(8,8)/8
    u,_,_=torch.linalg.svd(centering)
    hidden=u[:,:5].T.contiguous()
    norm=torch.nn.LayerNorm(8,eps=1e-5)
    actual_qkv=torch.nn.Linear(8,24,bias=False)
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.qkv=actual_qkv;self.num_heads=2
            self.q_norm=self.k_norm=torch.nn.Identity()
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.norm1=norm;self.attn=Attention()
    def vec(x):return torch.tensor([x,(1-x*x)**.5,0.,0.])
    q_target=torch.stack((torch.tensor([1.,0,0,0]),torch.tensor([-1.,0,0,0]),vec(.25),vec(.65),vec(-.15)))
    k_target=torch.stack((torch.tensor([1.,0,0,0]),torch.tensor([-1.,0,0,0]),vec(.25),vec(-.15),vec(.65)))
    desired=torch.stack((q_target[:,None].expand(-1,2,-1),k_target[:,None].expand(-1,2,-1),torch.zeros(5,2,4)),dim=1).reshape(5,24)
    with torch.no_grad():actual_qkv.weight.copy_((torch.linalg.pinv(norm(hidden))@desired).T)
    block=Block().eval().requires_grad_(False)
    source=hidden[:2]
    group=np.repeat(np.arange(3),12)
    q_hidden=hidden[2:][group]
    # Pad R state to query count for paired actual extraction; only two R points enter role evidence.
    pair=torch.stack((torch.cat((source,source[0].expand(34,-1))),q_hidden))
    qs,ks,extract=extract_pre_rope(block,pair,prefix=0)
    result=predict(qs[0,:2],ks[0,:2],qs[1],ks[1],np.array([1.,0.]),np.full((6,6),.45,np.float32),
                   source,q_hidden,source,q_hidden,(96,96),working_size=96)
    expected=group.reshape(6,6)==0
    correct={arm:bool(np.array_equal(field>.5,expected)) for arm,field in result['fields'].items()}
    if not correct['agreement'] or any(correct[x] for x in ('forward','reverse','symmetric_logits','symmetric_margin','direct_H20','direct_final')):
        raise AssertionError('Physically achievable complete-field witness does not distinguish controls')
    if any(v>.100001 for v in result['info']['field_max_changes'].values()):raise AssertionError('Residual bound violated')
    if float(np.ptp(result['evidence']['symmetric_logits']))>2e-6:
        raise AssertionError('Symmetric-score information tie missing from witness')
    expected_full=np.repeat(np.repeat(expected,16,axis=0),16,axis=1)
    complete_correct={arm:bool(np.array_equal(masks[0],expected_full) and np.array_equal(masks[1],expected_full))
                      for arm,masks in result['masks'].items()}
    if not complete_correct['agreement'] or any(complete_correct[x] for x in ('forward','reverse','symmetric_logits','symmetric_margin','direct_H20','direct_final')):
        raise AssertionError('Physical witness did not survive complete rendering')
    # Same frozen teacher and legal hidden states; query truth differs and cannot be inferred from R.
    permuted_group=np.choose(group,[1,0,2])
    negative_hidden=hidden[2:][permuted_group]
    negative_pair=torch.stack((pair[0],negative_hidden))
    nq,nk,_=extract_pre_rope(block,negative_pair,prefix=0)
    negative=predict(nq[0,:2],nk[0,:2],nq[1],nk[1],np.array([1.,0.]),np.full((6,6),.45,np.float32),
                     source,negative_hidden,source,negative_hidden,(96,96),working_size=96)
    pred=negative['fields']['agreement']>.5
    if int((pred&~expected).sum())!=12 or int((expected&~pred).sum())!=12:
        raise AssertionError('Negative class-transfer witness lost')
    symmetric_teacher=copy.deepcopy(block).eval().requires_grad_(False)
    with torch.no_grad():
        # Replace K with Q: directed observations coincide, no extra consistency signal.
        symmetric_teacher.attn.qkv.weight[8:16].copy_(symmetric_teacher.attn.qkv.weight[:8])
    sq,sk,_=extract_pre_rope(symmetric_teacher,pair,prefix=0)
    degenerate=predict(sq[0,:2],sk[0,:2],sq[1],sk[1],np.array([1.,0.]),np.full((6,6),.45,np.float32),
                       source,q_hidden,source,q_hidden,(96,96),working_size=96)
    if not np.array_equal(degenerate['fields']['agreement'],degenerate['fields']['symmetric_margin']):
        raise AssertionError('Q=K degeneracy should collapse agreement to symmetric control')
    directory=Path(out);directory.mkdir(parents=True,exist_ok=True)
    report={'state':'PHYSICAL_ARCHITECTURE_WITNESS_PASSED','actual_DINO_RGB_quality_verified':False,
            'independent_method_increment':0,'family_revision':'QK v3','fixture':'one actual frozen LN+unbiased combined QKV toy, heads2,D8',
            'positive_complete_token_masks':correct,'positive_complete_rendered_masks':complete_correct,
            'positive_group_margins':{k:[float(v[j]) for j in (0,12,24)] for k,v in result['evidence'].items()},
            'negative_false_positive_tokens':int((pred&~expected).sum()),'negative_missed_true_tokens':int((expected&~pred).sum()),
            'symmetric_logits_range':float(np.ptp(result['evidence']['symmetric_logits'])),
            'Q_equals_K_collapse_verified':True,'residual_bound_verified':True,'extract':extract,
            'real_query_GT_used':False,'query_fixture_labels_only_for_post_prediction_check':True}
    (directory/'toy_check.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(directory/'toy_fields.npz',**result['fields'])
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true');p.add_argument('--out',required=True)
    p.add_argument('--input',help='Bound NPZ of pre-RoPE rq/rk/qq/qk,cov,base,r_h20/q_h20,r_final/q_final,original_hw; never query truth')
    p.add_argument('--binding',help='JSON with input SHA and raw teacher/host producer identities')
    a=p.parse_args()
    if a.self_check:return self_check(a.out)
    if not a.input or not a.binding:p.error('Require sealed teacher packet and binding')
    import numpy as np
    import torch
    from ics.methods.qk_role_consensus_v3 import predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    binding=json.loads(Path(a.binding).read_text())
    digest=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    if binding.get('input_sha256')!=digest or binding.get('block_one_based')!=21 or binding.get('teacher_pre_RoPE') is not True:
        raise ValueError('Actual raw teacher extraction provenance must be bound')
    if binding.get('query_GT_used') is not False or not binding.get('native_producer') or not binding.get('mean_host_producer'):
        raise ValueError('GT-free teacher/MEAN producer binding required')
    out=Path(a.out)
    if out.exists():raise ValueError('Fresh output directory required')
    with np.load(a.input,allow_pickle=False) as data:
        keys=('rq','rk','qq','qk','cov','base','r_h20','q_h20','r_final','q_final','original_hw')
        v={key:data[key].copy() for key in keys}
    result=predict(v['rq'],v['rk'],v['qq'],v['qk'],v['cov'],v['base'],v['r_h20'],v['q_h20'],v['r_final'],v['q_final'],v['original_hw'])
    out.mkdir(parents=True)
    np.savez_compressed(out/'fields.npz',**result['fields'])
    np.savez_compressed(out/'masks.npz',**{name+'.work':np.packbits(ms[0]) for name,ms in result['masks'].items()},
                        **{name+'.original':np.packbits(ms[1]) for name,ms in result['masks'].items()})
    receipt={'state':'QK_V3_FULL_FIELDS_SEALED_UNSCORED','binding':binding,'info':result['info'],'new_encoder_forwards_in_consumer':0,
             'deployment_cost':'includes original rawH20 producer encoding; cache cost not counted as deployment zero',
             'files_sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('fields.npz','masks.npz')}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
