"""Contract/algebra checks only; no scientific toy gain or model execution."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from relational_exclusion_cpu import (Star,MODES,RelationContractError,build_stars,
    select_role_banks,assignment_costs,compare_same_banks,relational_exclusion,
    finalize_deletion,main)


def star(root,features,points=None):
    f=np.asarray(features,float);f=f/np.linalg.norm(f,axis=-1,keepdims=True)
    p=np.asarray(points if points is not None else [[0,0],[1,0],[0,1]],float)
    d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);d/=d.max()
    return Star(root,(root,root+1,root+2),f,d)


class RelationalContracts(unittest.TestCase):
    def setUp(self):
        self.e=np.eye(5)
        self.q=star(0,self.e[:3])
        self.F=[star(10,self.e[[0,1,3]]),star(20,self.e[[0,3,2]])]
        b=np.stack([.95*self.e[0]+np.sqrt(1-.95**2)*self.e[4],self.e[1],self.e[2]])
        self.B=[star(30,b),star(40,b)]

    def test_joint_flip_uses_consistent_star_and_equal_weight_relaxation(self):
        r=compare_same_banks(self.q,self.F,self.B)
        self.assertFalse(r['delete']['center_only'])
        self.assertFalse(r['delete']['independent_edges'])
        self.assertTrue(r['delete']['shared_U']);self.assertTrue(r['delete']['joint_flip'])
        f,b=r['role_costs']['F'],r['role_costs']['B']
        self.assertAlmostEqual(f['independent_edges'],0.)
        self.assertAlmostEqual(f['shared_U'],1/3)
        self.assertAlmostEqual(b['shared_U'],.05/3)
        self.assertAlmostEqual(b['independent_edges'],b['shared_U'])
        self.assertEqual([e['source_root'] for e in f['independent_winners']],[10,20])
        for role in ('F','B'):self.assertLessEqual(r['role_costs'][role]['independent_edges'],r['role_costs'][role]['shared_U']+1e-14)

    def test_relaxation_bound_exact_under_common_float_primitives(self):
        rng=np.random.default_rng(7)
        for _ in range(40):
            q=star(0,rng.normal(size=(3,7)))
            f=star(10,rng.normal(size=(3,7)));b=star(20,rng.normal(size=(3,7)))
            result=compare_same_banks(q,[f],[b])
            for role in ('F','B'):
                c=result['role_costs'][role]
                self.assertLessEqual(c['independent_edges'],c['shared_U'])
        tied=compare_same_banks(q,[f],[star(20,f.features)])
        self.assertFalse(tied['delete']['joint_flip'])

    def test_same_assignment_is_used_for_both_U_and_R(self):
        source=star(4,self.e[[0,2,1]],[[0,0],[2,0],[0,1]])
        costs=assignment_costs(self.q,source)
        self.assertEqual([x[2] for x in costs],[(0,1,2),(0,2,1)])
        for U,R,perm in costs:
            self.assertAlmostEqual(U,np.mean([1-self.q.features[i]@source.features[perm[i]] for i in range(3)]))
            self.assertAlmostEqual(R,np.mean([abs(self.q.distances[i,j]-source.distances[perm[i],perm[j]]) for i,j in ((0,1),(0,2),(1,2))]))

    def test_exact_role_ties_abstain_in_every_arm(self):
        r=compare_same_banks(self.q,[star(10,self.e[:3])],[star(20,self.e[:3])])
        self.assertFalse(any(r['delete'].values()));self.assertTrue(r['pareto_has_F'])

    def test_joint_flip_does_not_credit_preexisting_unary_BG_winner(self):
        q=star(0,self.e[:3]);F=[star(5,self.e[[4,3,0]])];B=[star(9,self.e[:3])]
        r=compare_same_banks(q,F,B)
        self.assertTrue(r['delete']['shared_U']);self.assertFalse(r['delete']['joint_flip'])

    def test_missing_role_abstains(self):
        r=compare_same_banks(self.q,self.F,[])
        self.assertTrue(r['status'].startswith('abstain'));self.assertFalse(any(r['delete'].values()))

    def test_balanced_top8_center_cap_and_stable_index(self):
        bank={i:star(i,self.e[:3]) for i in range(21)}
        cov=np.zeros(21);cov[:11]=.001
        F,B=select_role_banks(self.q,bank,cov)
        self.assertEqual([s.root for s in F],list(range(8)))
        self.assertEqual([s.root for s in B],list(range(11,19)))
        cov[:19]=.001;F,B=select_role_banks(self.q,bank,cov)
        self.assertEqual(len(F),2);self.assertEqual(len(B),2)

    def test_nonfinite_zero_duplicate_coordinates_and_insufficient_companions_abstain(self):
        f=np.tile(self.e[0],(1,3,1));xy=np.array([[[0,0],[1,0],[2,0]]]);v=np.ones((1,3),bool)
        stars,_=build_stars(f,xy,v);self.assertEqual(len(stars),3)
        self.assertEqual(stars[0].ids,(0,1,2))
        f[0,1]=np.nan;stars,_=build_stars(f,xy,v);self.assertEqual(stars,{})
        f=np.zeros_like(f);stars,_=build_stars(f,xy,v);self.assertEqual(stars,{})
        f=np.full_like(f,1e308);stars,_=build_stars(f,xy,v);self.assertEqual(stars,{})
        f=np.tile(self.e[0],(1,3,1));stars,_=build_stars(f,np.zeros_like(xy),v);self.assertEqual(stars,{})

    def test_all_source_background_allowed_with_optional_adjacency_only(self):
        f=np.tile(self.e[0],(1,4,1));xy=np.stack([np.arange(4),np.zeros(4)],axis=-1)[None]
        cov=np.array([[.1,0,0,0]])
        stars,_=build_stars(f,xy,np.ones((1,4),bool),source_coverage=cov)
        F,B=select_role_banks(self.q,stars,cov)
        self.assertEqual(len(F),1);self.assertEqual(len(B),1)
        self.assertEqual(F[0].root,0)

    def test_original_final_intersection_prevents_finalizer_readdition(self):
        pre=np.ones((2,4),bool);saved=np.array([[1,0,1,0],[1,0,1,0]],bool);calls=[]
        def finalize(mask):calls.append(mask.copy());return np.ones_like(saved)
        out=finalize_deletion(np.array([[1,0]],bool),native_pre_final=pre,native_final=saved,finalizer=finalize,mapping_contract="full_grid_nearest")
        self.assertEqual(len(calls),1);np.testing.assert_array_equal(out['final_mask'],saved)
        self.assertFalse((out['final_mask']&~saved).any())
        self.assertFalse(out['pre_final'][:,:2].any())
        with self.assertRaises(RelationContractError):
            finalize_deletion(np.zeros((1,2),bool),native_pre_final=pre,native_final=saved,finalizer=finalize)
        calls.clear();noop=finalize_deletion(np.zeros((1,2),bool),native_pre_final=pre,native_final=saved,finalizer=finalize,mapping_contract="full_grid_nearest")
        self.assertEqual(calls,[]);np.testing.assert_array_equal(noop['final_mask'],saved)

    def test_cli_serializes_provided_contract_fixture_without_model_or_finalizer(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);xy=np.stack([np.arange(4),np.zeros(4)],axis=-1)[None]
            features=self.e[:4][None].astype(float)
            np.savez(p/'input.npz',query_features=features,source_features=features,
                query_xy=xy,source_xy=xy,query_valid=np.ones((1,4),bool),source_valid=np.ones((1,4),bool),
                source_coverage=np.array([[1.,1.,0.,0.]]),native_patch_fg=np.ones((1,4),bool))
            (p/'meta.json').write_text(json.dumps({k:'explicit contract fixture' for k in ('representation_id','source_role_definition','native_patch_mask_definition','feature_provenance','coordinate_system')}))
            main(['--packet',str(p/'input.npz'),'--metadata',str(p/'meta.json'),'--out',str(p/'out.npz')])
            with np.load(p/'out.npz') as z:self.assertEqual(set(z.files),{mode+'_delete' for mode in MODES})
            report=json.loads((p/'out.json').read_text())
            self.assertEqual(report['selected_operator'],'joint_flip')
            self.assertEqual(report['encoder_calls'],0)
            self.assertIn('not supplied or run',report['output_scope'])

    def test_cli_requires_metadata_and_rejects_query_GT_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'meta.json').write_text(json.dumps({k:'explicit fixture contract' for k in ('representation_id','source_role_definition','native_patch_mask_definition','feature_provenance','coordinate_system')}))
            np.savez(p/'input.npz',query_GT=np.ones((2,2)))
            with self.assertRaises(RelationContractError):main(['--packet',str(p/'input.npz'),'--metadata',str(p/'meta.json'),'--out',str(p/'out.npz')])
            self.assertFalse((p/'out.npz').exists())

if __name__=='__main__':unittest.main(verbosity=2)
