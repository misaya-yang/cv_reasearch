"""Schema-only tests with explicit tiny tensors; no scientific feature/accuracy test."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from export_cached_relational_packet import export_packet,CLEAN_FIELDS


class CachedPacketSchemaTests(unittest.TestCase):
    def inputs(self,path):
        q=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,0.]],dtype=torch.float16)
        r=torch.tensor([[0.,1.],[1.,0.],[-1.,0.],[1.,1.]],dtype=torch.float16)
        torch.save({'q':q,'r':r,'debiased':False},path/'0_0_0.pt')
        native=np.zeros((4,4),bool);native[:2,:2]=True
        # Query-label object would fail allow_pickle=False if materialized.
        np.savez(path/'0_0_0.npz',score=np.array([[0.,.5],[.6,1.]],np.float32),
            cov=np.array([[.1,0.],[1.,0.]],np.float32),native=np.packbits(native),truth=np.array([{'must_not_read':True}],object))
        meta={'pair_id':'0_0_0','representation_id':'pinned Part1 output',
            'feature_provenance':'explicit schema fixture, not real features',
            'source_coverage_provenance':'source mask area coverage',
            'cache_alignment_provenance':'schema fixture only',
            'mapping_contract':'full_grid_regular_no_padding','feature_grid_hw':[2,2],
            'source_work_hw':[4,4],'query_work_hw':[4,4],'query_original_hw':[6,8]}
        return meta,q,r

    def test_clean_export_no_truth_access_and_row_major_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);meta,q,r=self.inputs(p)
            report=export_packet(features_path=p/'0_0_0.pt',packet_path=p/'0_0_0.npz',metadata=meta,native_patch_mode='score_native',out_path=p/'clean.npz')
            with np.load(p/'clean.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),CLEAN_FIELDS)
                np.testing.assert_array_equal(z['query_features'].reshape(4,2),q.float().numpy())
                np.testing.assert_array_equal(z['source_features'].reshape(4,2),r.float().numpy())
                np.testing.assert_array_equal(z['native_patch_fg'],[[False,False],[True,True]])
                np.testing.assert_array_equal(z['query_xy'],[[[1,1],[3,1]],[[1,3],[3,3]]])
            self.assertFalse(report['query_GT_read']);self.assertFalse(report['baseline57_reproduction_claim'])

    def test_cached_native_coverage_is_explicit_distinct_choice(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);meta,_,_=self.inputs(p)
            report=export_packet(features_path=p/'0_0_0.pt',packet_path=p/'0_0_0.npz',metadata=meta,native_patch_mode='cached_native_coverage',out_path=p/'clean.npz')
            with np.load(p/'clean.npz') as z:np.testing.assert_array_equal(z['native_patch_fg'],[[True,False],[False,False]])
            self.assertIn('finalized native',report['native_patch_mask_definition'])

    def test_metadata_grid_dtype_and_fresh_output_contracts(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);meta,_,_=self.inputs(p)
            base=dict(features_path=p/'0_0_0.pt',packet_path=p/'0_0_0.npz',native_patch_mode='score_native',out_path=p/'clean.npz')
            for change in ({'mapping_contract':'letterbox_unknown'},{'pair_id':'different'},{'feature_grid_hw':[1,4]},{'query_original_hw':None}):
                with self.assertRaises(ValueError):export_packet(metadata={**meta,**change},**base)
            export_packet(metadata=meta,**base)
            with self.assertRaises(ValueError):export_packet(metadata=meta,**base)
            torch.save({'q':torch.ones(4,2),'r':torch.ones(4,2),'debiased':False},p/'0_0_0.pt')
            with self.assertRaises(ValueError):export_packet(metadata=meta,**{**base,'out_path':p/'other.npz'})

if __name__=='__main__':unittest.main(verbosity=2)
