#!/usr/bin/env python3
"""Server-only CPU software fixtures: real processor, synthetic tiny model.

No checkpoint, pretrained model, real-task quality or CUDA is used. This module
is called only by the driver CPU preflight, never during paid GPU execution.
"""
from __future__ import annotations

import contextlib
import io
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


def interface_cases(sam3_root,base_source):
    if not sys.platform.startswith("linux") or not str(Path(__file__).resolve()).startswith("/root/"):
        raise RuntimeError("Only authorized no-card SERVER CPU may execute fixtures")
    import numpy as np
    import torch
    from PIL import Image
    import sam3_query_exemplar_probe as probe
    sys.path.insert(0,sam3_root)
    from sam3.model.sam3_image_processor import Sam3Processor
    if probe.sha(inspect.getfile(Sam3Processor))!=probe.PROCESSOR_SHA:raise RuntimeError("Imported processor is not the actual pinned source")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError("CPU fixture must not initialize CUDA")
    class TinyPrompt:
        def __init__(self):
            self.boxes=torch.empty((0,1,4),dtype=torch.float32);self.labels=torch.empty((0,1),dtype=torch.bool)
        def append_boxes(self,boxes,labels):
            self.boxes=torch.cat((self.boxes,boxes),0);self.labels=torch.cat((self.labels,labels),0)
    class TinyBackbone:
        def __init__(self):self.images=0;self.texts=0
        def forward_image(self,image):self.images+=1;return {"vision_features":image.clone(),"nested":{ "levels":[image[:, :1].clone()]}}
        def forward_text(self,text,device):
            if text!=["visual"]:raise AssertionError("No category-name text in visual fixture")
            self.texts+=1
            return {"language_features":torch.ones((1,1,4)),"language_mask":torch.zeros((1,1),dtype=torch.bool),"language_embeds":torch.ones((1,4))}
    class TinyModel:
        def __init__(self):self.backbone=TinyBackbone();self.inst_interactive_predictor=None;self.decoders=0;self.box_counts=[];self.mutate_find=False
        def _get_dummy_prompt(self):return TinyPrompt()
        def forward_grounding(self,backbone_out,find_input,geometric_prompt,find_target):
            self.decoders+=1;self.box_counts.append(len(geometric_prompt.boxes))
            if self.mutate_find:find_input.img_ids.add_(17)
            masks=torch.full((1,1,16,16),-8.,dtype=torch.float32)
            grid=(torch.arange(16)+.5)/16
            for box,label in zip(geometric_prompt.boxes[:,0],geometric_prompt.labels[:,0]):
                if not bool(label):raise AssertionError("No negative prompt fixture")
                cx,cy,w,h=box
                region=(grid[None,:]>=cx-w/2)&(grid[None,:]<=cx+w/2)&(grid[:,None]>=cy-h/2)&(grid[:,None]<=cy+h/2)
                masks[0,0][region]=8.
            return {"pred_boxes":geometric_prompt.boxes[:1,0].reshape(1,1,4),"pred_logits":torch.full((1,1,1),8.),
                    "pred_masks":masks,"presence_logit_dec":torch.full((1,1),8.)}
    geom=SimpleNamespace(CANVAS=64,rectangles=lambda:((0,26,64,38),(0,0,64,26)))
    image=Image.fromarray(np.full((64,64,3),127,dtype=np.uint8));source=[.5,.75,.25,.25];query=[.5,.2,.2,.15]
    def fresh():
        model=TinyModel();processor=Sam3Processor(model,resolution=64,device="cpu",confidence_threshold=.5)
        return model,processor,processor.set_image(image)
    checks=[]
    with torch.inference_mode():
        model,p,state=fresh();one,_,_=probe.prompted(p,state,source,None,geom);two,_,_=probe.prompted(p,state,source,None,geom)
        assert torch.equal(one,two) and one.dtype==torch.bool;checks.append("actual_processor_source_only_NoOp")
        model,p,state=fresh();before=probe.clone_image_state(state)
        probe.prompted(p,state,source,query,geom)
        assert "language_features" not in state["backbone_out"] and probe.equal_image_state(state,before)
        assert model.backbone.texts==1;checks.append("nested_language_and_image_cache_isolation")
        model,p,state=fresh();left=probe.clone_image_state(state);right=probe.clone_image_state(state)
        p.add_geometric_prompt(source,True,left);p.add_geometric_prompt(query,True,left);p.add_geometric_prompt(source,True,right)
        assert len(left["geometric_prompt"].boxes)==2 and len(right["geometric_prompt"].boxes)==1 and "geometric_prompt" not in state
        p.reset_all_prompts(left)
        assert "geometric_prompt" not in left and "language_features" not in left["backbone_out"] and "language_features" in right["backbone_out"]
        checks.append("actual_append_boxes_and_reset_do_not_cross_states")
        model,p,state=fresh();model.mutate_find=True;old_find=p.find_stage;ids=old_find.img_ids.clone()
        probe.prompted(p,state,source,query,geom)
        assert p.find_stage is old_find and torch.equal(old_find.img_ids,ids);checks.append("actual_FindStage_isolation_under_mutation")
        model,p,state=fresh();probe.prompted(p,state,source,None,geom);probe.prompted(p,state,source,query,geom)
        assert model.backbone.images==1 and model.decoders==3 and model.box_counts==[1,1,2];checks.append("one_actual_set_image_three_actual_grounding_calls")
        model,p,state=fresh();_,initial,calls=probe.prompted(p,state,None,query,geom)
        assert initial is None and calls==1 and model.box_counts==[1];checks.append("query_only_exactly_one_positive_box")
        model,p,state=fresh();native,_,_=probe.prompted(p,state,source,None,geom);count=model.decoders
        empty=np.zeros((64,64),bool);box=probe.bbox(empty)
        output,used=probe.joint_query_prompt(p,state,source,box,[64,64],geom,native)
        assert box is None and used==0 and model.decoders==count and torch.equal(output,native)
        checks.append("genuine_empty_predictor_native_fallback_no_extra_prompt")
        case={"row":{"support":"S_001.jpg","query":"Q_002.jpg"},"original_global_index":53,"query_shape":[64,64]}
        box=[10,7,17,9];other=probe.random_box(box,case)
        assert other==probe.random_box(box,case) and other[2:]==box[2:] and 0<=other[0]<=47 and 0<=other[1]<=55
        assert probe.random_box(None,case) is None;checks.append("UID_random_location_same_area_aspect_and_count")
        base=probe.frozen_base(base_source);ref=Image.new("RGB",(23,37));target=Image.new("RGB",(29,31))
        _,mapped=base.stitch(ref,target,[3.,4.,5.,6.]);s=base.rectangles()[0]
        expected=[5.5/23,(s[1]+7*s[3]/37)/base.CANVAS,5/23,6*s[3]/37/base.CANVAS]
        assert np.allclose(mapped,expected,rtol=0,atol=1e-12)
        mapped_q=probe.query_box_on_canvas([0,0,29,31],[31,29],base);q=base.rectangles()[1]
        assert np.allclose(mapped_q,[.5,q[3]/(2*base.CANVAS),1.,q[3]/base.CANVAS],rtol=0,atol=1e-12)
        checks.append("actual_A_source_and_query_canvas_coordinate_mapping")
        mask=np.zeros((31,29),bool);mask[3:18,5:23]=True;canvas=np.zeros((base.CANVAS,base.CANVAS),np.uint8)
        x,y,w,h=q;canvas[y:y+h,x:x+w]=base.on_canvas(mask,q)
        decoded=base.to_original(canvas,(29,31));packed=np.packbits(decoded)
        assert np.array_equal(decoded,mask) and packed.dtype==np.uint8 and np.array_equal(np.unpackbits(packed)[:mask.size].reshape(mask.shape),mask)
        checks.append("actual_A_original_size_readout_and_uint8_pack_decode")
    if len(checks)!=10 or torch.cuda.is_initialized():raise RuntimeError("CPU fixture completion/CUDA contract")
    return dict(state="ACTUAL_PROCESSOR_TINY_MODEL_INTERFACE10_PASSED",checks=checks,actual_processor=Sam3Processor.__module__,
        processor_resolution=64,device="cpu",synthetic_model_backbone_and_Prompt=True,fake_processor=False,
        pretrained_model_or_real_images=False,real_segmentation_gain_verified=False,CUDA_initialized=False)


def score_fixture(base_source):
    """Run the complete production scorer on explicit synthetic original grids."""
    import numpy as np
    from PIL import Image
    import sam3_query_exemplar_probe as probe
    with tempfile.TemporaryDirectory(prefix="c2_cpu_score_only_") as directory:
        out=Path(directory);(out/"legal").mkdir();(out/"oracle").mkdir();(out/"ann").mkdir()
        legal=[];oracle=[];old_sam=[];old_foris=[]
        yy,xx=np.indices((480,640));native=(yy>120)&(yy<360)&(xx>180)&(xx<470);foris=(yy>110)&(yy<360)&(xx>160)&(xx<450)
        def iu(pred,truth):return [int((pred&truth).sum()),int((pred|truth).sum())]
        for i in range(40):
            row=dict(fold=i%4,e=i,c=i,support="S_%d.jpg"%(i//2),query="Q_%d.jpg"%i)
            truth=(yy>130+(i%3))&(yy<350)&(xx>175)&(xx<455+(i%4))
            Image.fromarray((truth*(i+1)).astype(np.uint8)).save(out/"ann"/("Q_%d.png"%i))
            masks=dict(native=native,foris=foris,query_foris=native&((xx>170)&(xx<465)),query_self=native,
                       query_random=native|(xx<5),query_foris_only=foris,cheap_OR=native|foris,cheap_AND=native&foris)
            path=out/"legal"/(str(i)+".npz");np.savez_compressed(path,**{a:np.packbits(m) for a,m in masks.items()})
            legal.append(dict(row=row,path=str(path),sha256=probe.sha(path),actual_image_encoder_calls=1,
                extra_prompt_counts={"query_foris":1,"query_self":1,"query_random":1},query_only_source_reconstruction=None))
            path=out/"oracle"/(str(i)+".npz");np.savez_compressed(path,oracle_true_query_box=np.packbits(native))
            oracle.append(dict(row=row,path=str(path),sha256=probe.sha(path)))
            old_sam.append({**row,"original_iu":{"visual":iu(native,truth)}});old_foris.append({**row,"original_iu":{"native":iu(foris,truth)}})
        for name,data in (("sam.jsonl",old_sam),("foris.jsonl",old_foris),("legal/predictions.jsonl",legal),("oracle/predictions.jsonl",oracle)):
            (out/name).write_text("".join(json.dumps(r)+"\n" for r in data))
        manifest=dict(fixture=True,cases=[{"row":r["row"]} for r in legal],base_source=base_source,annotation_root=str(out/"ann"),oracle_enabled=True,original_sam_scored_records=str(out/"sam.jsonl"),original_foris_records=str(out/"foris.jsonl"))
        manifest_path=out/"manifest.json";probe.write(manifest_path,manifest)
        for name,recs,state in (("legal",legal,"ALL_LEGAL40_PREDICTIONS_FROZEN"),("oracle",oracle,"ORACLE40_PREDICTIONS_FROZEN")):
            probe.write(out/(name+"_freeze.json"),dict(state=state,episodes=40,manifest_sha256=probe.sha(manifest_path),query_GT_used_as_input=name=="oracle",files={r["path"]:r["sha256"] for r in recs}))
        start=time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):probe.score(SimpleNamespace(manifest=manifest_path,out=out))
        elapsed=time.monotonic()-start;report=probe.read(out/"report.json")
        if report["state"]!="COMPLETED" or report["episodes"]!=40 or len(report["comparisons"])!=12:
            raise RuntimeError("Complete synthetic scorer did not exercise all40/12 pairs")
        # Compare the optimized shared draws with the original frozen evaluator,
        # outside the complete-score timer. No synthetic gain is published.
        base=probe.frozen_base(base_source);recs=report["records"]
        expected=base.paired(recs,lambda r:r["original_iu"]["query_foris"],lambda r:r["original_iu"]["native"],draws=2000)
        measured=report["comparisons"]["query_foris__minus__native"]
        if not np.allclose([expected["miou"],expected["gain"],*expected["ci95"]],
                           [measured["miou"],measured["gain"],*measured["ci95"]],rtol=0,atol=1e-10):
            raise RuntimeError("Shared bootstrap does not reproduce the frozen statistic")
        return dict(state="SYNTHETIC_COMPLETE_SCORE40_TIMED",elapsed_seconds=elapsed,episodes=40,comparisons=12,
            original_shape=[480,640],bootstrap_draws=2000,photo_connected_resampling=True,
            original_frozen_paired_reference_exact_within_1e_10=True,real_task_quality_evidence=False,
            inputs_and_GT_explicitly_synthetic=True,fixture_temporary_files_removed_on_exit=True)
