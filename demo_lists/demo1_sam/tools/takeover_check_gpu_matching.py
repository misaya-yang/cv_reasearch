"""Real-data selection-equivalence check; no GT or new quality rule."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from takeover_virtual_flip import load_models, encode_prompt_rows, Sam
from takeover_prompt_stability import perturb_objects, OFFSETS
from takeover_gpu_consistency import gpu_consistency


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads((args.source_run/"report.json").read_text());expected=json.loads(args.expected.read_text())
    decoder,encoder,_=load_models(args.source_run/"mask_decoder_state.pt",args.checkpoint,"cuda:0")
    proxy=SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
    report={"status":"RUNNING","GT_used":False,"rows_checked":0,"mismatches":[],"max_consistency_difference":0.0}
    def save():args.output.write_text(json.dumps(report,indent=2)+"\n")
    with torch.inference_mode():
        for info in source["images"]:
            with np.load(args.source_run/info["encoded_inputs"]["npz"],allow_pickle=False) as encoded:
                image,pe,dense=[torch.from_numpy(encoded[k]).cuda() for k in ("image_embeddings","image_pe","dense_nomask")]
                size=tuple(int(x) for x in encoded["original_size"]);isize=tuple(int(x) for x in encoded["input_size"])
            for regime in ("central","near_boundary","box"):
                rec=next(r for r in source["records"] if r["image_id"]==info["image_id"] and r["method"]=="official" and r["regime"]==regime)
                parts=[encode_prompt_rows(encoder,perturb_objects(rec["prompt_metadata"],regime,size,isize,o),regime,size,1024,"cuda:0")[0] for o in OFFSETS]
                ps=torch.cat(parts);p=len(parts[0])
                low,_=decoder.predict_masks(image_embeddings=image,image_pe=pe,sparse_prompt_embeddings=ps,dense_prompt_embeddings=dense.expand(len(ps),-1,-1,-1))
                views=(Sam.postprocess_masks(proxy,low,isize,size)>0).reshape(4,p,4,*size)
                with np.load(args.source_run/rec["npz"],allow_pickle=False) as base:
                    chosen,consistency=gpu_consistency(torch.from_numpy(base["masks"]).cuda(),torch.from_numpy(base["iou_prediction"]).cuda(),views)
                    chosen=chosen.cpu().numpy();consistency=consistency.cpu().numpy()
                    for j,aid in enumerate(base["annotation_ids"]):
                        row=next(r for r in expected["rows"] if r["image_id"]==info["image_id"] and r["regime"]==regime and r["annotation_id"]==int(aid))
                        report["max_consistency_difference"]=max(report["max_consistency_difference"],float(np.max(np.abs(consistency[j]-row["consistency"]))))
                        if int(chosen[j])!=row["selections"]["perturb_consistency"]:report["mismatches"].append({"image_id":info["image_id"],"regime":regime,"annotation_id":int(aid),"GPU":int(chosen[j]),"CPU":row["selections"]["perturb_consistency"]})
                        report["rows_checked"]+=1
            save();print(json.dumps({"checked_images":report["rows_checked"],"mismatches":len(report["mismatches"])}),flush=True)
    report["status"]="PASS_SAME_SELECTIONS" if not report["mismatches"] else "FAIL_DIFFERENT_SELECTIONS";save()
    if report["mismatches"]:raise SystemExit(2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("source-run","checkpoint","expected","output"):p.add_argument("--"+k,type=Path,required=True)
    main(p.parse_args())
