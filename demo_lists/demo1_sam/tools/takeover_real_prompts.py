"""Prepare P128 distinct foreground-point prompts on a preselected real image."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"assets/source/segment-anything"))
from segment_anything.modeling import PromptEncoder
from segment_anything.utils.transforms import ResizeLongestSide


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--prompts",type=int,default=128)
    p.add_argument("--seed",type=int,default=2027)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    source=json.loads((a.source_run/"report.json").read_text())
    chosen=source["images"][0]  # Fixed first manifest image; no output-based choice.
    record=next(r for r in source["records"] if r["image_id"]==chosen["image_id"] and r["regime"]=="central" and r["method"]=="official")
    with np.load(a.source_run/record["npz"],allow_pickle=False) as saved:
        gt=saved["gt"].copy();ids=saved["annotation_ids"].copy()
    rng=np.random.default_rng(a.seed)
    pools=[np.argwhere(mask) for mask in gt]
    used=set();coords=[];rows=[];annotation_ids=[]
    for j in range(a.prompts):
        obj=j%len(pools)
        for _ in range(10000):
            y,x=pools[obj][rng.integers(len(pools[obj]))]
            if (int(x),int(y)) not in used:break
        else:raise ValueError("Insufficient distinct foreground coordinates")
        used.add((int(x),int(y)));coords.append([float(x),float(y)])
        annotation_ids.append(int(ids[obj]));rows.append({"annotation_id":int(ids[obj]),"positive_xy":[int(x),int(y)]})
    checkpoint=torch.load(ROOT/"assets/checkpoints/sam_vit_b_01ec64.pth",weights_only=True,map_location="cpu")
    encoder=PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(1024,1024),mask_in_chans=16)
    encoder.load_state_dict({k.removeprefix("prompt_encoder."):v for k,v in checkpoint.items() if k.startswith("prompt_encoder.")},strict=True)
    del checkpoint
    encoder=encoder.to("cuda:0").eval()
    with np.load(a.source_run/chosen["encoded_inputs"]["npz"],allow_pickle=False) as encoded:
        arrays={k:encoded[k].copy() for k in ("image_embeddings","image_pe","dense_nomask","original_size","input_size","image_id","mask_threshold")}
    transformed=ResizeLongestSide(1024).apply_coords(np.array(coords,dtype=np.float32)[:,None],tuple(arrays["original_size"]))
    with torch.inference_mode():
        sparse,dense=encoder(points=(torch.from_numpy(transformed).to("cuda:0"),torch.ones((a.prompts,1),dtype=torch.int64,device="cuda:0")),boxes=None,masks=None)
        assert sparse.shape==(a.prompts,2,256)
        assert torch.equal(dense[:1],torch.from_numpy(arrays["dense_nomask"]).to("cuda:0"))
        assert torch.allclose(encoder.get_dense_pe(),torch.from_numpy(arrays["image_pe"]).to("cuda:0"),atol=5e-5,rtol=0)
        arrays["sparse_perf"]=sparse.cpu().numpy()
    arrays["annotation_ids"]=np.array(annotation_ids,dtype=np.int64)
    np.savez(a.output_dir/"encoded_inputs.npz",**arrays)
    metadata={"status":"PREPARED_REAL_PRETRAINED_P128","created_at_utc":datetime.now(timezone.utc).isoformat(),
              "image_id":chosen["image_id"],"seed":a.seed,"prompts":a.prompts,"distinct_coordinates":len(used),
              "protocol":"128 independent single-positive-point prompts on the first fixed COCO image, cyclic original instances; uniform GT-foreground sampling without output-based choice; no repeated coordinates, no extra clicks per prompt; new performance prompt protocol, separate from central/near-boundary/box quality protocol",
              "source_subset_manifest_sha256":source["subset_manifest_sha256"],"prompt_rows":rows}
    (a.output_dir/"prompt_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ("status","image_id","seed","prompts","distinct_coordinates")}),flush=True)


if __name__=="__main__":main()
