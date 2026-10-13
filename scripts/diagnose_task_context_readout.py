"""Post-seal diagnosis of spatial rank and score-distribution contributions.

No new algorithm, model/encoder/fit or saved predictions. The diagnostic is
selected after the experiment was exposed and is not a fresh confirmation.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import opus55_reference_mask_trace as base
import run_task_context_compatibility as study

def transport(rank_source,marginal_source):
    source=rank_source.reshape(-1)
    ranks=rankdata(source,method="average")-1
    sorted_target=np.sort(marginal_source.reshape(-1),kind="stable")
    lo=np.floor(ranks).astype(np.int64);hi=np.ceil(ranks).astype(np.int64)
    fraction=ranks-lo
    return ((1-fraction)*sorted_target[lo]+fraction*sorted_target[hi]).reshape(rank_source.shape)

def main():
    start=time.perf_counter();out=study.DEFAULT_OUT
    seal=study.read(out/"sealed.json");report=study.read(out/"report.json")
    assert report["seal_sha256"]==base.sha(out/"sealed.json")
    pairs=study.read(out/"pairs.json");truth=study.read(study.PARENT/"query_truth.json")
    groups=study.photo_components(pairs);rows=[]
    for pair,receipt,row in zip(pairs,seal["records"],report["records"]):
        assert pair["id"]==receipt["id"]==row["id"]
        field_path=out/"fields"/receipt["file"];assert base.sha(field_path)==receipt["fields_sha256"]
        tasks=[]
        with np.load(field_path) as z:
            for variant in ("A","B"):
                binding=truth[pair["id"]]["masks"][variant]
                assert base.sha(binding["path"])==binding["sha256"]
                gt0=base.read_mask(binding["path"]).astype(np.float32)
                import torch
                import torch.nn.functional as F
                gt=F.interpolate(torch.from_numpy(gt0)[None,None],(1024,1024),mode="nearest")[0,0].numpy().astype(bool)
                nonlinear=study.dense_field(z[variant+"/context"],"zero")
                linear=study.dense_field(z[variant+"/linear_context"],"zero")
                old=next(t for t in row["tasks"] if t["variant"]==variant)
                values={"context":nonlinear,"linear":linear,
                        "linear_rank_context_distribution":transport(linear,nonlinear),
                        "context_rank_linear_distribution":transport(nonlinear,linear)}
                metrics={}
                for name,value in values.items():
                    pred=value>0;I=int((pred&gt).sum());U=int((pred|gt).sum())
                    metrics[name]={"iou":100*I/max(U,1),"I":I,"U":U,"predicted_pixels":int(pred.sum())}
                assert abs(metrics["context"]["iou"]-old["metrics"]["context/zero"]["iou"])<1e-12
                assert abs(metrics["linear"]["iou"]-old["metrics"]["linear_context/zero"]["iou"])<1e-12
                tasks.append({"variant":variant,"metrics":metrics,
                    "linear_rank_context_distribution_area_error":metrics["linear_rank_context_distribution"]["predicted_pixels"]-metrics["context"]["predicted_pixels"],
                    "context_rank_linear_distribution_area_error":metrics["context_rank_linear_distribution"]["predicted_pixels"]-metrics["linear"]["predicted_pixels"]})
        rows.append({"id":pair["id"],"dataset":pair["dataset"],"tasks":tasks})
    domains={}
    names=("context","linear","linear_rank_context_distribution","context_rank_linear_distribution")
    for dataset in ("coco","paco_part","suim"):
        selected=[r for r in rows if r["dataset"]==dataset]
        values={name:np.array([np.mean([t["metrics"][name]["iou"] for t in r["tasks"]]) for r in selected]) for name in names}
        domains[dataset]={"mean_complete_iou":{name:float(v.mean()) for name,v in values.items()},
            "compared_to_linear":{name:study.grouped_interval(selected,values[name]-values["linear"],groups) for name in names if name!="linear"},
            "max_area_errors":{key:max(abs(t[key]) for r in selected for t in r["tasks"])
                               for key in ("linear_rank_context_distribution_area_error","context_rank_linear_distribution_area_error")}}
    dest=ROOT/"evidence/local/task_context_compatibility120_20261012/readout_diagnosis.json"
    base.write(dest,{"state":"POST_SEAL_DIAGNOSIS_COMPLETE","domains":domains,"records":rows,
        "source_sha256":base.sha(__file__),"seal_sha256":base.sha(out/"sealed.json"),
        "original_report_sha256":base.sha(out/"report.json"),"seconds":time.perf_counter()-start,
        "new_encoder_calls":0,"new_model_calls":0,"new_saved_predictions":0,
        "diagnostic_binarizations_in_memory":240,
        "caveat":"Post-hoc diagnosis on already exposed outputs. Average-rank ties may alter marginal area slightly. IoU is nonlinear; crossed gains are not additive causal allocation."})
    print(json.dumps({"domains":domains,"seconds":time.perf_counter()-start}),flush=True)

if __name__=="__main__":main()
