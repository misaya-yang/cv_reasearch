"""What would the frozen recogniser score if WHERE were perfect (GT class regions)? stdout only."""
import sys, glob, torch, numpy as np
from PIL import Image
D = "/root/autodl-tmp/demo2_pilot"; C = 150
va = torch.load("/root/demo2_cache/regfeat_dinov2l_validation.pt"); ck = torch.load("/root/demo2_cache/regclf_dinov2l.pt")
net = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)); net.load_state_dict(ck["state"]); net.eval()
with torch.no_grad(): pred = net((va["x"].float() - ck["mu"]) / ck["sd"]).argmax(1)
fs = sorted(glob.glob(D + "/data/ADEChallengeData2016/annotations/validation/*.png"))
npx = torch.tensor([float(np.prod(Image.open(f).size)) for f in fs])
w = va["area"].double() * npx[va["img"]].double()
conf = torch.zeros(C, C, dtype=torch.float64); conf.index_put_((va["y"], pred), w, accumulate=True)
tp, g, p = conf.diag(), conf.sum(1), conf.sum(0)
print("GT regions + frozen DINOv2-L recogniser: mIoU %.2f  mAcc %.2f  aAcc %.2f  (regions %d, region acc %.4f)" % (
    float(torch.nanmean(tp / (g + p - tp))) * 100, float(torch.nanmean(tp / g)) * 100, float(tp.sum() / conf.sum()) * 100, len(pred), float((pred == va["y"]).float().mean())))
