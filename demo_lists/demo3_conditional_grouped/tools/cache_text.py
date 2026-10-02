from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import argparse,hashlib,json,pathlib,sys
import torch
from transformers import Qwen3Model,Qwen2Tokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=pathlib.Path,default=pathlib.Path('assets/checkpoints/qwen3'));p.add_argument('--data',type=pathlib.Path,default=pathlib.Path('assets/data/coco_pilot_v1/manifest.json'));p.add_argument('--batch',type=int,default=8);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('No GPU; downloads/CPU correctness can proceed independently')
 d=json.loads(a.data.read_text());tokenizer=Qwen2Tokenizer.from_pretrained(a.weights,local_files_only=True,padding_side='right');net=Qwen3Model.from_pretrained(a.weights,local_files_only=True,torch_dtype=torch.bfloat16).eval().cuda()
 out=a.data.parent/'text_cache';out.mkdir(exist_ok=True)
 for lo in range(0,len(d['rows']),a.batch):
  rows=d['rows'][lo:lo+a.batch];tok=tokenizer([r['caption'] for r in rows],truncation=True,max_length=128,padding='max_length',return_tensors='pt')
  with torch.no_grad():emb=net(**{k:v.cuda() for k,v in tok.items()}).last_hidden_state.cpu()
  if emb.shape[-1]!=2048:raise ValueError('Unexpected official text hidden size')
  for i,r in enumerate(rows):torch.save({'embedding':emb[i].clone(),'caption_sha256':hashlib.sha256(r['caption'].encode()).hexdigest(),'valid_length':int(tok['attention_mask'][i].sum())},out/(str(r['image_id'])+'.pt'))
  print('text_cache',min(lo+a.batch,len(d['rows'])),'/',len(d['rows']),flush=True)
 (out/'contract.json').write_text(json.dumps({'model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'max_length':128,'padding':'right','dtype':'bfloat16','note':'Native Qwen3Model hidden states; no semantic h cache.'},indent=2))
if __name__=='__main__':main()
