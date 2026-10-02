"""Train actual SAM2 static-skip head; development quality, no oracle speed claim."""
import argparse
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch import nn
from takeover_sam2_quality import ROOT, build_sam2, SAM2ImagePredictor, encode
from takeover_sam2_compact_head import (static_features, build_cache, decode,
    phase_patches, dynamic_choice, quality_all)
from takeover_compact_phase_fallback import build_patch_cache
from takeover_flip_score import bootstrap_image_groups

OUT = ROOT/'results/takeover_20261001_v1/sam2_local_student_v1'


def main(resume=False, continue_from=None, max_epochs=600, patience=80):
    OUT.mkdir(exist_ok=resume)
    report = {'status': 'WAITING_VRAM', 'pid': os.getpid(), 'curves': [], 'rows': [], 'summary': {},
              'protocol': 'Native frozen SAM2.1-L, FP32/TF32off. Old24 images sorted by ID: first12 fit PCA/train, last12 teacher-MSE validation/development diagnosis; no independent test claim. Rank64, Linear1024->64/GELU/Linear64; inputs final parent256 plus two highres tiles256+512. Static768 affine contribution+bias cached once/image, dynamic256 contribution per prompt, all4mask/IQ/masktoken/objscore preserved. Teacher coefficient MSE only, seed2027 AdamW.001 wd.01, min100/max600 epochs patience80. Budgets0/5/10% fixed from SAM1; real selected phase patches include both skips; no full teacher upscaler in candidate path. Compact losslessly compressed FLOAT32 pairs use one static bank/image and indexed parent features. Quality references native head1..3 and actual dynamic token0; no latency or publication claim.'}
    if resume:
        old = json.loads((OUT/'report.json').read_text())
        if old['status'] != 'ERROR':
            raise RuntimeError('Recovery requires failed previous worker')
        n = len(list(OUT.glob('report.failed_attempt*.json')))+1
        (OUT/f'report.failed_attempt{n}.json').write_text(json.dumps(old, indent=2)+'\n')
        report['curves'] = old.get('curves', [])
    if continue_from is not None:
        if resume:raise ValueError('Continuation is a new experiment; recovery is separate')
        prior=json.loads((continue_from/'report.json').read_text())
        if prior['status']!='COMPLETED_SAM2_REAL_HEAD_DEVELOPMENT':raise RuntimeError('Continuation requires a completed training source')
        for name in ('training_pairs.npz','basis.pt'):(OUT/name).symlink_to((continue_from/name).resolve())
        for name in ('best.pt','last.pt'):shutil.copy2(continue_from/name,OUT/name)
        report.update(curves=prior['curves'],continuation_source=str(continue_from),max_epochs=max_epochs,patience=patience)
        report['protocol']+=' NEW convergence experiment: continue saved epoch/optimizer/CUDA RNG from completed capped training in a separate directory; reuse identical indexed pairs/basis without extraction or duplication. '+str(max_epochs)+' total maximum epochs, patience'+str(patience)+', best teacher validation MSE only. Prior epoch598/600 had not reached plateau; this tests inadequate optimization rather than a new head architecture. Prior deployment/test results preserved, no GT epoch or budget selection; subsequent existing-cohort replays are development/implementation evidence.'

    def save():
        tmp = OUT/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(OUT/'report.json')

    save()
    telemetry = None
    telemetry_log = None
    handles = []
    try:
        while int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) < 6500:
            time.sleep(10)
        telemetry_log = (OUT/('gpu_telemetry_resume.csv' if resume else 'gpu_telemetry.csv')).open('a' if resume else 'x')
        telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm', '--format=csv', '--loop-ms=1000'], stdout=telemetry_log, stderr=subprocess.STDOUT)
        torch.set_num_threads(2)
        torch.manual_seed(2027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        subset = ROOT/'assets/coco_quality_seed2027_v1'
        infos = sorted(json.loads((subset/'manifest.json').read_text())['images'], key=lambda x:x['image_id'])
        assert len(infos) == 24
        model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'), device='cuda:0').eval().requires_grad_(False)
        predictor = SAM2ImagePredictor(model)
        decoder = model.sam_mask_decoder
        capture = {}
        handles = [decoder.transformer.register_forward_hook(lambda m,i,o:capture.update(tokens=o[0],state=o[1])),
                   decoder.output_upscaling[4].register_forward_hook(lambda m,i,o:capture.update(upscaled=o))]
        pairs = OUT/'training_pairs.npz'
        if not pairs.exists():
            report['status'] = 'EXTRACT_NATIVE_COMPACT_PAIRS'
            save()
            statics, states, features, image_indices, parent_indices = [], [], [], [], []
            total = torch.zeros(512, dtype=torch.float64, device='cuda:0')
            cross = torch.zeros(512,512, dtype=torch.float64, device='cuda:0')
            count = 0
            generator = torch.Generator(device='cuda:0').manual_seed(2027)
            with torch.inference_mode():
                for n, info in enumerate(infos):
                    rgb = np.asarray(Image.open(subset/info['image_file']).convert('RGB'))
                    predictor.set_image(rgb)
                    static, t0, t1 = static_features(predictor._features['high_res_feats'])
                    statics.append(static.cpu().numpy())
                    for regime in ('central','near_boundary','box'):
                        sparse, dense = encode(predictor, info['objects'], regime, rgb.shape[:2])
                        decoder.predict_masks(image_embeddings=predictor._features['image_embed'], image_pe=model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, repeat_image=len(sparse)>1, high_res_features=predictor._features['high_res_feats'])
                        x = capture['state']
                        phi = capture['upscaled'].reshape(len(x),32,64,4,64,4).permute(0,2,4,3,5,1).reshape(-1,512)
                        idx = torch.randperm(len(phi), device='cuda:0', generator=generator)[:2048]
                        sampled = phi[idx]
                        states.append(x.reshape(-1,256)[idx].cpu().numpy())
                        features.append(sampled.cpu())
                        image_indices.append(np.full(len(idx), n, dtype=np.uint8))
                        parent_indices.append((idx%4096).cpu().numpy().astype(np.uint16))
                        if n < 12:
                            values = sampled.double()
                            total += values.sum(0)
                            cross += values.T@values
                            count += len(values)
                        if n == 0 and regime == 'central':
                            tiny = SimpleNamespace(output_upscaling=copy.deepcopy(decoder.output_upscaling).double())
                            phase = build_patch_cache(tiny)
                            ss = x[0,:64].double()
                            a, b = t0[:64].double(), t1[:64].double()
                            dc1, ln, act1, dc2, act2 = tiny.output_upscaling
                            reference = act2(dc2(act1(ln(dc1(ss.reshape(-1,256,1,1))+b)))+a).flatten(2)
                            actual = phase_patches(tiny, phase, ss, a, b)
                            error = float((actual-reference).abs().max())
                            report['new_phase_head_only_FP64_check'] = {'parents':64,'max_error':error,'tolerance':1e-10,'passed':error<=1e-10}
                            save()
                            assert error <= 1e-10
                            del tiny, phase, ss, a, b, reference, actual
                    report['extracted_images'] = n+1
                    save()
                    print(json.dumps({'extracted_images':n+1}), flush=True)
                mean = total/count
                covariance = cross/count-mean[:,None]*mean[None,:]
                eigen, basis = torch.linalg.eigh((covariance+covariance.T)/2)
                eigen, basis = eigen.flip(0).clamp_min(0), basis.flip(1)[:,:64].float()
                mean = mean.float()
                targets = [((f.cuda()-mean)@basis).cpu().numpy() for f in features]
                report['PCA_rank64_retained_variance'] = float(eigen[:64].sum()/eigen.sum())
                torch.save({'mean':mean.cpu(),'basis':basis.cpu(),'eigenvalues':eigen.cpu(),'fit_images':[x['image_id'] for x in infos[:12]]}, OUT/'basis.pt')
                # Atomic compressed features; no full-resolution masks or base weights.
                with (OUT/'training_pairs.tmp').open('xb') as file:
                    np.savez_compressed(file, static_bank=np.stack(statics), dynamic=np.concatenate(states), targets=np.concatenate(targets), image_index=np.concatenate(image_indices), parent_index=np.concatenate(parent_indices))
                (OUT/'training_pairs.tmp').replace(pairs)
                del statics, states, features, targets, total, cross, covariance, eigen
        basis_state = torch.load(OUT/'basis.pt', map_location='cuda:0', weights_only=True)
        center, basis = basis_state['mean'], basis_state['basis']
        capture.clear()
        predictor.reset_predictor()
        model.cpu()
        torch.cuda.empty_cache()
        report['status'] = 'ASSEMBLE_INDEXED_TRAINING_PAIRS'
        save()
        with np.load(pairs, allow_pickle=False) as data:
            bank = torch.from_numpy(data['static_bank']).cuda()
            dynamic = torch.from_numpy(data['dynamic']).cuda()
            image_index = torch.from_numpy(data['image_index'].astype(np.int64)).cuda()
            parent_index = torch.from_numpy(data['parent_index'].astype(np.int64)).cuda()
            inputs = torch.cat((dynamic, bank[image_index,parent_index]), -1)
            targets = torch.from_numpy(data['targets']).cuda()
        mask = image_index < 12
        trainx, valx = inputs[mask], inputs[~mask]
        trainy, valy = targets[mask], targets[~mask]
        del bank, dynamic, image_index, parent_index, inputs, targets, mask
        mean, std = trainx.mean(0), trainx.std(0).clamp_min(1e-3)
        trainx, valx = (trainx-mean)/std, (valx-mean)/std
        student = nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,64)).cuda()
        optimizer = torch.optim.AdamW(student.parameters(), lr=.001, weight_decay=.01)
        best, bad, best_epoch, start_epoch = float('inf'), 0, 0, 1
        if (resume or continue_from is not None) and (OUT/'last.pt').exists():
            last = torch.load(OUT/'last.pt', weights_only=True)
            student.load_state_dict(last['state'])
            optimizer.load_state_dict(last['optimizer'])
            torch.cuda.set_rng_state(last['CUDA_rng'])
            start_epoch, bad = last['epoch']+1, last['bad']
            best_state = torch.load(OUT/'best.pt', weights_only=True)
            best, best_epoch = best_state['teacher_mse'], best_state['epoch']
            report['curves'] = [r for r in report['curves'] if r['epoch'] < start_epoch]
        report.update(status='TRAINING', parameters=sum(p.numel() for p in student.parameters()), train_samples=len(trainx), validation_samples=len(valx))
        save()
        for epoch in range(start_epoch,max_epochs+1):
            student.train()
            order = torch.randperm(len(trainx),device='cuda:0')
            total = 0
            for idx in order.split(4096):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(student(trainx[idx]),trainy[idx])
                loss.backward()
                optimizer.step()
                total += float(loss)*len(idx)
            student.eval()
            with torch.no_grad():mse = float(nn.functional.mse_loss(student(valx),valy))
            if mse < best-1e-7:
                best, bad, best_epoch = mse, 0, epoch
                torch.save({'state':student.state_dict(),'input_mean':mean,'input_std':std,'epoch':epoch,'teacher_mse':mse}, OUT/'best.pt')
            else:bad += 1
            report['curves'].append({'epoch':epoch,'train_mse':total/len(trainx),'validation_teacher_mse':mse,'best_epoch':best_epoch})
            if epoch%25 == 0:
                torch.save({'state':student.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'bad':bad,'CUDA_rng':torch.cuda.get_rng_state()}, OUT/'last.pt')
                save()
                print(json.dumps(report['curves'][-1]),flush=True)
            if epoch>=100 and bad>=patience:break
        selected = torch.load(OUT/'best.pt',weights_only=True)
        student.load_state_dict(selected['state'])
        student.eval().requires_grad_(False)
        with torch.no_grad():
            student[0].weight.div_(std)
            student[0].bias.sub_(student[0].weight@mean)
        torch.save({'state':{k:v.cpu() for k,v in student.state_dict().items()},'feature_center':center.cpu(),'basis':basis.cpu(),'normalization_folded':True,'epoch':best_epoch,'architecture':'SAM2 static768 + dynamic256 to hidden64/rank64'}, OUT/'deployment.pt')
        report.update(selected_epoch=best_epoch, plateau_before_cap=epoch<max_epochs, status='DIAGNOSE_DEVELOPMENT_REAL_HEAD')
        save()
        del trainx, valx, trainy, valy, optimizer
        model.cuda()
        with torch.inference_mode():
            for info in infos[12:]:
                rgb = np.asarray(Image.open(subset/info['image_file']).convert('RGB'))
                predictor.set_image(rgb)
                cache = build_cache(decoder, predictor._features['high_res_feats'], student)
                with np.load(subset/info['gt_file'], allow_pickle=False) as data:
                    truth = torch.from_numpy(data['gt']).cuda()
                    ids = data['annotation_ids'].tolist()
                assert ids == [o['annotation_id'] for o in info['objects']]
                for regime in ('central','near_boundary','box'):
                    sparse, dense = encode(predictor, info['objects'], regime, rgb.shape[:2])
                    image, pe = predictor._features['image_embed'], model.sam_prompt_encoder.get_dense_pe()
                    original, iq, _, obj = decoder.predict_masks(image_embeddings=image, image_pe=pe, sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, repeat_image=len(sparse)>1, high_res_features=predictor._features['high_res_feats'])
                    full = predictor._transforms.postprocess_masks(original, predictor._orig_hw[0])>0
                    oq, ob = quality_all(full, truth)
                    choice = iq[:,1:].argmax(-1)+1
                    dynamic = dynamic_choice(decoder, original, iq)
                    native_dynamic = decoder._dynamic_multimask_via_stability(original, iq)[0]
                    assert torch.equal(native_dynamic, original[torch.arange(len(iq),device='cuda:0'),dynamic][:,None])
                    for budget in (0,.05,.1):
                        predicted, scores, mask_tokens, obj_scores = decode(decoder, student, cache, image, pe, sparse, dense, center, basis, budget)
                        assert predicted.shape == original.shape and scores.shape == iq.shape and mask_tokens.shape==(len(sparse),4,256) and obj_scores.shape==obj.shape
                        masks = predictor._transforms.postprocess_masks(predicted, predictor._orig_hw[0])>0
                        pq, pb = quality_all(masks, truth)
                        own = scores[:,1:].argmax(-1)+1
                        own_dynamic = dynamic_choice(decoder, predicted, scores)
                        flips = (masks!=full).float().mean((-2,-1))
                        for j, aid in enumerate(ids):
                            c, dc, cc, cd = int(choice[j]),int(dynamic[j]),int(own[j]),int(own_dynamic[j])
                            report['rows'].append({'image_id':info['image_id'],'annotation_id':aid,'regime':regime,'budget':budget,'original_head_iou':float(oq[j,c]),'delta_iou':float(pq[j,cc]-oq[j,c]),'delta_boundary_iou':float(pb[j,cc]-ob[j,c]),'native_dynamic_iou':float(oq[j,dc]),'delta_dynamic_iou':float(pq[j,cd]-oq[j,dc]),'head_choice_changed':cc!=c,'dynamic_choice_changed':cd!=dc,'head_flip_fraction':float(flips[j,c]),'max_IQ_drift':float((scores-iq).abs().max()),'max_objscore_drift':float((obj_scores-obj).abs().max())})
                save()
                print(json.dumps({'development_images_evaluated':len({r['image_id'] for r in report['rows']})}),flush=True)
        for regime in ('central','near_boundary','box'):
            report['summary'][regime]={}
            for budget in (0,.05,.1):
                rows=[r for r in report['rows'] if r['regime']==regime and r['budget']==budget]
                report['summary'][regime][str(budget)]={'mean_head_delta_iou':float(np.mean([r['delta_iou'] for r in rows])),'head_image_cluster95':bootstrap_image_groups(rows,'delta_iou'),'mean_native_dynamic_delta_iou':float(np.mean([r['delta_dynamic_iou'] for r in rows])),'dynamic_image_cluster95':bootstrap_image_groups(rows,'delta_dynamic_iou'),'head_choice_changes':sum(r['head_choice_changed'] for r in rows),'dynamic_choice_changes':sum(r['dynamic_choice_changed'] for r in rows)}
        report['status']='COMPLETED_SAM2_REAL_HEAD_DEVELOPMENT'
        print(json.dumps(report['summary']),flush=True)
    except BaseException as exc:
        report.update(status='ERROR',error=repr(exc))
        raise
    finally:
        for handle in handles:handle.remove()
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=5)
        if telemetry_log is not None:telemetry_log.close()
        save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--continue-from',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-epochs',type=int,default=600)
    parser.add_argument('--patience',type=int,default=80)
    args=parser.parse_args()
    if args.output is not None:OUT=args.output
    main(args.resume,args.continue_from,args.max_epochs,args.patience)
