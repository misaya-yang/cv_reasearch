"""Reference task regression with an unlabeled physical-view covariance prior.

This is a research candidate, not an adopted or proven-original method.
Task scores are learned only from a lawful reference mask. Query whole/fine
physical correspondences define a prior over classifier directions. The
pair-permuted control shares the exact marginal covariance and eigenvalues.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

ARMS=('reference_only','query_variance','physical_prior','pairing_null')


def _check(x,name):
    if x.ndim!=2 or x.dtype!=torch.float32 or x.device.type!='cpu' or not torch.isfinite(x).all():
        raise ValueError(name+' must contain CPU FP32 unit feature rows')
    if (x.norm(dim=1)-1).abs().max()>1e-3:
        raise ValueError(name+' is not unit normalized')


@torch.inference_mode()
def physical_prior(whole,native,device='mps'):
    _check(whole,'whole');_check(native,'native')
    side=int(len(whole)**.5);channels=whole.shape[1]
    if side*side!=len(whole) or len(native)!=4*len(whole) or native.shape[1]!=channels:
        raise ValueError('Require whole square grid and its aligned 2x2-finer observation')
    pooled=native.reshape(side,2,side,2,channels).mean((1,3))
    x,y=whole.to(device),pooled.reshape(-1,channels).to(device)
    mx,my=x.mean(0),y.mean(0);xc,yc=x-mx,y-my
    cx=(xc.T@xc/len(x)).cpu().double();cy=(yc.T@yc/len(y)).cpu().double()
    mean_delta=(mx-my).cpu().double()
    marginal=cx+cy+mean_delta[:,None]*mean_delta[None,:]
    marginal=(marginal+marginal.T)/2
    ridge=float(marginal.trace()/channels)
    if ridge<=0:raise ValueError('Degenerate physical-view covariance')
    ev,vec=torch.linalg.eigh(marginal+ridge*torch.eye(channels,dtype=torch.float64))
    if ev.min()<=0:raise ValueError('Nonpositive regularized marginal covariance')
    whitening=(vec*ev.rsqrt()[None,:])@vec.T
    perm=torch.from_numpy(np.random.default_rng(0).permutation(len(y))).to(device)
    operators={}
    spectra={}
    for name,target in [('physical',yc),('null',yc[perm])]:
        cross=(xc.T@target/len(x)).cpu().double();agreement=cross+cross.T
        normalized=whitening@agreement@whitening
        rho,u=torch.linalg.eigh((normalized+normalized.T)/2)
        order=torch.arange(channels-1,-1,-1);rho,u=rho[order],u[:,order]
        operators[name]=u;spectra[name]=rho
    positive=spectra['physical'].clamp_min(0)
    if float(positive.max())>1+1e-4:raise RuntimeError('Physical correlation spectrum outside the derived bound')
    actual=whitening@(operators['physical']*positive.sqrt()[None,:])
    # Preserve actual spectrum/capacity, change only directions supplied by
    # pairing. This null is diagnostic, not a deployable alternative.
    null=whitening@(operators['null']*positive.sqrt()[None,:])
    transforms=dict(reference_only=torch.eye(channels),query_variance=whitening.float(),
                    physical_prior=actual.float(),pairing_null=null.float())
    return transforms,dict(marginal_ridge=ridge,physical_eigenvalues=spectra['physical'].tolist(),
        null_eigenvalues=spectra['null'].tolist(),positive_modes=int((positive>0).sum()),
        positive_spectrum_sum=float(positive.sum()),effective_modes=float(positive.sum()**2/(positive.square().sum().clamp_min(1e-30))),
        mean_view_shift_norm=float(mean_delta.norm()),
        identity='B=Cov(X)+Cov(Y)+(mean(X)-mean(Y))(mean(X)-mean(Y))T; B-A=Cov(X,Y)+Cov(Y,X)',
        null='permuted physical mates; same B and same retained physical eigenvalues',
        common_offset_unidentified=True)


@torch.inference_mode()
def _fit(reference,coverage,transform,device):
    x=(reference.to(device)@transform.to(device)).cpu().double()
    c=torch.from_numpy(np.asarray(coverage,dtype=np.float64))
    mx=x.mean(0);mc=c.mean();xc=x-mx;yc=c-mc
    gx=xc.float().to(device);gy=yc.float().to(device)
    covariance=(gx.T@gx/len(x)).cpu().double();rhs=(gx.T@gy/len(x)).cpu().double()
    covariance=(covariance+covariance.T)/2
    regularizer=float(covariance.trace()/x.shape[1])
    if regularizer<=0:raise ValueError('Degenerate reference risk in proposed task space')
    matrix=covariance+regularizer*torch.eye(x.shape[1],dtype=torch.float64)
    beta=torch.linalg.solve(matrix,rhs);bias=mc-mx@beta
    # Return an original-feature-space functional; inference does not need to
    # retain a transformed query or change its normalization/footprint.
    w=transform.double()@beta
    source_prediction=reference.double()@w+bias
    return w,bias,dict(lambda_=regularizer,
        reference_area_mse=float((source_prediction-c).square().mean()),
        normal_residual=float((matrix@beta-rhs).norm()/rhs.norm().clamp_min(1e-20)),
        direction_norm=float(w.norm()),source_foreground_fraction=float(mc))


def otsu_exact(field):
    """Exact one-dimensional two-means split, no histogram or tuned prior."""
    value=np.asarray(field,dtype=np.float64).ravel()
    unique,count=np.unique(value,return_counts=True)
    if len(unique)==1:
        return .5,dict(reason='constant_field_use_source_coverage_half',levels=1)
    mass=np.cumsum(count)[:-1].astype(float);total=count.sum();weighted=unique*count
    left=np.cumsum(weighted)[:-1];right=weighted.sum()-left
    variance=mass*(total-mass)*(left/mass-right/(total-mass))**2
    index=int(np.argmax(variance))
    threshold=float(unique[index])
    return threshold,dict(reason='query_two_means_minimum_within_group_SSE',levels=len(unique),
        lower_count=int(mass[index]),upper_count=int(total-mass[index]),between_variance=float(variance[index]/total**2))


@torch.inference_mode()
def predict(reference,coverage,whole,native,device='mps'):
    _check(reference,'reference')
    c=np.asarray(coverage)
    if c.shape!=(len(reference),) or not np.isfinite(c).all() or c.min()<0 or c.max()>1 or c.sum()==0 or (1-c).sum()==0:
        raise ValueError('Require lawful nonempty reference FG/BG area coverage')
    transforms,prior_info=physical_prior(whole,native,device)
    side=int(len(whole)**.5);fine_side=2*side;fields={};fits={};heads={}
    for arm in ARMS:
        w,b,info=_fit(reference,c,transforms[arm],device)
        gw,gb=w.float().to(device),b.float().to(device)
        coarse=(whole.to(device)@gw+gb).cpu().reshape(side,side)
        fine=(native.to(device)@gw+gb).cpu().reshape(fine_side,fine_side)
        # Analytic latent field for equal physical-area squared losses of the
        # whole and fine observations. This classical readout is common to
        # all arms; it does not itself fix the query offset.
        correction=.5*(coarse-fine.reshape(side,2,side,2).mean((1,3)))
        latent=fine+correction.repeat_interleave(2,0).repeat_interleave(2,1)
        fields[arm]=dict(whole=coarse.numpy(),native=fine.numpy(),latent=latent.numpy())
        t,split=otsu_exact(latent.numpy());fits[arm]=info|dict(otsu_threshold=t,otsu=split)
        heads[arm]=dict(coefficient=w.numpy(),bias=np.asarray(b))
    return dict(fields=fields,fits=fits,heads=heads,prior=prior_info)
