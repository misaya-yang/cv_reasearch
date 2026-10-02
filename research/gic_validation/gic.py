"""Independent implementation of the report's fixed-marginal linear GIC model.

No image matcher, ground-truth-dependent tuning or native pose-backend claim.
All covariance inputs and residuals use the same declared measurement units.
"""
import numpy as np


def grid_basis(xy, grid=5):
    xy=np.asarray(xy,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or np.any(xy<0) or np.any(xy>1):
        raise ValueError('source coordinates must be Nx2 in [0,1]')
    if grid<2:raise ValueError('grid must be at least two')
    z=xy*(grid-1);lo=np.minimum(np.floor(z).astype(int),grid-2);frac=z-lo
    a=np.zeros((len(xy),grid*grid))
    for dx in [0,1]:
        for dy in [0,1]:
            w=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
            a[np.arange(len(xy)),(lo[:,1]+dy)*grid+lo[:,0]+dx]=w
    return a/np.linalg.norm(a,axis=1,keepdims=True)


def symmetric_root(cov):
    cov=np.asarray(cov,dtype=np.float64)
    if not np.allclose(cov,cov.swapaxes(-1,-2),rtol=1e-10,atol=1e-12):
        raise ValueError('covariance must be symmetric')
    ev,q=np.linalg.eigh(cov)
    if np.any(ev<=0):raise ValueError('covariance must be positive definite')
    return (q*np.sqrt(ev)[...,None,:])@q.swapaxes(-1,-2)


def components(xy,cov,normals,rho,grid=5):
    if not 0<=rho<1:raise ValueError('rho must be in [0,1)')
    a=grid_basis(xy,grid);normals=np.asarray(normals,dtype=np.float64)
    if not np.allclose(np.linalg.norm(normals,axis=1),1):raise ValueError('unit normals required')
    root=symmetric_root(cov)
    v=np.einsum('ni,nij,nj->n',normals,cov,normals)
    b=np.einsum('ni,nij->nj',normals,root)
    u=np.sqrt(rho)*np.einsum('ni,nj->nij',a,b).reshape(len(a),-1)
    return (1-rho)*v,u,v


def inverse_product(diag,u,x):
    diag=np.asarray(diag,dtype=np.float64)
    if np.any(diag<=0):raise ValueError('positive independent variance required')
    x=np.asarray(x,dtype=np.float64);single=x.ndim==1
    if single:x=x[:,None]
    dx=x/diag[:,None];du=u/diag[:,None]
    if not np.any(u):return dx[:,0] if single else dx
    v=dx-du@np.linalg.solve(np.eye(u.shape[1])+u.T@du,u.T@dx)
    return v[:,0] if single else v


def logdet(diag,u):
    chol=np.linalg.cholesky(np.eye(u.shape[1])+u.T@(u/diag[:,None]))
    return float(np.log(diag).sum()+2*np.log(np.diag(chol)).sum())


def gls_step(r,j,diag,u,damping=0.):
    sj=inverse_product(diag,u,j);sr=inverse_product(diag,u,r)
    return -np.linalg.solve(j.T@sj+damping*np.eye(j.shape[1]),j.T@sr)


def innovation_step(r,j,diag,u):
    h=j/np.sqrt(diag)[:,None];q,rh=np.linalg.qr(h,mode='reduced')
    if np.linalg.matrix_rank(rh)<j.shape[1]:raise ValueError('unidentifiable geometry')
    v=u/np.sqrt(diag)[:,None];vt=q.T@v;vn=v-q@vt
    y=r/np.sqrt(diag);yn=y-q@(q.T@y)
    z=np.linalg.solve(np.eye(vn.shape[1])+vn.T@vn,vn.T@yn)
    return -np.linalg.solve(rh,q.T@y-vt@z)


def profiled_reml(r,j,diag,u):
    nu=len(r)-j.shape[1]
    if nu<=0:raise ValueError('insufficient residual degrees of freedom')
    si=inverse_product(diag,u,np.column_stack([j,r]))
    h=j.T@si[:,:j.shape[1]];b=j.T@si[:,-1]
    chol=np.linalg.cholesky(h);ldh=2*np.log(np.diag(chol)).sum()
    residual=r-j@np.linalg.solve(h,b)
    quadratic=float(residual@inverse_product(diag,u,residual))
    if quadratic<=0:raise ValueError('zero/negative profiled residual variance')
    return .5*(logdet(diag,u)+ldh+nu*np.log(quadratic/nu))


def select_rho(r,j,xy,cov,normals,grid=5):
    """Deterministic bounded search; BIC fallback without labels.

    The 39 candidates are numerical optimization samples, not independent
    scientific variants. Does not provide a finite-sample safety guarantee.
    """
    def objective(rho):
        d,u,_=components(xy,cov,normals,float(rho),grid)
        return profiled_reml(r,j,d,u)
    nodes=np.linspace(0,.95,39);vals=np.array([objective(x) for x in nodes])
    k=int(vals.argmin());lo=nodes[max(0,k-1)];hi=nodes[min(len(nodes)-1,k+1)]
    ratio=(np.sqrt(5)-1)/2
    a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo);fa=objective(a);fb=objective(b)
    for _ in range(24):
        if fa<fb:hi,b,fb=b,a,fa;a=hi-ratio*(hi-lo);fa=objective(a)
        else:lo,a,fa=a,b,fb;b=lo+ratio*(hi-lo);fb=objective(b)
    opt=float((lo+hi)/2);fopt=objective(opt);f0=float(vals[0])
    accepted=2*(f0-fopt)>np.log(len(r))
    return (opt if accepted else 0.),dict(candidate_rho=opt,profiled_improvement=2*(f0-fopt),
                                        bic_penalty=float(np.log(len(r))),accepted=bool(accepted))


def covariance_from_precision(precision,width_in,height_in,width_out,height_out):
    """RoMaV2 pixel precision -> original-image pixel covariance.

    This deliberately does NOT multiply precision by a normalized [-1,1]
    coordinate factor: upstream precision is in refinement-image pixel units.
    """
    p=np.asarray(precision,dtype=np.float64)
    scale=np.array([width_in/width_out,height_in/height_out])
    pout=p*scale[None,:,None]*scale[None,None,:]
    symmetric_root(pout)  # validate SPD, without repairing upstream failures
    return np.linalg.inv(pout)


def two_end_field(xy_a,xy_b,cov_a,cov_b,grad_a,grad_b,grid=5):
    """Two independent image error fields, with one shared correlation rho.

    Gradients are the exact residual sensitivities in each image's pixel units;
    unlike B-only point-to-line normals, they need not individually be unit.
    Return rho-independent marginal diagonal and low-rank field.
    """
    aa=grid_basis(xy_a,grid);ab=grid_basis(xy_b,grid)
    ba=np.einsum('ni,nij->nj',grad_a,symmetric_root(cov_a))
    bb=np.einsum('ni,nij->nj',grad_b,symmetric_root(cov_b))
    ua=np.einsum('ni,nj->nij',aa,ba).reshape(len(aa),-1)
    ub=np.einsum('ni,nj->nij',ab,bb).reshape(len(ab),-1)
    field=np.column_stack([ua,ub]);marginal=(ba*ba).sum(1)+(bb*bb).sum(1)
    if np.any(marginal<=0):raise ValueError('zero residual uncertainty')
    return marginal,field


class SpectralREML:
    """One small eigendecomposition, scalar-only profiled rho evaluations.

    Restricted likelihood eliminates geometry in the marginal-whitened space.
    The reduced normal covariance is (1-rho)I + rho*Z*Z.T. Omitted additive
    constants are rho-independent; compare objective DIFFERENCES to full REML.
    """
    def __init__(self,r,j,marginal,field):
        self.n=len(r)
        self.nu=len(r)-j.shape[1]
        if self.nu<=0:raise ValueError('insufficient degrees of freedom')
        w=1/np.sqrt(marginal);h=w[:,None]*j
        q,rh=np.linalg.qr(h,mode='reduced')
        if np.linalg.matrix_rank(rh)<j.shape[1]:raise ValueError('unidentifiable geometry')
        y=w*r;y=y-q@(q.T@y)
        z=w[:,None]*field;z=z-q@(q.T@z)
        ev,evec=np.linalg.eigh(z.T@z)
        tol=max(float(ev.max()),1.)*1e-10
        if ev.min() < -tol:raise ValueError('invalid projected covariance')
        good=ev>tol;self.ev=ev[good]
        self.energy=((evec[:,good].T@(z.T@y))**2)/self.ev
        total=float(y@y);remainder=total-float(self.energy.sum())
        if remainder < -1e-8*max(total,1.):raise ValueError('invalid projected residual energy')
        self.remainder=max(remainder,0.)
    def objective(self,rho):
        if not 0<=rho<1:raise ValueError('rho out of range')
        independent=1-rho;den=independent+rho*self.ev
        quadratic=self.remainder/independent+float((self.energy/den).sum())
        if quadratic<=0:raise ValueError('zero residual energy')
        ld=(self.nu-len(self.ev))*np.log(independent)+float(np.log(den).sum())
        return .5*(ld+self.nu*np.log(quadratic/self.nu))
    def select(self):
        nodes=np.linspace(0,.95,39);vals=np.array([self.objective(x) for x in nodes]);k=int(vals.argmin())
        lo=nodes[max(0,k-1)];hi=nodes[min(len(nodes)-1,k+1)]
        ratio=(np.sqrt(5)-1)/2
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo);fa=self.objective(a);fb=self.objective(b)
        for _ in range(24):
            if fa<fb:hi,b,fb=b,a,fa;a=hi-ratio*(hi-lo);fa=self.objective(a)
            else:lo,a,fa=a,b,fb;b=lo+ratio*(hi-lo);fb=self.objective(b)
        rho=float((lo+hi)/2);improvement=2*(self.objective(0)-self.objective(rho))
        accepted=improvement>np.log(self.n)
        return (rho if accepted else 0.),dict(candidate_rho=rho,profiled_improvement=float(improvement),
              bic_penalty=float(np.log(self.n)),accepted=bool(accepted),normal_rank=len(self.ev))
