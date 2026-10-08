"""Training-derived projectors and bounded perturbations with matched controls."""
import numpy as np
from .numeric import Ridge

def partner_projector(h,cap):
    # Audit guide:
    # Fit delay/orientation prediction directions on supplied discovery histories,
    # convert standardized coefficients to memory coordinates, and orthonormalize with
    # SVD. Return at most three directions plus the discovery mean. Label-predictive
    # directions can also encode nuisance memories, so control perturbations are
    # necessary.
    #
    orientation=(cap[:,0]<cap[:,1]).astype(float)
    targets=np.c_[orientation,cap]
    model=Ridge().fit(h,targets,10.)
    directions=model.coef/model.scale[:,None]
    q,s,_=np.linalg.svd(directions,full_matrices=False)
    rank=min(3,int(np.sum(s>1e-6)))
    return q[:,:rank],h.mean(0)

def perturb(h,donor,q,reference,kind,magnitude,rng):
    # Audit guide:
    # Construct a carry change using whole donor difference, the partner-predictive
    # subspace, attenuation, or matched random/complement directions. Multiply by the
    # prespecified magnitude. Magnitude zero must preserve the recipient state exactly;
    # compare paired behavioral branches with identical physical starts and random keys.
    #
    difference=donor-h
    partner=q@(q.T@difference)
    if kind=='whole':delta=difference
    elif kind=='partner':delta=partner
    elif kind=='attenuate':delta=-q@(q.T@(h-reference))
    elif kind=='random':
        delta=rng.normal(size=h.shape);delta=delta/np.linalg.norm(delta)*np.linalg.norm(partner)
    elif kind=='nonpartner':
        delta=rng.normal(size=h.shape);delta-=q@(q.T@delta);delta=delta/max(np.linalg.norm(delta),1e-12)*np.linalg.norm(partner)
    elif kind=='sham':delta=partner
    else:raise ValueError(kind)
    return h+magnitude*delta
