"""Pinned official InputDSA components with explicit boundary-safe DMDc pairing.

At upstream c437ce9 DMDc.compute_svd concatenates list trajectories BEFORE
forming X/Y, adding a spurious transition at every list boundary. This adapter
forms pairs within each trajectory and then concatenates the pairs. Upstream
operators and controllability distances otherwise remain unmodified.
"""
from pathlib import Path
import importlib.util
import sys
import types
import numpy as np
import torch

SOURCE=Path(__file__).parent/'reference/dsa'
package=types.ModuleType('_partner_dsa');package.__path__=[str(SOURCE)]
sys.modules.setdefault('_partner_dsa',package)
from _partner_dsa.dmdc import DMDc
from _partner_dsa.dmd import embed_signal_torch
from _partner_dsa.simdist_controllability import ControllabilitySimilarityTransformDist

class BoundaryDMDc(DMDc):
    def compute_svd(self):
        # Audit guide:
        # Construct current-state, next-state, and input pairs inside each contiguous
        # trajectory first, then concatenate the valid pairs. Joining raw trajectories
        # before pairing would invent a transition between unrelated rounds/episodes.
        # Use singular-value decompositions of state-plus-input predictors and next
        # states to obtain the bases for driven linear dynamics.
        #
        hs=self.H if isinstance(self.H,list) else list(self.H) if self.H.ndim==3 else [self.H]
        us=self.Hu if isinstance(self.Hu,list) else list(self.Hu) if self.Hu.ndim==3 else [self.Hu]
        if any(h.ndim!=2 for h in hs):raise ValueError('Use an explicit list of 2D contiguous segments')
        self.X=torch.cat([h[:-1] for h in hs],0).T
        self.Y=torch.cat([h[1:] for h in hs],0).T
        u=torch.cat([u[:-1] for u in us],0).T
        self.Omega=torch.vstack((self.X,u))
        self.Up,self.Sp,vh=torch.linalg.svd(self.Omega,full_matrices=False);self.Vp=vh.T
        n=self.X.shape[0];self.Up1=self.Up[:n];self.Up2=self.Up[n:]
        self.Ur,self.Sr,vh=torch.linalg.svd(self.Y,full_matrices=False);self.Vr=vh.T
        self.cumulative_explained_variance_input=self._compute_explained_variance(self.Sp)
        self.cumulative_explained_variance_output=self._compute_explained_variance(self.Sr)


def fit(segments,rank,delay=1,lamb=.01):
    # Audit guide:
    # Lift each trajectory into delay-coordinate states and fit the boundary-safe DMDc
    # model with an explicit rank and regularization. Short segments that cannot supply
    # valid delayed pairs are excluded. Rank/delay selection belongs to validation
    # evidence in dynamics.run, not held-out test scores.
    #
    segments=[(np.asarray(h,np.float32),np.asarray(u,np.float32)) for h,u in segments if len(h)>delay+1]
    if not segments:raise ValueError('No segments long enough for delay setting')
    model=BoundaryDMDc([s[0] for s in segments],[s[1] for s in segments],n_delays=delay,n_control_delays=1,
        rank_input=min(2*rank+segments[0][1].shape[1],segments[0][0].shape[1]*delay+segments[0][1].shape[1]),
        rank_output=rank,lamb=lamb,device='cpu')
    model.fit()
    return model


def pairs(segments,delay):
    # Audit guide:
    # Create delayed current/next-state pairs for scoring using the same within-segment
    # boundaries as fitting. Align the control with the transition it drives.
    # Concatenate only completed pairs so episode boundaries never become training
    # examples.
    #
    x=[];y=[];u=[]
    for h,control in segments:
        if len(h)<=delay+1:continue
        embedded=np.asarray(embed_signal_torch(np.asarray(h,np.float32),delay))
        x.append(embedded[:-1]);y.append(embedded[1:]);u.append(control[delay-1:-1])
    return np.concatenate(x),np.concatenate(y),np.concatenate(u)


def arrays(model):
    # Audit guide:
    # Export both reduced-coordinate and full lifted-coordinate state/input operators as
    # NumPy arrays. The reduced operators support similarity comparisons; full operators
    # predict the delay-lifted trajectory.
    #
    return {k:np.asarray(getattr(model,k).detach().cpu()) for k in ('A_v','B_v','A_havok_dmd','B_havok_dmd')}


def distance(a,b):
    # Audit guide:
    # Compare two driven systems using the pinned controllability metric and an explicit
    # input basis. Rename its joint/state/control return order to joint/intrinsic/input.
    # Learned encoder coordinates from separate networks cannot be treated as identical
    # inputs without an alignment or a common raw-input basis.
    #
    metric=ControllabilitySimilarityTransformDist(score_method='euclidean',compare='joint',align_inputs=False,return_distance_components=True)
    # Verified in pinned source: JOINT, STATE, CONTROL (not state, input, joint).
    joint,state,control=metric.fit_score(a['A_v'],b['A_v'],a['B_v'],b['B_v'])
    return dict(joint=float(joint),intrinsic=float(state),input=float(control))
