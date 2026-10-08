"""Train-only linear analyses with validation selection and grouped uncertainty."""
import numpy as np

class Ridge:
    def fit(self,x,y,alpha=1.):
        # Audit guide:
        # Standardize predictors using this fit data only and solve regularized least
        # squares for one or several targets. Save means/scales/offsets for unchanged
        # application to held-out examples. The penalty controls coefficient size; test
        # data must not estimate the scaling.
        #
        x,y=np.asarray(x,float),np.asarray(y,float)
        self.mean=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1
        self.offset=y.mean(0);z=(x-self.mean)/self.scale
        self.coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-self.offset))
        return self
    def predict(self,x):
        return ((x-self.mean)/self.scale)@self.coef+self.offset

def select_ridge(x,y,train,val):
    # Audit guide:
    # Try the fixed penalty grid using discovery fits and validation prediction error.
    # Return the chosen discovery-fitted model; do not fit or tune on the test mask.
    #
    best=None
    for alpha in (1.,10.,100.,1000.):
        model=Ridge().fit(x[train],y[train],alpha)
        error=np.mean((model.predict(x[val])-y[val])**2)
        if best is None or error<best[0]:best=(error,model,alpha)
    return best[1],best[2]

def r2(y,p):
    denom=np.sum((y-y.mean(0))**2)
    return float(1-np.sum((y-p)**2)/denom) if denom>1e-12 else None

def grouped_ci(values,groups,seed=0,n=200):
    # Audit guide:
    # Reduce values to group means and bootstrap those groups. For trajectory analyses
    # the group is an episode so correlated event samples stay together. This interval
    # does not automatically measure uncertainty across learner seeds.
    #
    groups=np.asarray(groups);unique=np.unique(groups)
    units=np.asarray([np.mean(np.asarray(values)[groups==g],axis=0) for g in unique])
    rng=np.random.default_rng(seed)
    draws=np.asarray([units[rng.integers(0,len(units),len(units))].mean(0) for _ in range(n)])
    return dict(mean=np.mean(units,axis=0).tolist(),low=np.quantile(draws,.025,axis=0).tolist(),high=np.quantile(draws,.975,axis=0).tolist(),n_units=len(units))

def cross_distance(first,second,precision):
    # Audit guide:
    # Multiply differences in profile means from independent splits and weight
    # coordinates by supplied noise precision. Unlike a squared Euclidean distance the
    # resulting cross-validated estimate can be negative. Precision must be estimated
    # from training evidence.
    #
    a=first[:,None,:]-first[None,:,:];b=second[:,None,:]-second[None,:,:]
    return np.sum(a*b*precision,axis=-1)/first.shape[-1]
