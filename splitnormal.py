import numpy as np
import scipy.special

stdErf=scipy.special.erf(1/np.sqrt(2))

def splitNormal(x,mu,sigma,cigma):
    epsilon=cigma/sigma
    alphas=sigma*np.ones_like(x)
    alphas[x>mu]=cigma
    return (1/np.sqrt(2*np.pi*sigma**2))*(2/(1+epsilon))*np.exp(-0.5*((x-mu)/alphas)**2)

def cdf(x,mu,sigma,cigma):
    epsilon=cigma/sigma
    alphas=sigma*np.ones_like(x)
    alphas[x>mu]=cigma
    epsFactor=np.ones_like(x)
    epsFactor[x>mu]=epsilon
    return (1/(1+epsilon))*(1 + epsFactor*scipy.special((x-mu) / sqrt(2 * alphas**2)) )
    
def inverse(F,mu,sigma,cigma):
    epsilon=cigma/sigma
    alphas=sigma*np.ones_like(F)
    alphas[F>1/(1+epsilon)]=cigma
    betas=np.ones_like(F)
    betas[F>1/(1+epsilon)]=1/epsilon
    return mu + np.sqrt(2 * alphas**2) * scipy.special.erfinv(betas*((1+epsilon)*F -1))
    
def random(n,mu,sigma,cigma):
    Fs=np.random.random(n)
    xs=inverse(Fs,mu,sigma,cigma)
    return xs

def fit(xs):
    xs=np.sort(xs)
    N=xs.size
    Delta=int(N*stdErf) #hardcoded version of erf(1/sqrt(2))
    
    js=np.arange(0,N-Delta-1)
    w_js=xs[js+Delta]-xs[js]
    J=np.argmin(w_js)
    w_J=w_js[J]
    x_J=xs[J]
    
    ks=np.arange(J+1,J+Delta-1)
    theta_ks=(ks/N) - ((xs[ks]-x_J)/w_J)
    K=ks[np.argmin(np.abs(theta_ks))]
    mu=xs[K]
    sigma=mu-x_J
    cigma=w_J-sigma

    return mu,sigma,cigma