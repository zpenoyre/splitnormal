import numpy as np

def splitNormal(x,mu,sigma,cigma):
    epsilon=cigma/sigma
    alphas=sigma*np.ones_like(x)
    alphas[x>mu]=cigma
    return (1/np.sqrt(2*np.pi*sigma**2))*(2/(1+epsilon))*np.exp(-0.5*((x-mu)/alphas)**2)

def fit(xs):
    xs=np.sort(xs)
    N=xs.size
    Delta=int(N*0.6826894921370859) #hardcoded version of erf(1/sqrt(2))
    
    js=np.arange(0,N-Delta-1)
    w_js=xs[js+Delta]-xs[js]
    J=np.argmin(w_js)
    w_J=w_js[J]
    x_J=xs[J]
    
    ks=np.arange(J,J+Delta-1)
    theta_ks=(ks/N) - ((xs[ks]-xs[J])/w_J)
    K=np.argmin(np.abs(theta_ks))
    mu=xs[ks[K]]
    sigma=mu-x_J
    cigma=w_J-sigma

    return mu,sigma,cigma