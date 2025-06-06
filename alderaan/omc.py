import numpy as np
from   scipy import stats
from   sklearn.cluster import KMeans

import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as exo
import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2.theano import GaussianProcess
from   celerite2.theano import terms as GPterms

from .constants import *


__all__ = ['matern32_model',
           'poly_model',
           'sin_model',
           'mix_model',
           'flag_outliers'
          ]


def matern32_model(xtime, yomc, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data
    Fits data with a regularized Matern-3/2 GP kernel
    
    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    xt_predict : ndarray
        time values to predict OMC model; if not provided xtime will be used

    Returns
    -------
    model : pm.Model()
    """ 
    with pm.Model() as model:
        # delta between each transit time
        dx = np.mean(np.diff(xtime))

       # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime
            
        # build the kernel
        log_sigma = pm.Normal('log_sigma', mu=np.log(np.std(yomc)), sd=5)
        rho = pm.Uniform('rho', lower=2*dx, upper=xtime.max()-xtime.min())
        
        kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)

        # regularization penalty
        pm.Potential('penalty', 2*np.log(rho))
        
        # mean and log(yvar) -- "diag" in GaussianProcess is diagonal variance
        mean = pm.Normal('mean', mu=np.mean(yomc), sd=np.std(yomc))
        log_yvar = pm.Normal('log_yvar', mu=np.log(np.var(yomc)), sd=10)

        # here's the gp and it's likelihood
        gp = GaussianProcess(kernel, t=xtime, diag=T.exp(log_yvar)*T.ones(len(xtime)), mean=mean)

        gp.marginal('gp', observed=yomc)

        # track GP prediction
        trend = pm.Deterministic('trend', gp.predict(yomc, xtime))
        pred  = pm.Deterministic('pred', gp.predict(yomc, xt_predict))
        
    return model


def poly_model(xtime, yomc, polyorder, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data 
    Fits data with a polynomial (up to cubic)
    
    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    polyorder : int
        polynomial order
    xt_predict : ndarray
        time values to predict OMC model; if not provided xtime will be used
        
    Returns
    -------
    model : pm.Model()
    """    
    with pm.Model() as model:
        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime
        
        C0 = pm.Normal('C0', mu=0, sd=10)
        C1 = 0.0
        C2 = 0.0
        C3 = 0.0

        if polyorder >= 1: C1 = pm.Normal('C1', mu=0, sd=10)
        if polyorder >= 2: C2 = pm.Normal('C2', mu=0, sd=10)
        if polyorder >= 3: C3 = pm.Normal('C3', mu=0, sd=10)
        if polyorder >= 4: raise ValueError("only configured for 3rd order polynomials")
        
        def poly_fxn(c0, c1, c2, c3, xt):
            return c0 + c1*xt + c2*xt**2 + c3*xt**3
        
        # mean and variance
        trend = pm.Deterministic('trend', poly_fxn(C0, C1, C2, C3, xtime))
        log_yvar = pm.Normal('log_yvar', mu=np.log(np.var(yomc)), sd=10)
        
        # here's the likelihood
        pm.Normal('obs', mu=trend, sd=T.sqrt(T.exp(log_yvar)*T.ones(len(xtime))), observed=yomc)
        
        # track predicted trend
        pred = pm.Deterministic('pred', poly_fxn(C0,C1,C2,C3,xt_predict))
            
    return model


def sin_model(xtime, yomc, period, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data
    Fits data with a single-frequency sinusoid
    
    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    period : float
        pre-estimated sinusoid period; the model places tight priors on this period
    xt_predict : ndarray
        time values to predict OMC model; if not provided xtime will be used

    Returns
    -------
    model : pm.Model()
    """    
    with pm.Model() as model:
        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # periodic component
        df = 1/(xtime.max()-xtime.min())
        f  = pm.Normal('f', mu=1/period, sd=df)
        Ah = pm.Normal('Ah', mu=0, sd=5*np.std(yomc))
        Bk = pm.Normal('Bk', mu=0, sd=5*np.std(yomc))
            
        def sin_fxn(A, B, f, xt):
            return A*T.sin(2*pi*f*xt) + B*T.cos(2*pi*f*xt)
        
        # mean and variance
        trend = pm.Deterministic('trend', sin_fxn(Ah, Bk, f, xtime))
        log_yvar = pm.Normal('log_yvar', mu=np.log(np.var(yomc)), sd=10)
        
        # here's the likelihood
        pm.Normal('obs', mu=trend, sd=T.sqrt(T.exp(log_yvar)*T.ones(len(xtime))), observed=yomc)
        
        # track predicted trend
        pred = pm.Deterministic('pred', sin_fxn(Ah,Bk,f,xt_predict))
            
    return model


def mix_model(x):
    """
    Build a 1D PyMC3 mixture model
    The model is composed of two normal distributions with the same mean but different variances
    
    Parameters
    ----------
    x : ndarray
        vector of data values
        
    Returns
    -------
    model : pm.Model()    
    """
    xnorm = x / np.std(x)
    xnorm -= np.mean(xnorm)
    
    with pm.Model() as model:
        
        # mixture parameters
        w = pm.Dirichlet('w', np.array([1.,1.]))
        mu = pm.Normal('mu', mu=0.0, sd=1.0, shape=1)
        tau = pm.Gamma('tau', 1.0, 1.0, shape=2)
        
        # here's the potential
        obs = pm.NormalMixture('obs', w, mu=mu*T.ones(2), tau=tau, observed=xnorm)
        
    return model


def flag_outliers(res, loc, scales):
    """
    Flag outliers in a residuals vector using a mixture model
    Applies unsupervised K-means clustering to assign points to either a narrow foreground or wide background distribution
    
    Parameters
    ----------
    res : ndarray (N)
        vector of residuals
    loc : float
        normal mean inferred from mix_model()
    scales : tuple
        normal standard deviations inferred from mix_model()
            
    Returns
    -------
    fg_prob : ndarray (N)
        probability the each item in res belongs to the foreground distribution
    bad : bool (N)
        binary classification of each item in res into foreground/background distribution
    """
    resnorm = res/np.std(res)
    resnorm -= np.mean(resnorm)
    
    order = np.argsort(scales)
    scales = scales[order]
    
    z_fg = stats.norm(loc=loc, scale=scales[0]).pdf(resnorm)
    z_bg = stats.norm(loc=loc, scale=scales[1]).pdf(resnorm)

    fg_prob = z_fg/(z_fg+z_bg)
    fg_prob = (fg_prob - fg_prob.min())/(fg_prob.max()-fg_prob.min())

    # use KMeans clustering to assign each point to the foreground or background
    km = KMeans(n_clusters=2)
    group = km.fit_predict(fg_prob.reshape(-1,1))
    centroids = np.array([np.mean(fg_prob[group==0]), np.mean(fg_prob[group==1])])
    bad = group == np.argmin(centroids)
        
    return fg_prob, bad