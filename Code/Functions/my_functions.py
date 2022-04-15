
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import graphviz as gr
from matplotlib import style
import seaborn as sns
random.seed(10)


# In[3]:


def fn_variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)
# Note this is equivalent to np.var(Yt,ddof)
def fn_generate_cov(dim):
    acc  = []
    corr=0.5
    for i in range(dim):
        row = np.ones((1,dim)) * corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc,axis=0)

def fn_generate_multnorm(nobs,corr,nvar):

    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc = 1, scale = .5,size = (nvar,1))))**(1/2)
    # generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(mu[i],std[i],nobs),(nobs,-1)))
    
    normvars = np.concatenate(acc,axis=1)

    cov = fn_generate_cov(nvar)
    C = np.linalg.cholesky(cov)

    Y = np.transpose(np.dot(C,np.transpose(normvars)))


    return Y

def fn_randomize_treatment(N,k=0.5):
    # assign treatment randomly (0 or 1) where k is the percentage of treatment group
    treated = random.sample(range(N), round(N*k))
    return np.array([(1 if i in treated else 0) for i in range(N)]).reshape([N,1])


# In[4]:


def fn_tauhat_means(Yt,Yc):
    nt = len(Yt)
    nc = len(Yc)
    tauhat = np.mean(Yt)-np.mean(Yc)
    se_tauhat = (np.var(Yt,ddof=1)/nt+np.var(Yc,ddof=1)/nc)**(1/2)
    return (tauhat,se_tauhat)

def fn_bias_rmse_size(theta0,thetahat,se_thetahat,cval = 1.96):
    """
    theta0 - true parameter value
    thetatahat - estimated parameter value
    se_thetahat - estiamted se of thetahat
    """
    b = thetahat - theta0
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b**2))
    tval = b/se_thetahat # paramhat/se_paramhat H0: theta = 0
    size = np.mean(1*(np.abs(tval)>cval))
    # note size calculated at true parameter value
    return (bias,rmse,size)


def fn_plot_with_ci(n_values,tauhats,tau,lb,ub,caption):
    fig = plt.figure(figsize = (10,6))
    plt.plot(n_values,tauhats,label = '$\hat{\\tau}$')
    plt.xlabel('N')
    plt.ylabel('$\hat{\\tau}$')
    plt.axhline(y=tau, color='r', linestyle='-',linewidth=1,
                label='True $\\tau$={}'.format(tau))
    plt.title('{}'.format(caption))
    plt.fill_between(n_values, lb, ub,
        alpha=0.5, edgecolor='#FF9848', facecolor='#FF9848',label = '95% CI')
    plt.legend()


# In[5]:


def fn_generate_data(tau,N,p,p0,corr,conf,flagX):
    """
    Tau is the treatment effect
    N is the range of sample size 
    P is the number of covariates generated 
    p0 is the number of covariates included (non zero coefficient)
    corr is the correlation between the covariates
    conf is the confounder
    """
    nvar = p+2 # 1 confounder and 1 variable for randomizing treatment
    T = fn_randomize_treatment(N) # choose treated units
    corr = 0.5 # correlation for multivariate normal

    if conf==False:
        conf_mult = 0 # remove confounder from outcome
        allX = fn_generate_multnorm(N,corr,nvar)
        W0 = allX[:,0].reshape([N,1]) # variable for RDD assignment
        C = allX[:,1].reshape([N,1]) # confounder
        X = allX[:,2:] # observed covariates
        T = fn_randomize_treatment(N) # choose treated units
        err = np.random.normal(0,1,[N,1])
        beta0 = np.random.normal(5,5,[p,1])
        beta0[p0:p] = 0 # sparse model
        Yab = tau*T+X@beta0+conf_mult*0.6*C+err
        
        if flagX==False:
            return (Yab,T)

        else:
            return (Yab,T,X)
    else:
        conf_mult = 1 # include confounder from outcome
        allX = fn_generate_multnorm(N,corr,nvar)
        W0 = allX[:,0].reshape([N,1]) # variable for RDD assignment
        C = allX[:,1].reshape([N,1]) # confounder
        X = allX[:,2:] # observed covariates
        T = fn_randomize_treatment(N) # choose treated units
        err = np.random.normal(0,1,[N,1])
        beta0 = np.random.normal(5,5,[p,1])
        beta0[p0:p] = 0 # sparse model
        Yab = tau*T+X@beta0+conf_mult*0.6*C+err
        if flagX==False:
            return (Yab,T,C)

        else:
            return (Yab,T,X,C)
    
def fn_InBetween_generate_data(tau,N,flagZ):
    
    r = fn_randomize_treatment(N) # in between variable
    T = fn_randomize_treatment(N) # choose treated units
    Z = np.array([(1 if T[i]==1 & r[i]==1 else 0) for i in range(N)]).reshape([N,1]) # Variable effected by treatment Treatment  
    err = np.random.normal(0,1,[N,1])  
    Yab = tau*T+Z+err
    if flagZ==False:
        return (Yab,T)
    else:
        return (Yab,T,Z)


# In[6]:


def fn_run_experiments(tau,Nrange,p,p0,corr,conf,flagX):
    n_values = []
    tauhats = []
    sehats = []
    lb = []
    ub = []

    for N in tqdm(Nrange):
        n_values = n_values + [N]

        if flagX==False:
            if conf==False:
                Yexp,T = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            else:
                Yexp,T,C = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            Yt = Yexp[np.where(T==1)[0],:]
            Yc = Yexp[np.where(T==0)[0],:]
            tauhat,se_tauhat = fn_tauhat_means(Yt,Yc)            
        elif flagX==1:
            # use the right covariates in regression
            if conf==False:
                Yexp,T,X = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            else:
                Yexp,T,X,C = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            Xobs = X[:,:p0]
            covars = np.concatenate([T,Xobs],axis = 1)
            mod = sm.OLS(Yexp,covars)
            res = mod.fit()
            tauhat = res.params[0]
            se_tauhat = res.HC1_se[0]
        elif flagX==2:
            # use some of the right covariates and some "wrong" ones
            if conf==False:
                Yexp,T,X = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            else:
                Yexp,T,X,C = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            Yexp,T,X = fn_generate_data(tau,N,p,p0,corr,conf,flagX)
            Xobs1 = X[:,:np.int(p0/2)]
            Xobs2 = X[:,-np.int(p0/2):]
            covars = np.concatenate([T,Xobs1,Xobs2],axis = 1)
            mod = sm.OLS(Yexp,covars)
            res = mod.fit()
            tauhat = res.params[0]
            se_tauhat = res.HC1_se[0]

 
        tauhats = tauhats + [tauhat]
        sehats = sehats + [se_tauhat]    
        lb = lb + [tauhat-1.96*se_tauhat]
        ub = ub + [tauhat+1.96*se_tauhat]
        
    return (n_values,tauhats,sehats,lb,ub)

def fn_InBetween_run_experiments(tau,Nrange,flagZ):

    # Z is the variable in between the treatment and output
    n_values = []
    tauhats = []
    sehats = []
    lb = []
    ub = []
    
    for N in tqdm(Nrange):
        n_values = n_values + [N]
        if flagZ==False:
            Yexp,T = fn_InBetween_generate_data(tau,N,flagZ)
            Yt = Yexp[np.where(T==1)[0],:]
            Yc = Yexp[np.where(T==0)[0],:]
            tauhat,se_tauhat = fn_tauhat_means(Yt,Yc)            
        elif flagZ==1:
            # use the in between variable in regression
            Yexp,T,Z = fn_InBetween_generate_data(tau,N,flagZ)
            Zobs = Z[:,:1]
            covars = np.concatenate([T,Zobs],axis = 1)
            mod = sm.OLS(Yexp,covars)
            res = mod.fit()
            tauhat = res.params[0]
            se_tauhat = res.HC1_se[0]
            
        tauhats = tauhats + [tauhat]
        sehats = sehats + [se_tauhat]    
        lb = lb + [tauhat-1.96*se_tauhat]
        ub = ub + [tauhat+1.96*se_tauhat]
    return (n_values,tauhats,sehats,lb,ub)

