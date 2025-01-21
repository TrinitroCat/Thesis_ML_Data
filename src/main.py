'''
Version 0.8
2024/10/10
Pu Pengxin
'''

import pandas as pd  

import sys

import time


import numpy as np  

import scipy as sp
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution,dual_annealing
from scipy.special import comb,perm

import gc

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False  


import random
import copy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,GroupKFold  
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

'''
--------------------------------------------------------------------------------------------------------------------------------------
Main Code
--------------------------------------------------------------------------------------------------------------------------------------
'''

class Read(object):
    '''
    Read the CSV file.
    Method:
        rawdata(): read the labeled raw data, return X_raw,y_raw,X_name,y_name
        minmaxs_data(): read the labeled data and do min-max scale, return X_raw,y_raw,X_name,y_name
        unlabeled_data(): read the unlabeled raw data, return X_raw,X_name,y_name
        unlabeled_minmaxs_data(X_base=None): read the unlabeled data and do min-max scale. If X_base is not None, the min-max values are determined by X_base.
        
    X_raw, y_raw, X_name, y_name is the features, labels, features' name and sample's namea respectively.
    '''
    def __init__(self,Path):
        self.Path=Path
        pass
    
    def rawdata(self, depend_var_col=-1):
        '''
        Read data and do min-max scaler to each colunm.
        
        parameter:
            depend_var_col: int, the slice [:,depend_var_col:] will be set to y_raw
        
        return:
            X_raw,y_raw,X_name,y_name
        '''
        Path = self.Path
        df = pd.read_csv (Path)
        X_raw = df.values[:,1:depend_var_col]
        if depend_var_col == -1:
            y_raw = df.values[:,depend_var_col]
        else:
            y_raw = df.values[:,depend_var_col:]
        X_name = df.columns.values[1:depend_var_col]
        y_name = df.values[:,0]
        print('Tip: The returned X is a matrix with samples as rows and features as columns; y is a one-dimensional array of labels for the sample.')
        return X_raw,y_raw,X_name,y_name
    
    def minmaxs_data(self, X_base=None, depend_var_col=-1):
        '''
        parameters:
            depend_var_col: int, the slice [:,depend_var_col:] will be set to y_raw
            X_base: ndarray, the min-max scaler will fit on X_base and then transform input X
            
        return:
            X_st_raw,y_raw,X_name,y_name
        '''
        X_raw,y_raw,X_name,y_name=self.rawdata(depend_var_col)
        if X_base is None:
            X_base = X_raw
        temp = MinMaxScaler()
        temp.fit(X_base)
        X_straw = temp.transform(X_raw)
        return X_straw,y_raw,X_name,y_name
    
    def unlabeled_data(self):
        Path=self.Path
        df = pd.read_csv (Path)
        X_raw = df.values[:,1:]
        X_name = df.columns.values[1:]
        y_name = df.values[:,0]
        return X_raw, X_name, y_name
    
    def unlabeled_minmaxs_data(self,X_base=None):
        X_raw,X_name,y_name=self.unlabeled_data()
        if X_base is None:
            X_base=X_raw
        temp=MinMaxScaler()
        temp.fit(X_base)
        X_straw=temp.transform(X_raw)
        return X_straw,X_name,y_name
    
    def data(self):
        pass

'''
Feature Transform
---------------------------------------------------------------------------------------------------------------------------------------
'''
    
class Trans(object):                                            
    def __init__(self,X,y=None,X_name=None):
        self.X=np.asarray(X)
        self.X_name=X_name
        self.y=np.asarray(y)
        self.X_trans=np.asarray(X)
        self.X_trans_name=None
    
    def __plus(self,a,b,name=False):
        if name==True:
            N='(%s)+(%s)'%(a,b)
            return N
        else:
            return a+b

    def __diff(self,a,b,name=False):
        if name==True:
            N='abs((%s)-(%s))'%(a,b)
            return N
        else:
            return abs(a-b)

    def __mult(self,a,b,name=False):
        if name==True:
            N='(%s)*(%s)'%(a,b)
            return N
        else:
            return a*b

    def __divi(a,b):
        return a/(b+1)

    def __power2(self,a,name=False):
        if name==True:
            N='(%s)**2'%a
            return N
        else:
            return a**2

    def __power3(self,a,name=False):
        if name==True:
            N='(%s)**3'%a
            return N
        else:
            return a**3

    def __sqrt(self,a,name=False):
        if name==True:
            N='abs(%s)**0.5'%a
            return N
        else:
            return abs(a)**0.5

    def __abs_log(self,a,name=False):
        if name==True:
            N='np.log1p(abs(%s))'%a
            return N
        else:
            if a==-1.:
                print('Warning: Encountered log(0), return was set to 0.')
                return 0.
            else:
                return np.log1p(abs(a))
    
    def __exp(self,a,name=False):
        if name==True:
            N='np.exp(%s)'%a
            return N
        else:
            return np.exp(a)

    def poly_sklearn(self,degree,bia,name=False):
        X,X_name=self.X,self.X_name
        X=np.asarray(X)
        pol=PolynomialFeatures(degree,include_bias=bia)
        X_trans=pol.fit_transform(X)
        if name==True:
            X_trans_name=pol.feature_names_in_
            self.X_trans=X_trans
            self.X_trans_name=X_trans_name
            return
        else:
            self.X_trans=X_trans
            return
        
    def poly(self,name=False):
        X,X_name=self.X,self.X_name
        X=np.asarray(X)
        X_trans,X_trans_name=self.custom_func(unary=(self.__power2,),abelian=(self.__mult,),nonabel=(),name=name)
        print('### The binomial transformation ###')
        if name==True:
            return X_trans,X_trans_name
        else:
            return X_trans
    
    def elem_func(self,name=False):
        X,X_name=self.X,self.X_name
        for l in range(len(X)):
            X_non=np.array([])
            for f in [self.__exp,self.__abs_log,self.__power2,self.__power3,self.__sqrt]:
                for i in X[l,:]:
                    X_non=np.append(X_non,f(i))
            X_bin=np.array([])
            for f in [self.__plus,self.__diff,self.__mult]:
                k=1
                for i in X[l,:]:
                    X_r=X[l,k:]
                    for j in X_r:
                        X_bin=np.append(X_bin,f(i,j))
                    k=k+1
            k=0
            for i in X[l,:]:
                X_r=np.delete(X[l,:],k)
                for j in X_r:
                    if j!=-1.:
                        X_bin=np.append(X_bin,i/(j+1))
                    else:
                        X_bin=np.append(X_bin,0.)
                        print('Warning: Encountered 1/0, return was set to 0.')
                k=k+1
            X_temp=np.hstack((X[l,:],X_non,X_bin))
            if l==0:
                X_trans=[X_temp]
            else:
                X_trans=np.vstack((X_trans,X_temp))
            del X_temp
        print('### Elementary functions transform, including +、-、×、*/(*+1)、^2、^3、sqrt(|*|)、exp、log|1+*| ###')
        if name==True:
            X_non_na=np.array([])
            for f in [self.__exp,self.__abs_log,self.__power2,self.__power3,self.__sqrt]:
                for i in X_name:
                    X_non_na=np.append(X_non_na,f(i,name==True))
            X_bin_na=np.array([])
            for f in [self.__plus,self.__diff,self.__mult]:
                k=1
                for i in X_name:
                    X_r=X_name[k:]
                    for j in X_r:
                        X_bin_na=np.append(X_bin_na,f(i,j,name==True))
                    k=k+1
            k=0
            for i in X_name:
                X_r=np.delete(X_name,k)
                for j in X_r:
                    N='(%s)/(%s+1)'%(i,j)
                    X_bin_na=np.append(X_bin_na,N)
                k=k+1
            X_trans_name=np.hstack((X_name,X_non_na,X_bin_na))
            return X_trans,X_trans_name
        else:
            return X_trans
        
    def custom_func(self,unary=(),abelian=(),nonabel=(),name=False):   
        X,X_name=self.X,self.X_name
        for l in range(len(X)):
            X_non=np.array([])
            for f in unary:
                for i in X[l,:]:
                    X_non=np.append(X_non,f(i))
            X_bin=np.array([])
            for f in abelian:
                k=0
                for i in X[l,:]:
                    X_r=X[l,k:]
                    for j in X_r:
                        X_bin=np.append(X_bin,f(i,j))
                    k=k+1
            for f in nonabel:
                for i in X[l,:]:
                    for j in X[1,:]:
                        X_bin=np.append(X_bin,f(i,j))
            X_temp=np.hstack((X[l,:],X_non,X_bin))
            if l==0:
                X_trans=[X_temp]
            else:
                X_trans=np.vstack((X_trans,X_temp))
            del X_temp
        if name==True:
            X_non_na=np.array([])
            for f in unary:
                for i in X_name:
                    X_non_na=np.append(X_non_na,f(i,name==True))
            X_bin_na=np.array([])
            for f in abelian:
                k=0
                for i in X_name:
                    X_r=X_name[k:]
                    for j in X_r:
                        X_bin_na=np.append(X_bin_na,f(i,j,name==True))
                    k=k+1
            for f in nonabel:
                for i in X_name:
                    for j in X_name:
                        X_bin_na=np.append(X_bin_na,f(i,j))
            X_trans_name=np.hstack((X_name,X_non_na,X_bin_na))
            return X_trans,X_trans_name
        else:
            return X_trans
    
    def associa(self):
        X,X_name=self.X,self.X_name
        if len(X.shape)==1:
            trans_data={}
            for i in range(len(X_name)):
                trans_data[X_name[i]]=X[i]
            return trans_data
        elif len(X.shape)!=2:
            raise ValueError('X must be a vector or matrix')
        trans_data={}
        for i in range(len(X_name)):
            trans_data[X_name[i]]=X[:,i]
        return trans_data
    
    def gauss_noise(self,n,mu=0,sigma=1,alpha=0.01,seed=0):
        np.random.seed(seed)
        Xn=self.X
        yn=self.y
        for i in range(n):
            X_noise=np.random.normal(mu,sigma,np.shape(self.X))*alpha
            Xn=np.r_[Xn,self.X+X_noise]
            yn=np.r_[yn,self.y]
        return Xn,yn

'''
Feature Screening
---------------------------------------------------------------------------------------------------------------------------------------
'''
        
class Select(object):
    '''
    Feature selected.
    
    Methods: 
        SIS: sure independent screening. 
        selfcorr: column-wise correlation coefficient within `X`.
        selfcorr_remove: column-wise correlation coefficient within `X`, and delete one of the column pair with corr. coeff. greater than `thres`.
        wrap: calculating the features' permutation importance of given model, and select the first `size` ones.
        RFE: Recursive feature elimination.
    '''
    def __init__(self,X,y,X_name=None):
        self.X=np.asarray(X,dtype=np.float64)
        self.y=np.asarray(y,dtype=np.float64)
        self.X_name=np.asarray(X_name)
        self.X_slec,self.slec_r,self.slec_name,self.slec_args=np.array([]),np.array([]),np.array([]),np.array([])
        self.X_sieve={}
        self.iter_MSE=np.array([])
    
    def PCCs(self,X,y):
        X,y=np.asarray(X),np.asarray(y)
        if len(X.shape)==1:
            X=(np.atleast_2d(X)).T
        elif len(X.shape)!=2:
            raise ValueError('X must be a vector or matrix')
        PCCs=np.array([])
        ymean=np.mean(y)
        for i in range(len(X[0,:])):
            Xmean=np.mean(X[:,i])
            yk,cov,varx,vary=0,0.,0.,0.
            for j in X[:,i]:
                a,b=(j-Xmean),(y[yk]-ymean)
                cov=cov+a*b
                varx=varx+a**2
                vary=vary+b**2
                yk=yk+1
            c=varx*vary
            if c==0.:
                PCCs=np.append(PCCs,0.)
            else:
                PCCs=np.append(PCCs,cov/(c**0.5))
        if len(PCCs)==1:
            PCCs=PCCs[0]
        return PCCs
    
    def distcorr(self,Xi,Y):
        Xi=np.array(Xi)
        Y=np.array(Y).flatten()
        if np.prod(Xi.shape)==len(Xi):
            Xi=Xi[:,None]
        if np.prod(Y.shape)==len(Y):
            Y=Y[:,None]
        Xi=np.atleast_2d(Xi)
        Y=np.atleast_2d(Y)
        n=Xi.shape[0]
        if Y.shape[0]!=Xi.shape[0]:
            raise ValueError('Number of samples must match')
        a=squareform(pdist(Xi))
        b=squareform(pdist(Y))
        A=a-a.mean(axis=0)[None,:]-a.mean(axis=1)[:,None]+a.mean()
        B=b-b.mean(axis=0)[None,:]-b.mean(axis=1)[:,None]+b.mean()
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        if dcov2_xx*dcov2_yy==0:
            dcor=0.
        else:
            dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor
    
    def squeue(self,index,n,name=False):
        X=self.X
        titel=np.argsort(index)
        titel_slec=titel[-n:]
        lenth=len(X[:,0])
        for l in range(lenth):
            for i in titel_slec:
                self.X_slec=np.append(self.X_slec,X[l,i])
        self.X_slec=np.reshape(self.X_slec,(lenth,n))
        for i in titel_slec:
            self.slec_r=np.append(self.slec_r,index[i])
        self.slec_args=titel_slec
        if name==True:
            if len(self.X_name)!=len(X[0,:]):
                raise ValueError('Number of samples and its names must match')
            for i in titel_slec:
                self.slec_name=np.append(self.slec_name,self.X_name[i])
            return self.X_slec,self.slec_r,self.slec_name,titel_slec
        else:
            return self.X_slec,self.slec_r,titel_slec
        
    def SIS(self,method='pearson',size=0,name=True):
        X,y=self.X,self.y
        X_sieve=self.X_sieve
        if method=='pearson':
            temp=self.PCCs(X,y)
            temp=abs(temp)
            X_temp=self.squeue(temp,size,name)
            if name==True:
                for i,na in enumerate(['X','index','name','args']):
                    X_sieve[na]=X_temp[i]
            else:
                for i,na in enumerate(['X','index','args']):
                    X_sieve[na]=X_temp[i]
            return X_sieve
        elif method=='distance':
            temp=np.array([])
            for i in range(len(X[0,:])):
                temp=np.append(temp,self.distcorr(X[:,i],y))
            temp=abs(temp)
            X_temp=self.squeue(temp,size,name)
            if name==True:
                for i,na in enumerate(['X','index','name','args']):
                    X_sieve[na]=X_temp[i]
            else:
                for i,na in enumerate(['X','index','args']):
                    X_sieve[na]=X_temp[i]
            return X_sieve
        elif method=='unknown':
            pass
    
    def selfcorr(self,method='pearson',name=False):
        '''
        column-wise correlation coefficient within `X`.
        Args:
            method: str, correlation metrics. 'pearson' or 'distance'
            name: boolean, whether to output args name
        '''
        X=self.X
        n=len(X[0,:])
        corr=np.zeros((n,n))
        if method=='pearson':
            f=self.PCCs
        elif method=='distance':
            f=self.distcorr
        elif method=='unknown':
            pass
        k=0
        for i in X[0,:]:
            X_r=X[0,k:]
            r=0
            for j in X_r:
                corr[k,k+r]=f(X[:,k],X[:,k+r])
                r=r+1
            k=k+1
        if name==False:
            return corr
        elif name==True:
            if len(self.X_name)!=len(X[0,:]):
                raise ValueError('Number of samples and its names must match')
            name_corr={}
            k=0
            for i in self.X_name:
                X_r=X_name[k:]
                r=0
                for j in X_r:
                    name_corr['(%s,%s)'%(i,j)]=corr[k,k+r]
                    r=r+1
                k=k+1
            return name_corr
        
    def selfcorr_remove(self,method='pearson',thres=0.8,name=False):
        '''
        column-wise correlation coefficient within `X`, and delete one of the column pair with corr. coeff. greater than `thres`.
        Args:
            method: str, correlation metrics. 'pearson' or 'distance'
            name: boolean, whether to output args name
        '''
        args=np.array([])
        n_feat=len(self.X[0,:])
        corr=self.selfcorr(method,name=False)
        for i,listval in enumerate(corr):
            corr_r=listval[i:]
            for j,val in enumerate(corr_r):
                if ((abs(val)>thres) and (j!=0)):
                    args=np.append(args,int(i+j))
        
        if len(args)==0:
            print('Notice: No feature was removed.')
        
        args=set(range(n_feat)).difference(set(args))
        args=np.asarray(list(args))
        self.X_slec=self.X[:,args]
        if name==True:
            self.slec_name=self.X_name[args]
            return self.X_slec, self.slec_name
        else:
            return self.X_slec
    
    def wrap(self,model,size=0,name=False,repeats=10,seed=None,score='neg_mean_squared_error'):
        X,y=self.X,self.y
        X_sieve=self.X_sieve
        reg = model.fit(X,y)
        y_pred=reg.predict(X)
        valua=permutation_importance(model, X, y, n_repeats=repeats,random_state=seed,scoring=score)
        importance=valua.importances_mean
        X_temp=self.squeue(importance,size,name)
        r2=r2_score(y,y_pred)
        MSE=mean_squared_error(y,y_pred)
        if name==True:
            for i,na in enumerate(['X','index','name','args']):
                    X_sieve[na]=X_temp[i]
        else:
            for i,na in enumerate(['X','index','args']):
                    X_sieve[na]=X_temp[i]
        return X_sieve
    
    def RFE(self,model,n_elim=1,maxit=1E5,eps_mean_MSE=0.,eps_max_MSE=0.,eps_mean_R2=0.,eps_min_R2=0.,cv=10,seed=None,repeats=10,imp_score='neg_mean_squared_error',name=True):
        '''
       Recursive feature elimination, with the evaluation of permutation importance and cross-validation.
        
        参数: 
            model: model (particularly in sklearn) with methods `pred` and `fit`.
            n_elim: feature number per elimination.
            maxit: max iter. number.
            cv: fold number of cross-validation.
            seed: random seed.
            repeats: repeatitions of permutation importance.
            imp_score: evaluation score of permut. importance.
            name: whether output feature names.
        '''
        X,y=self.X,self.y
        n_feat=len(X[0,:])
        self.iter_R2=np.array([])
        X_temp=X
        n_=int(n_elim)
        cvMSE_min=1E5
        r=1
        
        if name==True:
            Name=self.X_name
        else:
            Name=None
            
        if (n_elim<1) or n_elim>=n_feat:
            raise ValueError('n_elim must between 1 and n_features')
        elif type(n_elim)!=int or type(cv)!=int or type(repeats)!=int:
            raise TypeError('n_elim, cv and repeats must be integer')
            
        while True:
            if r>maxit:
                print('*'*132,'\n\n### Reach the maximum iteration numbers. MAIN LOOP DONE. ###')
                break
            print('*'*132,'\n\n','### iteration %i ###\n'%r)
            cv_model=CV(model,cv,importance=False,gauss=False,gauss_para=(4,0.,1.,0.01),seed=seed)
            cv_model.run(X_temp,y)
            self.iter_MSE=np.append(self.iter_MSE,cv_model.mean_MSE)
            self.iter_R2=np.append(self.iter_R2,cv_model.cv_R2)
            print('\n')
            print('### val. mean R^2= ',cv_model.cv_R2,'###')
            print('### val. min. R^2= ',cv_model.min_R2,'###')
            print('### val. mean MSE= ',cv_model.mean_MSE,'###')
            print('### val. max. MSE= ',cv_model.max_MSE,'###')
            print('\n')
            if cv_model.mean_MSE<cvMSE_min:
                cvMSE_min=copy.deepcopy(cv_model.mean_MSE)
                cvR2_min=copy.deepcopy(cv_model.cv_R2)
                X_opt=copy.deepcopy(X_temp)
                X_opt_name=copy.deepcopy(Name)
            if ((cv_model.mean_MSE<eps_mean_MSE)and(cv_model.max_MSE<eps_max_MSE)\
            and(cv_model.cv_R2>eps_mean_R2)and(cv_model.min_R2>eps_min_R2)):
                print('*'*132,'\n\n### Converged. ###')
                break
            X_s=Select(X_temp,y,X_name=Name).wrap(model,size=n_feat,name=name,repeats=repeats,seed=seed,score=imp_score)
            arg=X_s['args'][:n_]
            X_temp=np.delete(X_temp,arg,axis=1)
            if name==True:
                Name=np.delete(Name,arg)
            n_feat=len(X_temp[0,:])
            print('Rest %i features'%n_feat)
            if n_feat<=n_:
                n_=1
            if n_feat==1:
                print('*'*132,'\n### Only 1 feature rest. MAIN LOOP DONE. ###\n')
                break
            r=r+1
        print('*'*34)
        print('* Minimum cross-validation mean MSE: %f *\n* Corresponding R^2: %f         *'%(cvMSE_min,cvR2_min))
        print('*'*34)
        self.X_slec=X_opt
        self.slec_name=X_opt_name
        return
    
    def sym_RFE(self,model,n_elim=1,maxit=1E5,eps_mean_MSE=0.,eps_max_MSE=0.,eps_mean_R2=0.,eps_min_R2=0.,cv=10,seed=None,repeats=10,imp_score='neg_mean_squared_error',name=True,n_group = 1):
        '''
        对称的递归特征消除，使用排列重要性(permutation importance)评价特征重要性，交叉验证评价模型效果
        
        参数: 
            model: 模型
            n_elim: 每次消除的特征数
            maxit: 最大迭代步数
            cv: 交叉验证折数
            seed: 随机种子
            repeats: 排列重要性的重复次数
            imp_score: 排列重要性的评价指标
            name: 是否输出名字
            n_group: 将输入的X, y等分为n_group份，并系统抽样作为交叉验证的划分
        '''
        X,y=self.X,self.y
        n_feat=len(X[0,:])
        self.iter_R2=np.array([])
        X_temp=X
        n_=int(n_elim)
        cvMSE_min=1E5
        r=1
        
        if name==True:
            Name=self.X_name
        else:
            Name=None
            
        if (n_elim<1) or n_elim>=n_feat:
            raise ValueError('n_elim must between 1 and n_features')
        elif type(n_elim)!=int or type(cv)!=int or type(repeats)!=int:
            raise TypeError('n_elim, cv and repeats must be integer')
            
        while True:
            if r>maxit:
                print('*'*132,'\n\n###已达最大步数，迭代结束###')
                break
            print('*'*132,'\n\n','###第%i轮迭代###\n'%r)
            cv_model=CV(model,cv,importance=False,gauss=False,gauss_para=(4,0.,1.,0.01),seed=seed)
            cv_model.sym_run(X_temp,y,n_group)
            self.iter_MSE=np.append(self.iter_MSE,cv_model.mean_MSE)
            self.iter_R2=np.append(self.iter_R2,cv_model.cv_R2)
            print('\n')
            print('### 验证集平均R^2= ',cv_model.cv_R2,'###')
            print('### 验证集最小R^2= ',cv_model.min_R2,'###')
            print('### 验证集平均MSE= ',cv_model.mean_MSE,'###')
            print('### 验证集最大MSE= ',cv_model.max_MSE,'###')
            print('\n')
            if cv_model.mean_MSE<cvMSE_min:
                cvMSE_min=copy.deepcopy(cv_model.mean_MSE)
                cvR2_min=copy.deepcopy(cv_model.cv_R2)
                X_opt=copy.deepcopy(X_temp)
                X_opt_name=copy.deepcopy(Name)
            if ((cv_model.mean_MSE<eps_mean_MSE)and(cv_model.max_MSE<eps_max_MSE)\
            and(cv_model.cv_R2>eps_mean_R2)and(cv_model.min_R2>eps_min_R2)):
                print('*'*132,'\n\n###收敛成功###')
                break
            X_s=Select(X_temp,y,X_name=Name).wrap(model,size=n_feat,name=name,repeats=repeats,seed=seed,score=imp_score)
            arg=X_s['args'][:n_]
            X_temp=np.delete(X_temp,arg,axis=1)
            if name==True:
                Name=np.delete(Name,arg)
            n_feat=len(X_temp[0,:])
            print('剩余特征数%i个'%n_feat)
            if n_feat<=n_:
                n_=1
            if n_feat==1:
                print('*'*132,'\n###仅剩一个特征，迭代结束###\n')
                break
            r=r+1
        print('*'*34)
        print('*验证集cv平均MSE最小值: %f *\n*对应的平均R^2: %f         *'%(cvMSE_min,cvR2_min))
        print('*'*34)
        self.X_slec=X_opt
        self.slec_name=X_opt_name
        return
    
    def RFS(self,model,feature_size_range=(1,1),maxit=100,cv=10,name=True):
        '''
        随机特征选择, 迭代地在输入特征中随机选择个数在区间feature_size_range之内的特征交叉验证, 保留效果最好的那一组
        
        参数:
        model: calss, 模型
        feature_size_range: tuple, 所需特征数目的范围
        maxit: int, 迭代步数
        name: bool, 是否输出名字
        '''
        X=self.X
        y=self.y
        n_samp,n_feat=np.shape(X)
        MSE_old=1E8
        mem=set()
        repeat_n=0
        iter_MSE=[]
        iter_R2=[]
        
        if maxit<1:
            raise ValueError('maxit must be greater than 1 !!!')
        elif len(feature_size_range)!=2:
            raise ValueError('feature_size_range must be a tuple of lenth 2')
        elif feature_size_range[0]>feature_size_range[1]:
            raise ValueError('The left endpoint of the feature_size_range is greater than the right endpoint')
        elif feature_size_range[1]>n_feat:
            raise ValueError('feature_size_range is out of input features number !!!')
        elif feature_size_range[0]<0:
            raise ValueError('feature size must be a positive integer !!!')
        elif type(name)!=bool:
            raise TypeError('name must be a boolean !!!')
        elif cv<=1 or cv>n_samp:
            raise ValueError('Number of CV folds must between 2 and the number of samples')
        maxit=int(maxit)
        cv=int(cv)
        min_feat_size=int(feature_size_range[0])
        max_feat_size=int(feature_size_range[1])
        
        print('Variables initialized. Starting main loop...')
        
        for i in range(maxit):
            rand_size=np.random.randint(min_feat_size,high=max_feat_size)
            choice_args=np.random.choice(range(n_feat),size=rand_size,replace=False)
            
            judge=hash(tuple(set(choice_args)))
            if judge in mem:
                repeat_n+=1
                continue
            mem.add(judge)
            X_chosen=X[:,choice_args]
            
            cv_model=CV(model,cv,importance=False,gauss=False,gauss_para=(4,0.,1.,0.01),seed=None)
            cv_model.run(X_chosen,y,print_=False)
            R2=cv_model.cv_R2
            MSE=cv_model.mean_MSE
            
            if MSE<MSE_old:
                X_opt_args=copy.deepcopy(choice_args)
                MSE_old=copy.deepcopy(MSE)
                iter_MSE.append(MSE)
                iter_R2.append(R2)
                print('*********** Iteration %i ***********\n CV_MSE = %.3f  ,  CV_R2 = %.2f  \n'%(i+1,MSE,R2))
                
        print('\n\n'+'*'*130,'\nMax iteration was reached','\nMin CV_MSE: %.3f'%MSE_old,'Number of features: %i'%(len(X_opt_args)),'\nChoice repetitions times: %i'%repeat_n,'\n'+'*'*130)
        self.iter_MSE=iter_MSE
        self.iter_R2=iter_R2
        self.X_slec=X[:,X_opt_args]
        if name==True:
            self.slec_name=self.X_name[X_opt_args]
        return
        
            
    def RFA(self,model,n_elim=1,maxit=100,SIS_size=-1,eps_mean_MSE=0.1,eps_max_MSE=1.,eps_mean_R2=0.,eps_min_R2=0.,cv=10,seed=None,repeats=10,name=True):
        X,y=self.X,self.y
        n_feat=len(X[0,:])
        n_samp=len(X)
        X_se_seq=np.empty((0,n_samp))
        if name==True:
            Name=self.X_name
        else:
            Name=None
        if (n_elim<1) or n_elim>=n_feat:
            raise ValueError('n_elim must between 1 and n_features')
        if SIS_size==-1:
            SIS_size=3*n_samp
        X_temp=X
        n_=int(n_elim)
        cvMSE_min=1E5
        r=0
        resi=y
        pass
            
        

'''
hyperpara. search
--------------------------------------------------------------------------------------------------------------------------------------
'''        
class Search(object):
    def __init__(self,model_func,s_args=(),cv=10,thres='MSE',seed=None):
        self.model_func=model_func
        self.cv=cv
        self.seed=seed
        self.thres=thres
        self.s_args=s_args
        self.opt_para=None
        self.__i=1
        
    def _SAA(self,func,bounds,args=None,x0=None,integer_args=-1,maxit=500,init_temp=5230.0,temp_descent_factor=1.62,fin_temp=0.5,print_=True):
        '''
        模拟退火算法求最小值，X扰动采用柯西分布，T降温采用快速退火: T = T0*(2^temp_descent_factor - 1)/((1 + t)^temp_descent_factor - 1)
        
        参数:
            func: 待优化函数
            bounds: tuple, 优化的自变量取值范围
            args: 不参与优化的函数的其它变量
            x0: tuple, 初始点
            integer_args: int, 指定前 integer_args 个变量是整数
            maxit: int, 每个温度下迭代的次数
            init_temp: float, 初始温度
            temp_descent_factor: float, 温度下降因子
            fin_temp: float, 终止温度
        '''
        X_dim=len(bounds)
        integer_args=int(integer_args)
        if x0 is None:
            print('x0 is not read, select a random point within bounds as x0')
            x0=np.zeros(X_dim)
            for i,boundi in enumerate(bounds):
                if i<integer_args:
                    x0[i]=np.random.randint(boundi[0],high=boundi[1])
                else:
                    x0[i]=(boundi[1]-boundi[0])*np.random.random()+boundi[0]   
        elif len(x0)!=X_dim:
            raise ValueError('Dimensions of x0 and bounds do not match !!!')
        x0=np.array(x0)
        func_old=func(*x0,*args)
        x_old=x0
        x_new=x0
        temp=init_temp
        t=0
        itera=np.array(range(maxit),dtype=int)
        gamma=temp
        
        if maxit<=0:
            raise ValueError('maxit must be a positive integer !!!')
        elif init_temp<=0 or fin_temp<=0:
            raise ValueError('temperature must be positive numbers !!!')
        elif init_temp<=fin_temp:
            raise ValueError('init_temp must be greater than fin_temp !!!')
        elif temp_descent_factor<1E-5:
            raise ValueError('temp_descent_factor must greater than 0. !!!')
            
        maxit=int(maxit)
        
        print('Variables initialized, entering main loop...')
        S_time=time.time()
        while temp>=fin_temp:
            init_time=time.time()
            for i in itera:
                while True:
                    for j,boundi in enumerate(bounds):
                        if j<integer_args:
                            x_new[j]=np.random.randint(boundi[0],high=boundi[1])
                        else:
                            x_new[i]=(boundi[1]-boundi[0])*np.random.random()+boundi[0]
                    dist_p=(1/(gamma*np.pi))*np.random.random()
                    cauchy_x=1/(gamma*np.pi*(1+(np.linalg.norm(x_new-x_old)/gamma)**2))
                    if dist_p<=cauchy_x:
                        break
                func_new=func(*x_new,*args)
                if func_new<func_old:
                    func_old=copy.deepcopy(func_new)
                    x_old=copy.deepcopy(x_new)
                    if func_new<func_best:
                        func_best=copy.deepcopy(func_new)
                        x_best=copy.deepcopy(x_new)
                elif np.random.random()<np.exp(-(func_new-func_old)/temp):
                    func_old=copy.deepcopy(func_new)
                    x_old=copy.deepcopy(x_new)
            t+=1
            temp=init_temp*(2**temp_descent_factor - 1)/((1 + t)**temp_descent_factor - 1)
            end_time=time.time()
            if print_==True:
                print('\n'+'*'*50+'Iteration  %i'%t+'*'*50,'\nTemperature: %.2f  ,  min function value: %.4f  ,  optimize x: %s  ,  Time: %f'%(temp,func_best,x_best,end_time-init_time))
        
        E_time=time.time()
        print('*'*130)
        print('*Iteration Terminated.\n*Min Value of function: %.4f \n*Total Time: %f'%(func_best,E_time-S_time))
        print('*'*130)
        return x_best
        
    
    def __func(self,x_args,s_args):
        model=self.model_func(*x_args,*self.s_args)
        print('*'*138,'\n第%i次迭代\n'%self.__i,'*'*138)
        cv_val=CV(model,cv=self.cv,importance=False,gauss=False,gauss_para=(2,0.,1.,0.01),seed=self.seed)
        cv_val.run(self.X,self.y)
        print('\nCV_mean_MSE= %f'%cv_val.mean_MSE,'\nCV_mean_R^2= %f'%cv_val.cv_R2)
        self.__i=self.__i+1
        if self.thres=='MSE':
            return cv_val.mean_MSE
        elif self.thres=='R2':
            return -cv_val.cv_R2
    
    def optimiser(self,X,y,bounds='auto',method='diff_evol',maxiter=1000,tol=0.01):
        self.X,self.y=X,y
        func=self.__func
        if bounds=='auto':
            bounds=[]
            for i in self.x_args:
                bounds.append((0.01,5))
        if method=='diff_evol':        
            A=differential_evolution(func,bounds,args=self.s_args,strategy='best1bin',maxiter=maxiter,popsize=15,\
                               tol=tol,mutation=(0.5,1),recombination=0.7,seed=self.seed,callback=None,\
                               disp=True,polish=True,init='latinhypercube',atol=0,updating='immediate',\
                               workers=1)
        elif method=='annealing':
            A=dual_annealing(func, bounds, args=self.s_args, maxiter=maxiter, initial_temp=5230.0, \
                             restart_temp_ratio=2e-05, visit=2.62, accept=-5.0, maxfun=10000.0, seed=self.seed, \
                             no_local_search=False, x0=None)
        
        self.opt_para=A.x
        self.__i=1
        return
    
    def brute(self,n_grid,evaluate='CV_MSE',disp=True):
        fun=self.fun
        interval=self.interval
        args=self.args
        
            
'''
Cross-Validation
--------------------------------------------------------------------------------------------------------------------------------------
'''
    
class CV(object):
    '''
    Cross-Validation. 
    Additionally use importance=True to output the Permutation-Importances in attribute importances, imp_repeat to control the random permutation times.
                 use gauss=True to add gaussian-noise in X, gauss_para=(n,mu,sigma,alpha) to control the gaussian-noise distribution.
    
    Method:
        run(X_raw,y_raw,X_name=False,print_=True): To run CV and fill attributes.
    
    Attribute:
        mean_MSE
        mean_MAE
        mean_MAPE
        max_MSE
        cv_R2
        min_R2
        importances
    '''
    mean_MSE=None
    mean_MAE,mean_MAPE=None,None
    max_MSE=None
    cv_R2=None
    min_R2=None
    importances=None
    
    def __init__(self,model,cv=10,importance=False,imp_repeat=10,gauss=False,gauss_para=(4,0.,1.,0.01),seed=None):
        self.model,self.cv,self.importance,self.gauss,self.gauss_para,self.seed=model,cv,importance,gauss,gauss_para,seed
        self.imp_repeat=imp_repeat
    
    def group_run(self,X_raw,y_raw,group=None,X_name=False,print_=True):
        n_feat=len(y_raw)
        cvMSE_va,cvMSE_va2,cvMSE_tr,n,cvr2=np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        cvMAE_va,cvMAPE_va=np.array([]),np.array([])
        gauss_para=self.gauss_para
        model=self.model
        for i,(arg_tr,arg_va) in enumerate(GroupKFold(n_splits=self.cv,shuffle=True,random_state=self.seed).split(X_raw,groups=group)):
            X_train,y_train,X_valid,y_valid=X_raw[arg_tr],y_raw[arg_tr],X_raw[arg_va],y_raw[arg_va]
            if self.gauss==False:
                pass
            elif self.gauss==True:
                X_train,y_train=Trans(X_train,y_train).gauss_noise(gauss_para[0],mu=gauss_para[1],sigma=gauss_para[2],\
                                                           alpha=gauss_para[3],seed=self.seed)
            else:
                raise ValueError('the Parameter \'gauss\' must be a boolean value.')
            model.fit(X_train,y_train)
            y_pred_va=model.predict(X_valid)
            y_pred_tr=model.predict(X_train)
            r2_va=r2_score(y_valid,y_pred_va)
            MSE_va=mean_squared_error(y_valid,y_pred_va)
            MAE_va=mean_absolute_error(y_valid,y_pred_va)
            MAPE_va=mean_absolute_percentage_error(y_valid,y_pred_va)
            
            r2_tr,MSE_tr=r2_score(y_train,y_pred_tr),mean_squared_error(y_train,y_pred_tr)
            MAE_tr=mean_absolute_error(y_train,y_pred_tr)
            MAPE_tr=mean_absolute_percentage_error(y_train,y_pred_tr)
            cvr2=np.append(cvr2,r2_va)
            
                
            if print_==True:
                print('第%i折交叉验证：\n'%(i+1),'验证集R^2= ',r2_va,'\n验证集MSE= ',MSE_va,'验证集MAE=',MAE_va,'验证集MAPE=',MAPE_va\
                      ,'\n训练集R^2：',r2_tr,'训练集MSE=',MSE_tr,'训练集MAE=',MAE_tr,'训练集MAPE=',MAPE_tr)
            n_va=len(arg_va)
            cvMSE_va=np.append(cvMSE_va,n_va*MSE_va)
            cvMAE_va=np.append(cvMAE_va,n_va*MAE_va)
            cvMAPE_va=np.append(cvMAPE_va,n_va*MAPE_va)
            cvMSE_va2=np.append(cvMSE_va2,MSE_va)
            n=np.append(n,n_va)
        self.mean_MSE=np.sum(cvMSE_va)/np.sum(n)
        self.mean_MAE=np.sum(cvMAE_va)/np.sum(n)
        self.mean_MAPE=np.sum(cvMAPE_va)/np.sum(n)
        self.max_MSE=max(cvMSE_va2)
        self.cv_R2=np.mean(cvr2)
        self.min_R2=min(cvr2)
        if self.importance==True:
            model.fit(X_raw,y_raw)
            valua=permutation_importance(model, X_raw, y_raw, n_repeats=10,random_state=self.seed)
            self.importances=valua.importances_mean
        return 
    
    def run(self,X_raw,y_raw,X_name=False,print_=True):
        n_feat=len(y_raw)
        cvMSE_va,cvMSE_va2,cvMSE_tr,n,cvr2=np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        cvMAE_va,cvMAPE_va=np.array([]),np.array([])
        gauss_para=self.gauss_para
        model=self.model
        for i,(arg_tr,arg_va) in enumerate(KFold(n_splits=self.cv,shuffle=True,random_state=self.seed).split(X_raw)):
            X_train,y_train,X_valid,y_valid=X_raw[arg_tr],y_raw[arg_tr],X_raw[arg_va],y_raw[arg_va]
            if self.gauss==False:
                pass
            elif self.gauss==True:
                X_train,y_train=Trans(X_train,y_train).gauss_noise(gauss_para[0],mu=gauss_para[1],sigma=gauss_para[2],\
                                                           alpha=gauss_para[3],seed=self.seed)
            else:
                raise ValueError('the Parameter \'gauss\' must be a boolean value.')
            model.fit(X_train,y_train)
            y_pred_va=model.predict(X_valid)
            y_pred_tr=model.predict(X_train)
            r2_va=r2_score(y_valid,y_pred_va)
            MSE_va=mean_squared_error(y_valid,y_pred_va)
            MAE_va=mean_absolute_error(y_valid,y_pred_va)
            MAPE_va=mean_absolute_percentage_error(y_valid,y_pred_va)
            
            r2_tr,MSE_tr=r2_score(y_train,y_pred_tr),mean_squared_error(y_train,y_pred_tr)
            MAE_tr=mean_absolute_error(y_train,y_pred_tr)
            MAPE_tr=mean_absolute_percentage_error(y_train,y_pred_tr)
            cvr2=np.append(cvr2,r2_va)
            
                
            if print_==True:
                print('第%i折交叉验证：\n'%(i+1),'验证集R^2= ',r2_va,'\n验证集MSE= ',MSE_va,'验证集MAE=',MAE_va,'验证集MAPE=',MAPE_va\
                      ,'\n训练集R^2：',r2_tr,'训练集MSE=',MSE_tr,'训练集MAE=',MAE_tr,'训练集MAPE=',MAPE_tr)
            n_va=len(arg_va)
            cvMSE_va=np.append(cvMSE_va,n_va*MSE_va)
            cvMAE_va=np.append(cvMAE_va,n_va*MAE_va)
            cvMAPE_va=np.append(cvMAPE_va,n_va*MAPE_va)
            cvMSE_va2=np.append(cvMSE_va2,MSE_va)
            n=np.append(n,n_va)
        self.mean_MSE=np.sum(cvMSE_va)/np.sum(n)
        self.mean_MAE=np.sum(cvMAE_va)/np.sum(n)
        self.mean_MAPE=np.sum(cvMAPE_va)/np.sum(n)
        self.max_MSE=max(cvMSE_va2)
        self.cv_R2=np.mean(cvr2)
        self.min_R2=min(cvr2)
        if self.importance==True:
            model.fit(X_raw,y_raw)
            valua=permutation_importance(model, X_raw, y_raw, n_repeats=self.imp_repeat,random_state=self.seed)
            self.importances=valua.importances_mean
        return 
        
    def sym_run(self,X_raw,y_raw,n_group,X_name=False,print_=True):
        '''
        系统抽样方法，n_group为划分的组数，其将整个X_args平均划分为n_group组
        '''
        if type(n_group) != int:
            raise TypeError('n_group must be an integer!!!') 
        elif len(X_raw)%n_group != 0:
            raise ValueError('Number of X cannot be divisible by n_group')
        
        n_samples_in = len(X_raw)//n_group
        n_feat = len(X_raw[0,:])
        group = np.reshape(X_raw, (n_group, n_samples_in, n_feat))
        y_group = np.reshape(y_raw, (n_group, n_samples_in))
        
        cvMSE_va,cvMSE_va2,cvMSE_tr,n,cvr2=np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        cvMAE_va,cvMAPE_va=np.array([]),np.array([])
        gauss_para=self.gauss_para
        model=self.model
        for i,(arg_tr,arg_va) in enumerate(KFold(n_splits=self.cv,shuffle=False,random_state=self.seed).split(group[0])):
            X_train = np.empty((0,n_feat))
            X_valid = np.empty((0,n_feat))
            y_train = np.empty((0))
            y_valid = np.empty((0))
            for j in range(n_group):
                X_train = np.r_[X_train, group[j, arg_tr,:]]
                X_valid = np.r_[X_valid, group[j, arg_va,:]]
                y_train = np.r_[y_train, y_group[j, arg_tr]]
                y_valid = np.r_[y_valid, y_group[j, arg_va]]
                
            if self.gauss==False:
                pass
            elif self.gauss==True:
                X_train,y_train=Trans(X_train,y_train).gauss_noise(gauss_para[0],mu=gauss_para[1],sigma=gauss_para[2],\
                                                           alpha=gauss_para[3],seed=self.seed)
            else:
                raise ValueError('the Parameter \'gauss\' must be a boolean value.')
            model.fit(X_train,y_train)
            y_pred_va=model.predict(X_valid)
            y_pred_tr=model.predict(X_train)
            r2_va=r2_score(y_valid,y_pred_va)
            MSE_va=mean_squared_error(y_valid,y_pred_va)
            MAE_va=mean_absolute_error(y_valid,y_pred_va)
            MAPE_va=mean_absolute_percentage_error(y_valid,y_pred_va)
            
            r2_tr,MSE_tr=r2_score(y_train,y_pred_tr),mean_squared_error(y_train,y_pred_tr)
            MAE_tr=mean_absolute_error(y_train,y_pred_tr)
            MAPE_tr=mean_absolute_percentage_error(y_train,y_pred_tr)
            cvr2=np.append(cvr2,r2_va)
            
                
            if print_==True:
                print('第%i折交叉验证：\n'%(i+1),'验证集R^2= ',r2_va,'\n验证集MSE= ',MSE_va,'验证集MAE=',MAE_va,'验证集MAPE=',MAPE_va\
                      ,'\n训练集R^2：',r2_tr,'训练集MSE=',MSE_tr,'训练集MAE=',MAE_tr,'训练集MAPE=',MAPE_tr)
            n_va=len(arg_va)
            cvMSE_va=np.append(cvMSE_va,n_va*MSE_va)
            cvMAE_va=np.append(cvMAE_va,n_va*MAE_va)
            cvMAPE_va=np.append(cvMAPE_va,n_va*MAPE_va)
            cvMSE_va2=np.append(cvMSE_va2,MSE_va)
            n=np.append(n,n_va)
        self.mean_MSE=np.sum(cvMSE_va)/np.sum(n)
        self.mean_MAE=np.sum(cvMAE_va)/np.sum(n)
        self.mean_MAPE=np.sum(cvMAPE_va)/np.sum(n)
        self.max_MSE=max(cvMSE_va2)
        self.cv_R2=np.mean(cvr2)
        self.min_R2=min(cvr2)
        if self.importance==True:
            model.fit(X_raw,y_raw)
            valua=permutation_importance(model, X_raw, y_raw, n_repeats=self.imp_repeat,random_state=self.seed)
            self.importances=valua.importances_mean
        return 
        
'''
Initial feature set generation
--------------------------------------------------------------------------------------------------------------------------------------
'''

class FeaturesJoint(object):
    '''
    读入文件格式: 内容为每个元素的各种性质，行为元素种类，列为元素性质的数值。
                  第一行、列分别为元素和性质的名称。
    
    para:
        path: str, path of input file which is atom properties.
        header: like 'header' in pandas.read_csv
    
    Atrri:
        
        
    '''
    def __init__(self,path,header=0):
        self.dataDict={}
        self.joinDataDict={}
        self.joinData=None
        df=pd.read_csv(path,header=header)
        rawdata=np.asarray(df.values[:,1:],dtype=np.float64)
        rawname=df.values[:,0]
        self.data=rawdata[~np.isnan(rawdata).any(axis=1)]
        self.name=rawname[~np.isnan(rawdata).any(axis=1)]
        self.n_samp=len(self.data)
        self.n_feature=len(self.data[0,:])
        for i,n in enumerate(self.data):
            self.dataDict[self.name[i]]=n
        
    def join(self,menbers=2,sequential=False):
        '''
        members为整数时，将遍历整个data，组成一切可能的member元化合物的特征集;
        为n个元素名的元组组成的的字符串列表时，组成字符串所示的n元化合物的特征集
        
        para:
            menbers: int, 要组合的元素个数
            sequential: bool, True则表示二者交换位置所得结果等价
        
        '''
        if sequential==True:
            if type(menbers)==int:
                temp1=self.data
                temp2=np.empty((0,2*len(temp1[0,:])),np.float64)
                temp1_name=self.name
                temp2_name=[]
                for i in range(menbers-1):
                    for j,val1 in enumerate(temp1):
                        for key,val2 in (self.dataDict).items():
                            temp2=np.append(temp2,[np.r_[val1,val2]],axis=0)
                            temp2_name=np.append(temp2_name,temp1_name[j]+key)
                    temp1=copy.deepcopy(temp2)
                    temp1_name=copy.deepcopy(temp2_name)
                    temp2=np.empty((0,self.n_feature+len(temp1[0,:])),np.float64)
                    temp2_name=[]
                self.joinData=np.c_[np.array(temp1_name).T,temp1]
                for i,n in enumerate(temp1_name):
                    self.joinDataDict[n]=temp1[i]
                pass
            elif type(menbers)==list and type(menbers[0])==tuple:
                print('将在后续版本放送')
                pass
            else:
                raise TypeError('members must be an int or a list composed of str tuples')
        else:
            if type(menbers)==int:
                temp1=self.data
                temp4=self.data
                temp2=np.empty((0,2*len(temp1[0,:])),np.float64)
                temp1_name=self.name
                temp4_name=self.name
                temp2_name=[]
                for i in range(menbers-1):
                    for j,val1 in enumerate(temp1):
                        temp3=temp4[j+i:]
                        temp3_name=temp4_name[j+i:]
                        for k,val2 in enumerate(temp3):
                            temp2=np.append(temp2,[np.r_[val1,val2]],axis=0)
                            temp2_name.append(temp1_name[j]+temp3_name[k])
                    temp1=copy.deepcopy(temp2)
                    temp1_name=copy.deepcopy(temp2_name)
                    temp2=np.empty((0,self.n_feature+len(temp1[0,:])),np.float64)
                    temp2_name=[]
                self.joinData=np.c_[np.array(temp1_name).T,temp1]
                for i,n in enumerate(temp1_name):
                    self.joinDataDict[n]=temp1[i]
                pass
            elif type(menbers)==list and type(menbers[0])==tuple:
                print('将在后续版本放送')
                pass
            else:
                raise TypeError('members must be an int or a list composed of str tuples')

'''
贪婪算法随机搜索初等函数变换特征
-------------------------------------------------------------------------------------------------------------------------------------
'''                

class GRSF(object):
    def __init__(self,niter=1,size=1,p_append=0.5,max_layer=None,seed=None,ncyc=5):
        '''
        Iteratively use unary and binary mapping to transform the input feartures to greedily obtain the best transformed features 
        evaluated by PCC or cross-validation of linear regressor.
        
        The unary mapping is one of square, cube, log, exp, square root and identity;
        The binary mapping is one of addition, substraction, multiplication, division and identity(to the first varible).
        
        Parameters:
            niter: int, the number of iterations;
            size: int, the number of selected features;
            p_append: float, between 0 and 1. The probability of retaining a new transform. To control the complexity of transform.
            max_layer: int, the maximum number of transform layers. A layer contains a unary mapping and a binary mapping. 
            seed: int, the random seed.
            ncyc: int, the repeatition number of fitting.
            
        
        This program picks the top size features with the largest score 
        '''
        self.size=size
        self.niter=niter
        self.seed=seed
        self.p_append=p_append
        self.max_layer=max_layer
        self.ncyc=ncyc
    
    def _plus(self,a,b,name=False):
        if name==True:
            N='((%s)+(%s))/2'%(a,b)
            return N
        else:
            return (a+b)/2

    def _diff(self,a,b,name=False):
        if name==True:
            N='abs((%s)-(%s))'%(a,b)
            return N
        else:
            return abs(a-b)

    def _mult(self,a,b,name=False):
        if name==True:
            N='(%s)*(%s)'%(a,b)
            return N
        else:
            return a*b

    def _divi(self,a,b,name=False):
        if name==True:
            N='(%s)/(%s+1)'%(a,b)
            return N
        else:
            return a/(b+1)

    def _power2(self,a,name=False):
        if name==True:
            N='(%s)**2'%a
            return N
        else:
            return a**2

    def _power3(self,a,name=False):
        if name==True:
            N='(%s)**3'%a
            return N
        else:
            return a**3

    def _sqrt(self,a,name=False):
        if name==True:
            N='abs(%s)**0.5'%a
            return N
        else:
            return abs(a)**0.5

    def _abs_log(self,a,name=False):
        if name==True:
            N='np.log1p(abs(%s))'%a
            return N
        else:
            return np.log1p(abs(a))
            
    def _iden(self,a,name=False):
        if name==True:
            N='(%s)'%a
            return N
        else:
            return a
        
    def _iden2(self,a,b,name=False):
        if name==True:
            N='(%s)'%a
            return N
        else:
            return a
    
    def _exp(self,a,name=False):
        if name==True:
            N='np.exp(%s-1)'%a
            return N
        else:
            return np.exp(a-1.)
        
    def _distcorr(self,Xi,Y):
        Xi=np.asarray(Xi)
        Y=np.asarray(Y)
        if np.prod(Xi.shape)==len(Xi):
            Xi=Xi[:,None]
        if np.prod(Y.shape)==len(Y):
            Y=Y[:,None]
        Xi=np.atleast_2d(Xi)
        Y=np.atleast_2d(Y)
        n=Xi.shape[0]
        if Y.shape[0]!=Xi.shape[0]:
            raise ValueError('Number of samples must match')
        a=squareform(pdist(Xi))
        b=squareform(pdist(Y))
        A=a-a.mean(axis=0)[None,:]-a.mean(axis=1)[:,None]+a.mean()
        B=b-b.mean(axis=0)[None,:]-b.mean(axis=1)[:,None]+b.mean()
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        if dcov2_xx*dcov2_yy==0:
            dcor=0.
        else:
            dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor
    
    def _PCCs(self,X,y):
        X,y=np.asarray(X),np.asarray(y)
        if len(X.shape)==1:
            X=(np.atleast_2d(X)).T
        elif len(X.shape)!=2:
            raise ValueError('X must be a vector or matrix')
        PCCs=np.array([])
        ymean=np.mean(y)
        for i in range(len(X[0,:])):
            Xmean=np.mean(X[:,i])
            yk,cov,varx,vary=0,0.,0.,0.
            for j in X[:,i]:
                a,b=(j-Xmean),(y[yk]-ymean)
                cov=cov+a*b
                varx=varx+a**2
                vary=vary+b**2
                yk=yk+1
            c=varx*vary
            if c==0.:
                PCCs=np.append(PCCs,0.)
            else:
                PCCs=np.append(PCCs,cov/(c**0.5))
        if len(PCCs)==1:
            PCCs=PCCs[0]
        return PCCs
    
    def run(self,X,y,X_name=None):
        '''
        Evaluate the fitting performance by PCC.
        '''
        s_=time.time()
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        size=self.size
        n_samp,n_feat=np.shape(X)
        N_no=[self._exp,self._abs_log,self._power2,self._power3,self._sqrt,self._iden]
        N_bi=[self._plus,self._diff,self._mult,self._divi,self._iden2]
        mem=set()
        corr_list=np.zeros(size,dtype=np.float64)
        feat_list=np.zeros((n_samp,size))
        layer=0
        layer_n=-1
        X_temp=X
        
        if self.p_append>1. or self.p_append<0.:
            raise ValueError('p_append must between 0 and 1')
        if type(X_name)==np.ndarray:
            X_name_temp=X_name.tolist()
        elif type(X_name)==list:
            X_name_temp=X_name
            
        if X_name is not None:
            if len(X_name)!=n_feat:
                raise ValueError('Number of features and features\' name does not match!!!  N_features: %i  N_features_name: %i'%(n_feat,len(X_name)))
        else:
            print('Not read the X_name, generating the names of features: X1, X2, X3, ...')
            X_name=['']*n_feat
            for i_ in range(n_feat):
                X_name[i_]='X%i'%(i_+1)
            
        print('Enter the main loop ...')
        name_list=['']*size
        repeat_n,bi_repeat_n,abel_repeat_n,ident_repeat_n = 0,0,0,0
        for _ in range(self.niter):
            i1,i2,j1,j2,k = np.random.randint(n_feat),np.random.randint(n_feat),\
            np.random.randint(6),np.random.randint(6),np.random.randint(5)
            rand_tup=(i1,i2,j1,j2,k)
            iden_tup=(i1,j1)
            
            if ((iden_tup in mem) and k==4):
                ident_repeat_n+=1
                continue
            elif (j1==5 and k==4):
                ident_repeat_n+=1
                continue
            elif (rand_tup in mem):
                repeat_n+=1
                continue
            elif (i1==i2 and j1==j2):
                bi_repeat_n+=1
                continue
            elif (k<3 and ((i2,i1,j2,j1,k) in mem)):
                abel_repeat_n+=1
                continue
            
            mem.add(rand_tup)
            mem.add(iden_tup)
            f=N_bi[k](N_no[j1](X_temp[:,i1]),N_no[j2](X_temp[:,i2]))
            f_name=N_bi[k](N_no[j1](X_name_temp[i1],name=True),N_no[j2](X_name_temp[i2],name=True),name=True)

            distcorr = self._PCCs(f,y)
            for kk in range(size):
                if distcorr>corr_list[-(kk+1)]:
                    for ii in range(size-kk-1):
                        corr_list[ii]=copy.deepcopy(corr_list[ii+1])
                        feat_list[:,ii]=copy.deepcopy(feat_list[:,ii+1])
                        name_list[ii]=copy.deepcopy(name_list[ii+1])
                    corr_list[-kk-1]=copy.deepcopy(distcorr)
                    feat_list[:,-kk-1]=copy.deepcopy(f)
                    name_list[-kk-1]=copy.deepcopy(f_name)
                    break


            if self.max_layer is None:
                if np.random.random_sample()<self.p_append:
                    X_temp=np.c_[X_temp,f]
                    X_name_temp.append(f_name)
                    n_feat=len(X_temp[0,:])
                pass
            elif layer>=self.max_layer:
                if i1>layer_n or i2>layer_n:
                    pass
                else:
                    if np.random.random_sample()<self.p_append:
                        X_temp=np.c_[X_temp,f]
                        X_name_temp.append(f_name)
                        n_feat=len(X_temp[0,:])
            else:
                if np.random.random_sample()<self.p_append:
                    X_temp=np.c_[X_temp,f]
                    X_name_temp.append(f_name)
                    n_feat=len(X_temp[0,:])
                    if i1>layer_n or i2>layer_n:
                        layer+=1
                        layer_n=n_feat-2

        self.X_trans_name=name_list
        self.val_list=corr_list
        self.X_trans=feat_list
        e_=time.time()
        print('*'*137,'\nEnd of main loop\n','*'*137,
              '\nMax transformation layer: %i'%(layer+1),
              '\nTransformation repetition times(the simple repetitions, the same args in bi-mapping, the exchanged args in commutative bi-mapping and the identical mapping): %i,%i,%i,%i'%(repeat_n,bi_repeat_n,abel_repeat_n,ident_repeat_n),
              '\nTotal Time: %f s'%(e_-s_))
        
    def CV_run(self,X,y,X_name=None,model=linear_model.LinearRegression(n_jobs=4,),cv=5):
        '''
        Evaluate the fitting performance by cross-validation on a linear regressor model.
        '''
        s_=time.time()
        
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        max_y=max(y)
        size=self.size
        n_samp,n_feat=np.shape(X)
        N_no=[self._exp,self._abs_log,self._power2,self._power3,self._sqrt,self._iden]
        N_bi=[self._plus,self._diff,self._mult,self._divi,self._iden2]
        mem=set()
        corr_list=np.full(size,max_y,dtype=np.float64)
        feat_list=np.ones((n_samp,size))
        layer=0
        layer_n=0
        X_temp=X
        y_temp=y
        X_fit_temp=np.empty((n_samp,0))
        X_trans_name=[]
        val_list=np.empty(self.ncyc,dtype=np.float64)
        old_MSE=114514.
        
        if self.p_append>1. or self.p_append<0.:
            raise ValueError('p_append must between 0 and 1')
        if type(X_name)==np.ndarray:
            X_name_temp=X_name.tolist()
        elif type(X_name)==list:
            X_name_temp=X_name
            
        if X_name is not None:
            if len(X_name)!=n_feat:
                raise ValueError('Number of features and features\'name does not match!!!')
        else:
            print('Not read the X_name, generating the names of features: X1, X2, X3, ...')
            X_name=['']*n_feat
            for i_ in range(n_feat):
                X_name[i_]='X%i'%(i_+1)
            
        print('Enter the main loop ...')
        name_list=['']*size
        repeat_n,bi_repeat_n,abel_repeat_n=0,0,0
        for __ in range(self.ncyc):
            for _ in range(round(self.niter*(1+__**0.33))):
                i1,i2,j1,j2,k = np.random.randint(n_feat),np.random.randint(n_feat),\
                np.random.randint(6),np.random.randint(6),np.random.randint(5)
                rand_tup=(i1,i2,j1,j2,k)
                if (rand_tup in mem):
                    repeat_n+=1
                    continue
                elif (i1==i2 and j1==j2):
                    bi_repeat_n+=1
                    continue
                elif (k<3 and ((i2,i1,j2,j1,k) in mem)):
                    abel_repeat_n+=1
                    continue
                mem.add(rand_tup)
                f=N_bi[k](N_no[j1](X_temp[:,i1]),N_no[j2](X_temp[:,i2]))
                f_name=N_bi[k](N_no[j1](X_name_temp[i1],name=True),N_no[j2](X_name_temp[i2],name=True),name=True)

                if abs(self._PCCs(f,y_temp))>0.:
                    CVer=CV(model,cv=cv)
                    CVer.run(f.reshape(-1,1),y_temp,print_=False)
                    for kk in range(size):
                        if CVer.mean_MSE<corr_list[-(kk+1)]:
                            for ii in range(size-kk-1):
                                corr_list[ii]=copy.deepcopy(corr_list[ii+1])
                                feat_list[:,ii]=copy.deepcopy(feat_list[:,ii+1])
                                name_list[ii]=copy.deepcopy(name_list[ii+1])
                            corr_list[-kk-1]=copy.deepcopy(CVer.mean_MSE)
                            feat_list[:,-kk-1]=copy.deepcopy(f)
                            name_list[-kk-1]=copy.deepcopy(f_name)
                            break


                if self.max_layer is None:
                    if np.random.random_sample()<self.p_append:
                        X_temp=np.c_[X_temp,f]
                        X_name_temp.append(f_name)
                        n_feat=len(X_temp[0,:])
                    pass
                elif layer>=self.max_layer:
                    if i1>layer_n or i2>layer_n:
                        pass
                    else:
                        if np.random.random_sample()<self.p_append:
                            X_temp=np.c_[X_temp,f]
                            X_name_temp.append(f_name)
                            n_feat=len(X_temp[0,:])
                else:
                    if np.random.random_sample()<self.p_append:
                        X_temp=np.c_[X_temp,f]
                        X_name_temp.append(f_name)
                        n_feat=len(X_temp[0,:])
                        if i1>layer_n or i2>layer_n:
                            layer+=1
                            layer_n=n_feat-2
            
            if name_list[-1]!='':
                X_fit_temp=np.c_[X_fit_temp,copy.deepcopy(feat_list[:,-1])]

                CVer=CV(model,cv=cv)
                CVer.run(X_fit_temp,y,print_=True)

                if CVer.mean_MSE<old_MSE:
                    model.fit(X_fit_temp,y)
                    y_pred1=model.predict(X_fit_temp)
                    R2=r2_score(y_pred1,y)
                    y_temp=y-y_pred1
                    print('*'*135,'\nIteration: %i'%(__+1),'\nCV_MSE = %f\nMean Square Residual = %s'%(CVer.mean_MSE,(y_temp@y_temp)/n_samp),\
                         '\nTrain R2 = %f'%R2,'\n'+('*'*135))
                    old_MSE=copy.deepcopy(CVer.mean_MSE)
                    X_trans_name.append(name_list[-1])
                else:
                    X_fit_temp=np.delete(X_fit_temp,-1,axis=1)
                    print('*'*135,'\nIteration: %i'%(__+1),'\nCV_MSE = %f'%CVer.mean_MSE\
                          ,'\nThe CV-MSE increase, so the new feature is removed!!!','\n'+('*'*135))

                val_list[__]=old_MSE
            else:
                print(name_list,'\n',feat_list)
                print('*'*135,'\nIteration: %i'%(__+1)\
                          ,'\nThe greedy algorithm does not give any new feature!!!','\n'+('*'*135))
            
            corr_list=np.full(size,max_y,dtype=np.float64)
            feat_list=np.ones((n_samp,size))
            layer=0
            layer_n=0
            X_temp=X
            name_list=['']*size
            n_feat=len(X[0,:])
            
        self.X_trans_name=X_trans_name
        self.val_list=val_list
        self.X_trans=X_fit_temp
        e_=time.time()
        print('*'*137,'\nEnd of main loop\n','*'*137,
              '\nTransformation repetition times(including the simple repetitions, the same args in bi-mapping and the exchanged args in commutative bi-mapping): %i,%i,%i'%(repeat_n,bi_repeat_n,abel_repeat_n),
              '\nTotal Time: %f s'%(e_-s_))
        
    def decode(self,X,X_origin_name,X_trans_name=None):
        
        n_samp,n_feat=np.shape(X)
        if len(X_origin_name)!=n_feat:
            raise ValueError('Number of features and features\'name does not match!!!')
        for i,name_val in enumerate(X_origin_name):
            exec(name_val+'=X[:,i]')
        
        if X_trans_name is None:
            trans_name=self.X_trans_name
        else:
            trans_name=X_trans_name
        
        n_trans_feat=len(trans_name)
        X_trans=np.empty((n_samp,n_trans_feat),dtype=np.float64)
        for i,expr in enumerate(trans_name):
            X_trans[:,i]=eval(expr)
            
        return X_trans
            
            
        
        pass
        
'''
utils
--------------------------------------------------------------------------------------------------------------------------------------
'''
def name2value(X,X_origin_name,X_trans_name, name_can_be_var=False):
    '''
    Transform the original features X_ori to X_trans by 
    the transformational rule: name of original X to name of transformed X.
    
    Parameters:
        X: 2d-array of float, the values matrix of original features.
        X_origin_name: 1d-array of str, the original features' name.
        X_trans_name: 1d-array of str, the transformed features' name.
        name_can_be_var: boolean, for true, directly use the original X names as varible names which can run faster; or, use intermediate varibles to replace the origin names.
        
    Return: 
        X_trans: 2d-array of float, the values matrix of transformed features.
     
    '''
    n_samp,n_feat=np.shape(X)
    
    if len(X_origin_name)!=n_feat:
        raise ValueError('Number of features and features\'name does not match!!!')
    
    if name_can_be_var:
        for i,name_val in enumerate(X_origin_name):
            exec(name_val+'=X[:,i]')  
        
        trans_name=X_trans_name
        n_trans_feat=len(trans_name)
        
        X_trans=np.empty((n_samp,n_trans_feat),dtype=np.float64)
        for i,expr in enumerate(trans_name):
            X_trans[:,i]=eval(expr)
    else:
        trans_name = X_trans_name
        n_trans_feat = len(trans_name)
        new_trans_name = np.array(X_trans_name, dtype=str).tolist()
        
        name_len = np.empty_like(X_origin_name, dtype=int)
        for i,val in enumerate(X_origin_name):
            name_len[i] = -len(val)
        args_ = np.argsort(name_len)
        sorted_ori_name = X_origin_name[args_]
        X_sorted = X[:,args_]
        
        for i,name_val in enumerate(sorted_ori_name):
            X_temp = 'X%i'%i
            exec(X_temp+'=X_sorted[:,i]')
            for j,ts_na in enumerate(new_trans_name): 
                new_trans_name[j] = ts_na.replace(name_val, X_temp)  
        
        X_trans=np.empty((n_samp,n_trans_feat),dtype=np.float64)
        for i,expr in enumerate(new_trans_name):
            X_trans[:,i]=eval(expr)
        
    return X_trans
    


def density_of_state(X,interval=None,samp_points=301):
    '''
    Calculating the density of state of X within interval i.e., the number of values in X within the range [l, l+dl) where dl equals to interval/samp_points.
    
    Parameters:
        X: ndarray, the input data
        interval: tuple, the interval counted points.
        samp_points: int, the points number of output.
    
    Return:
        y: 2darray, the density of state list, with shape of (samp_points, 2)
    
    '''
    if interval is None:
        interval = (np.min(X),np.max(X))
    elif interval[0] >= interval[1]:
        raise ValueError('The beginning of interval is greater than the end')
    if not interval[0] <= interval[1]:
        raise TypeError('Do Not support the input interval %s with type %s'%(interval, type(interval)))
    
    dn = (interval[1] - interval[0])/samp_points
    split_ = np.arange(interval[0],interval[1]-0.5*dn,dn)
    y = np.zeros((samp_points,2),dtype=float)
    y[:,0]=split_
    
    for val in X.flatten():
        for i,cond in enumerate(split_):
            if val >= cond and val < cond+dn:
                y[i,1] = y[i,1] + 1
                break
    return y


def empty():
    pass
    
'''
Feature importance
----------------------------------------------------------------------------------------------------------------------------------------
'''
class Trans_importance(object):
    '''
    扩展的排列重要性：当初始特征集X_ori经过某种变换 X_trans=T(X_ori) 并训练模型后，
    若仍想获得原特征X_ori对模型的重要性，即可用此法。
    
    考虑由变换特征X_trans训练的模型f(X_trans)=f(T(X_ori))是由映射f和T组成的复合函数。
    本方法通过改变初始特征集X_ori，评估X_ori对模型f·T(X_ori)的重要性。
    '''
    importance_R2=None
    importance_MSE=None
    R2_std=None
    MSE_std=None
    
    def __init__(self,model,repeats=10,seeds=None):
        '''
        For a set of transformed features X' and the corresponding operator T where X' = T(X),
        the model f: y = f(X') = f(T(X)).
        this class uses to obtain the permutation importance of X to f.
        '''
        self.model=model
        self.seeds=seeds
        self.repeats=repeats
    
    def _eval_model(self,X,y,relearn=False):
        mod=self.model
        if relearn==True:
            mod.fit(X,y)
        y_pred=mod.predict(X)
        R2=r2_score(y,y_pred)
        MSE=mean_squared_error(y,y_pred)
        return R2,MSE
    
    def permut_imp(self,X_ori,y_ori,X_ori_name,X_trans_name,relearn=False):
        '''排列重要性'''
        X_trans_std=name2value(X_ori,X_ori_name,X_trans_name)
        R2_std,MSE_std=self._eval_model(X_trans_std,y_ori)
        self.R2_std=R2_std
        self.MSE_std=MSE_std
        
        n_samp,n_feat=np.shape(X_ori)
        ori_args=np.array(range(n_samp))
        R2_shuf,MSE_shuf=np.empty((self.repeats,n_feat),dtype=np.float64),np.empty((self.repeats,n_feat),dtype=np.float64)
        
        for f in range(n_feat):
            X_ori_shuf=copy.deepcopy(X_ori)
            for i in range(self.repeats):
                shuffle_args=np.random.permutation(ori_args)
                X_ori_shuf[:,f]=X_ori[shuffle_args,f]
                X_trans_shuf=name2value(X_ori_shuf,X_ori_name,X_trans_name)
                R2_shuf[i,f],MSE_shuf[i,f]=self._eval_model(X_trans_shuf,y_ori,relearn=relearn)
            
        self.importance_R2=R2_std-np.mean(R2_shuf,axis=0)
        self.importance_MSE=np.mean(MSE_shuf,axis=0)-MSE_std

class Trans_response(object):
    def __init__(self, model, X_ori_name, X_trans_name, name_can_be_var=False):
        '''
        For a set of transformed features X' and the corresponding operator T where X' = T(X),
        the model f: y = f(X') = f(T(X)) = (f*T)(X) = f'(X).
        this class uses to calculate the transformed model f' i.e., the composite function f*T.
        
        Parameter:
            model: function_like (e.g. models in sklearn), the f.
            name_can_be_var: boolean, if each original name is also a valid varible name, set True can run faster; or set False to avoid ERROR.
        
        Attribute:
            f_trans: function, the f'.
        '''
        self.model = model
        self.X_ori_name, self.X_trans_name = X_ori_name, X_trans_name
        self.n_var = name_can_be_var
        
    def __fit(self, X_ori, X_ori_name, X_trans_name):
        '''
        use X_trans_name and X_ori_name to fit f_trans
        '''
        X_trans_ = name2value(X_ori, X_ori_name, X_trans_name, name_can_be_var=self.n_var)
        
        return X_trans_
    
    def f_trans(self,X):
        X_t = self.__fit(X, self.X_ori_name, self.X_trans_name)
        return self.model.predict(X_t)
