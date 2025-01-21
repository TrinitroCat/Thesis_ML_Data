'''
A module of analysing the trained ML model.
'''
#scikit-learn
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

#time
import time

#numba
#import numba
#from numba import jit

# in situ
from . import main
from main import name2value

#numpy
import numpy as np  
#深拷贝
import copy


class Trans_importance(object):
    '''
    扩展的排列重要性：当初始特征集X_ori经过某种变换 X_trans=T(X_ori) 并训练模型后，
    若仍想获得原特征X_ori对模型的重要性，即可用此法。
    
    考虑由变换特征X_trans训练的模型f(X_trans)=f(T(X_ori))是由映射f和T组成的复合函数。
    本方法通过改变初始特征集X_ori，评估X_ori对模型f·T(X_ori)的重要性。
    '''
    #声明类的属性
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
        #评价模型函数
        mod=self.model
        if relearn==True:  #是否重新训练模型
            mod.fit(X,y)
        y_pred=mod.predict(X)
        R2=r2_score(y,y_pred)
        MSE=mean_squared_error(y,y_pred)
        return R2,MSE
    
    def permut_imp(self,X_ori,y_ori,X_ori_name,X_trans_name,relearn=False):
        '''排列重要性'''
        #先对X_ori解码，计算标准值
        X_trans_std=name2value(X_ori,X_ori_name,X_trans_name)
        R2_std,MSE_std=self._eval_model(X_trans_std,y_ori)
        self.R2_std=R2_std
        self.MSE_std=MSE_std
        
        #前置数据
        n_samp,n_feat=np.shape(X_ori)
        ori_args=np.array(range(n_samp))
        R2_shuf,MSE_shuf=np.empty((self.repeats,n_feat),dtype=np.float64),np.empty((self.repeats,n_feat),dtype=np.float64)
        
        #遍历每个特征并随机排序，计算评分
        for f in range(n_feat):
            X_ori_shuf=copy.deepcopy(X_ori)
            for i in range(self.repeats):
                shuffle_args=np.random.permutation(ori_args)
                X_ori_shuf[:,f]=X_ori[shuffle_args,f]
                #print('i:%i\n'%i,X_ori_shuf)
                X_trans_shuf=name2value(X_ori_shuf,X_ori_name,X_trans_name)
                R2_shuf[i,f],MSE_shuf[i,f]=self._eval_model(X_trans_shuf,y_ori,relearn=relearn)
            
        #计算评分与标准值的差，并赋予属性
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
        