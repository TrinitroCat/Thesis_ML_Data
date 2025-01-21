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
#���
import copy


class Trans_importance(object):
    '''
    ��չ��������Ҫ�ԣ�����ʼ������X_ori����ĳ�ֱ任 X_trans=T(X_ori) ��ѵ��ģ�ͺ�
    ��������ԭ����X_ori��ģ�͵���Ҫ�ԣ������ô˷���
    
    �����ɱ任����X_transѵ����ģ��f(X_trans)=f(T(X_ori))����ӳ��f��T��ɵĸ��Ϻ�����
    ������ͨ���ı��ʼ������X_ori������X_ori��ģ��f��T(X_ori)����Ҫ�ԡ�
    '''
    #�����������
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
        #����ģ�ͺ���
        mod=self.model
        if relearn==True:  #�Ƿ�����ѵ��ģ��
            mod.fit(X,y)
        y_pred=mod.predict(X)
        R2=r2_score(y,y_pred)
        MSE=mean_squared_error(y,y_pred)
        return R2,MSE
    
    def permut_imp(self,X_ori,y_ori,X_ori_name,X_trans_name,relearn=False):
        '''������Ҫ��'''
        #�ȶ�X_ori���룬�����׼ֵ
        X_trans_std=name2value(X_ori,X_ori_name,X_trans_name)
        R2_std,MSE_std=self._eval_model(X_trans_std,y_ori)
        self.R2_std=R2_std
        self.MSE_std=MSE_std
        
        #ǰ������
        n_samp,n_feat=np.shape(X_ori)
        ori_args=np.array(range(n_samp))
        R2_shuf,MSE_shuf=np.empty((self.repeats,n_feat),dtype=np.float64),np.empty((self.repeats,n_feat),dtype=np.float64)
        
        #����ÿ��������������򣬼�������
        for f in range(n_feat):
            X_ori_shuf=copy.deepcopy(X_ori)
            for i in range(self.repeats):
                shuffle_args=np.random.permutation(ori_args)
                X_ori_shuf[:,f]=X_ori[shuffle_args,f]
                #print('i:%i\n'%i,X_ori_shuf)
                X_trans_shuf=name2value(X_ori_shuf,X_ori_name,X_trans_name)
                R2_shuf[i,f],MSE_shuf[i,f]=self._eval_model(X_trans_shuf,y_ori,relearn=relearn)
            
        #�����������׼ֵ�Ĳ����������
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
        