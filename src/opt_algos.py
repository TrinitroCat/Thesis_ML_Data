'''
The module of optimization algorithms.
'''
import numpy as np
import scipy as sp
import time
import copy

class Int_Opt(object):
    '''
    To optimize functions contained argument(s) within integers to minimum.
    '''
    def __init__(self):
        self.X_opt = None
        self.y_min = None
    
    def SAA(self,func,bounds,args=None,x0=None,integer_args=-1,maxit=500,init_temp=5230.0,temp_descent_factor=1.62,fin_temp=0.5,verbose=1):
        '''
        Simulation Anneal Algorithm，where X was perturbed by Cauchy distribution，T descent as: T = T0*(2^temp_descent_factor - 1)/((1 + t)^temp_descent_factor - 1)
        
        Parameters:
            func: function to opt.
            bounds: tuple, range of independent varibles.
            args: other varibles not optimized.
            x0: tuple, initial point.
            integer_args: int, to specify first `integer_args` vars. were interger.
            maxit: int, number of iter. in each temperature.
            init_temp: float, initial temperature.
            temp_descent_factor: positive float, temp. descent factor.
            fin_temp: float, final temp.
            verbose: int, 0 to scilence mode, 1 to print key information, and >1 to print informations in each iteration steps.
        '''
        #Initialize variables
        X_dim=len(bounds)
        integer_args=int(integer_args)
        if x0 is None:
            if verbose > 0:
                print('x0 is not read, select a random point within bounds as x0')
            x0=np.zeros(X_dim)
            for i,boundi in enumerate(bounds):
                if i<integer_args:
                    x0[i]=np.random.randint(boundi[0],high=boundi[1])
                else:
                    x0[i]=(boundi[1]-boundi[0])*np.random.random()+boundi[0]   
        elif len(x0)!=X_dim:
            raise ValueError('Dimensions of x0 and bounds do not match !!!')
        xint0=np.array(x0[:integer_args],dtype=int)
        xfloat0=np.array(x0[integer_args:])
        if args is None:
            func_old=func(*xint0,*xfloat0)
        else:
            func_old=func(*xint0,*xfloat0,*args)
        func_best=func_old
        x_old=x0
        x_new=x0
        x_best=x0
        temp=init_temp
        t=0  #time step
        itera=np.array(range(maxit),dtype=int)  #iter number in each temp. to avoid repeatedly generate range(maxit)
        gamma=temp  #shape factor of Cauchy-dist
        
        #Check input
        if maxit<=0:
            raise ValueError('maxit must be a positive integer !!!')
        elif init_temp<=0 or fin_temp<=0:
            raise ValueError('temperature must be positive numbers !!!')
        elif init_temp<=fin_temp:
            raise ValueError('init_temp must be greater than fin_temp !!!')
        elif temp_descent_factor<1E-5:
            raise ValueError('temp_descent_factor must greater than 0. !!!')    
        maxit=int(maxit)
        
        #main loop
        if verbose > 0:
            print('Variables initialized, entering main loop...')
        S_time=time.time()
        while temp>=fin_temp:
            init_time=time.time()
            for i in itera:
                #generate multi-Cauchy dist point; ARS method.(deprecated)
                '''
                while True:
                    for j,boundi in enumerate(bounds):  #generate uniform dist x
                        if j<integer_args:
                            x_new[j]=np.random.randint(boundi[0],high=boundi[1])
                        else:
                            x_new[j]=(boundi[1]-boundi[0])*np.random.random()+boundi[0]
                    dist_p=(1/(gamma*np.pi))*np.random.random()  #generate a random value between 0 and the max value of cauchy dist
                    cauchy_x=1/(gamma*np.pi*(1+(np.linalg.norm(x_new-x_old)/gamma)**2))  #calculate cauchy PDF(x)
                    if dist_p<=cauchy_x:  #select the point under PDF(x)
                        break
                print(gamma)
                tr.append(copy.deepcopy(x_new))
                '''
                #Use scpipy ramdom generater
                x_new=sp.stats.cauchy.rvs(loc=x_old,scale=gamma,size=X_dim)
                for j,boundi in enumerate(bounds):  #correct the x to its domain
                    if j<integer_args:
                        x_new[j]=round(x_new[j])
                    if x_new[j]>boundi[1]:
                        x_new[j]=boundi[1]
                    elif x_new[j]<boundi[0]:
                        x_new[j]=boundi[0]
                #select x
                if args is None:
                    func_new=func(*(x_new[:integer_args].astype(int)),*x_new[integer_args:])
                else:
                    func_new=func(*(x_new[:integer_args].astype(int)),*x_new[integer_args:],*args)
                if func_new<func_old:
                    func_old=copy.deepcopy(func_new)
                    x_old=copy.deepcopy(x_new)
                    if func_new<func_best:
                        func_best=copy.deepcopy(func_new)  #record the local minimum point and func. value
                        x_best=copy.deepcopy(x_new)
                elif np.random.random()<np.exp(-(func_new-func_old)/temp):
                    func_old=copy.deepcopy(func_new)
                    x_old=copy.deepcopy(x_new)
            #drop down temp
            t+=1
            temp=init_temp*(2**temp_descent_factor - 1)/((1 + t)**temp_descent_factor - 1)
            gamma=temp**min(temp_descent_factor,2)
            end_time=time.time()
            if verbose > 1:
                print('\n'+'*'*50+'Iteration  %i'%t+'*'*50,'\nTemperature: %.2f  ,  min function value: %.4f  ,  optimize x: %s  \nTime: %f'%(temp,func_best,x_best,end_time-init_time))
        
        E_time=time.time()
        if verbose > 0:
            print('*'*130)
            print('*Iteration Terminated.\n*Min Value of function: %.4f \n*Total Time: %f'%(func_best,E_time-S_time))
            print('*'*130)
        self.X_opt = x_best
        self.y_min = func_best
            
    def CCD(self,):
        '''
        Cyclic Coordination Descent for integer optimization. TODO.
        '''
        raise NotImplementedError
        pass