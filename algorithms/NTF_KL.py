import torch
from torch import nn
from torch.nn import functional as Func
import numpy as np
import tensorly as tl
import time

import import_ipynb
from helper_functions.generate_data import *
from helper_functions.metrics import *
from helper_functions.models import *
import copy
class NTF_KL:
    def __init__(self,input_parameters,network_parameters,simulation_paramaters,flags,init_parameters):
        self.I = network_parameters['I']
        self.K = network_parameters['K']
        self.F = network_parameters['F']
        self.flag_groundtruth = input_parameters['flag_groundtruth']
        self.observed_data = input_parameters['observed_data']
        
        self.no_of_BCD_iterations = simulation_paramaters['no_of_BCD_iterations']
        self.no_inner_iter_MM = simulation_paramaters['no_inner_iter_MM']      
        self.tol = simulation_paramaters['tol']
        if not self.flag_groundtruth:
            self.ind_val = simulation_paramaters['ind_val']
            self.Y_val = simulation_paramaters['Y_val']
        
        
        self.network_parameters=network_parameters
        self.simulation_paramaters=simulation_paramaters
        self.flags=flags
        
        self.init_parameters=init_parameters
        

    def train_NTF_KL(self):
        # Get observed data
        Y  = self.observed_data.Y.clone().detach()
        Omega = self.observed_data.Omega.clone().detach()
        if self.flag_groundtruth==1:
            A_g = self.observed_data.A_true.copy()
            Lambda_g   = self.observed_data.Lambda.clone().detach()
            M_g      = self.observed_data.T.clone().detach()
        if self.flag_groundtruth==2:
            P_g   = self.observed_data.P.clone().detach()
        Y1 = Y.view(np.prod(self.I,dtype=np.int),1)
        

        
        # Get the linear indices corresponding to Omega 
        Y_isnannot = torch.logical_not(torch.isnan(Y1.squeeze()))
        Omega_linear = torch.nonzero(Y_isnannot)
        Omega_linear = Omega_linear.squeeze()
        

        

        

       #if self.init_parameters['flag_init']:
       #    M      = self.init_parameters['M_init']
       #    A      = self.init_parameters['A_init'].copy()
       #    Lambda = self.init_parameters['Lambda_init']
       #else:       
       #    # Algorithm Initialization
       #    data_initialization=Initialization(self.network_parameters,self.simulation_paramaters,self.flags,Z)
       #    data_initialization.initialize_TF() 
       #    M      = data_initialization.T.clone().detach()
       #    A      = data_initialization.A1.copy()
       #    Lambda = data_initialization.Lambda.clone().detach()
        if self.init_parameters['flag_init']:
            A = self.init_parameters['A_init'].copy()
        else:
            # Generate tensor factor matrices and feature vector z
            A=[]
            for k in range(self.K):
                U = np.random.rand(self.I[k],self.F) # Plant embeddings
                A.append(U);


        # Generate Lambda tensor
        weights = None
        M = torch.tensor(tl.cp_to_tensor([weights,A]))
        Lambda =M
        print('check1')


        #Initialize output arrays
        cost= [np.nan]*self.no_of_BCD_iterations
        U_mse= [np.nan]*self.no_of_BCD_iterations
        lambda_mre= [np.nan]*self.no_of_BCD_iterations
        timestamps = [0]*self.no_of_BCD_iterations
        rRMSE = [np.nan]*self.no_of_BCD_iterations
        

        # Calculate the metrics 
        P = torch.ones(self.I)
        Y_pred = Lambda
        Y_pred_prev=Y_pred
        Y_pred_best = Y_pred
        if self.flag_groundtruth==0:
            print('check2')
            Y_val_pred = prediction(Y_pred,self.ind_val)
            print('check3')
            rRMSE[0] = get_rRMSE(Y_val_pred,self.Y_val) 
            print('check4')
            rRMSE_best=rRMSE[0]
        cost[0] = get_totalcost(M,P,Y,Omega)
        if self.flag_groundtruth==1:
            U_mse[0] = 0#getMSE(A,A_g)
            lambda_mre[0] = getMSE_entry(Lambda,Lambda_g)#torch.norm(Lambda-Lambda_g,p=1)/(I*J*K)
            print('#################################################################################')
            print('BCD iter [{}/{}], cost:{:.4f}, MSE of U:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0], U_mse[0]))
            print('BCD iter [{}/{}],  MRE of Lambda:{:.4f}'.format(0, self.no_of_BCD_iterations, lambda_mre[0]))
            print('#################################################################################')
        elif self.flag_groundtruth==0:           
            print('#################################################################################')
            print('BCD iter [{}/{}], cost:{:.4f}, rRMSE_val:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0],rRMSE[0]))
            print('#################################################################################')
        else:
            p_mre[0] =  getMSE_entry(P,P_g)# torch.norm(P-P_g,p=1)/(I*J*K)
            print('#################################################################################')
            print('BCD iter [{}/{}], cost:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0]))
            print('#################################################################################')            
            
        #Start training for U,V,W...................
        ts1 = time.time()
        ts2 = time.time()
        timestamps[0] = ts2-ts1
        for k in range(self.K):
            A[k]=torch.tensor(A[k],dtype=torch.float)
        loss_prev=0
        eps=1e-12
        for iter in range(self.no_of_BCD_iterations-1): 
            for k in range(self.K):                       
                num_rows = A[k].size()[0]
                Psi = torch.zeros(self.I[k],self.F)
                Phi = torch.zeros(self.I[k],self.F)
                #print('num_row='+str(num_rows))
                for jj in range(num_rows):
                    k_index_sel = torch.nonzero(Omega[:,k]==jj)
                    k_index_sel = k_index_sel.squeeze()
                    sparse_indices=Omega[k_index_sel,:]
                    len_observed = len(sparse_indices)
                    W = torch.ones(len_observed,self.F)
                    V_hat = torch.zeros(len_observed,1)
                    for p in range(len_observed):
                        for n in range(0,k):
                            W[p,:] = W[p,:]*A[n][sparse_indices[p,n],:]
                        for n in range(k+1,self.K):
                            W[p,:] = W[p,:]*A[n][sparse_indices[p,n],:]
                    for epoch in range(self.no_inner_iter_MM): 
                        for p in range(len_observed):
                            V_hat[p] = Y[sparse_indices[p,0],sparse_indices[p,1],sparse_indices[p,2]]/torch.clamp(torch.dot(W[p,:],A[k][sparse_indices[p,k],:]),min=1e-10)
                            Phi[jj,:] += V_hat[p]*W[p,:]
                            Psi[jj,:] +=P[sparse_indices[p,0],sparse_indices[p,1],sparse_indices[p,2]]*W[p,:]                           
                        A[k][jj,:] = torch.div(A[k][jj,:]*Phi[jj,:],torch.clamp(Psi[jj,:],min=1e-10))

            weights = None
            features=[]
            for k in range(self.K):
                A[k][torch.isnan(A[k])]=1e-6
                A[k][torch.isinf(A[k])]=1e-6
                features.append(A[k].numpy())
            M = tl.cp_to_tensor([weights,features])
            M = torch.tensor(M)
            Lambda = M
            Lambda1 = Lambda.view(np.prod(self.I,dtype=np.int),1)
            loss = get_tensorcost(M,P,Y,Omega)
            #print('epoch [{}/{}], U loss function:{:.4f}'.format(epoch + 1, self.no_iteration_MM, loss))
            if (abs(loss-loss_prev)/abs(loss_prev)) < self.tol:
                break
            loss_prev=loss
            ts2 = time.time()     

            
                
                

            # Computing total loss and MSE
            timestamps[iter+1] = ts2-ts1



            #print(A[0])
            #print(A_g[0])
            Y_pred = Lambda
            if self.flag_groundtruth==0:
                Y_val_pred = prediction(Y_pred,self.ind_val)
                rRMSE[iter+1] = get_rRMSE(Y_val_pred,self.Y_val) 

            cost[iter+1]=get_totalcost(M,P,Y,Omega)
            if self.flag_groundtruth==1:
                U_mse[iter+1] = 0#getMSE(A,A_g)
                lambda_mre[iter+1] = getMSE_entry(Lambda,Lambda_g) #torch.norm(Lambda-Lambda_g,p=1)/(I*J*K)
                print('#################################################################################')
                print('BCD iter [{}/{}], total loss function:{:.4f}, MSE of U:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1], U_mse[iter+1]))
                print('BCD iter [{}/{}], MRE of Lambda:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, lambda_mre[iter+1]))
                print('#################################################################################')
            elif self.flag_groundtruth==0:              
                print('#################################################################################')
                print('BCD iter [{}/{}], cost:{:.4f}, rRMSE_val:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1],\
                                                                               rRMSE[iter+1]))
                print('#################################################################################')
            else:  
                print('#################################################################################')
                print('BCD iter [{}/{}], cost:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1]))
                print('#################################################################################')

 
            
            if self.flag_groundtruth==0:
                stop_condition = iter > 1 and rRMSE[iter+1]>rRMSE[iter]
                if rRMSE[iter+1] < rRMSE_best:
                    Y_pred_best = Y_pred
                    rRMSE_best = rRMSE[iter+1]
            elif self.flag_groundtruth==1:
                stop_condition = abs(cost[iter]-cost[iter+1])/abs(cost[iter]) < self.tol
            else:
                stop_condition = abs(cost[iter]-cost[iter+1])/abs(cost[iter]) < self.tol
            if stop_condition: #(abs(cost[iter]-cost[iter+1])/abs(cost[iter])) < self.tol:
                Y_pred = Y_pred_best
                cost[iter+2:] =[float("nan")]*(self.no_of_BCD_iterations-iter-2)
                timestamps[iter+2:]=[float("nan")]*(self.no_of_BCD_iterations-iter-2)
                if self.flag_groundtruth==1:
                    U_mse[iter+2:]  =[float("nan")]*(self.no_of_BCD_iterations-iter-2)
                    lambda_mre[iter+2:] =[float("nan")]*(self.no_of_BCD_iterations-iter-2)
                break

            Y_pred_prev = Y_pred
        
        
        if self.flag_groundtruth==1:
            scaling_fact_lambda = torch.div(Lambda_g,Lambda)
            scaling_fact_lambda = scaling_fact_lambda.view(1,torch.numel(P))                
            output_parameters = {'cost':cost,
                                 'U_mse':U_mse,
                                 'lambda_mre':lambda_mre,
                                 'timestamps':timestamps,
                                 'Y_pred':Y_pred
            }
        else: 
            output_parameters = {'cost':cost,
                                 'timestamps':timestamps,
                                 'Y_pred':Y_pred_best.clone().detach(),
                                 'Lambda':Lambda.clone().detach(),
                                 'A':A.copy(),
            } 

        A =[]

        return output_parameters
    

def initialize_p(m,val):
        m.w.data = val
