import torch
from torch import nn
from torch.nn import functional as Func
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import statistics
import import_ipynb
import pandas as pd
from helper_functions.metrics import *
from helper_functions.models import *
from helper_functions.plots import *
from helper_functions.generate_data import *
from helper_functions.utils import *
from algorithms.UNCLE_TC_GLOBAL import *
from algorithms.NTF_KL import *
from algorithms.BPTF import *
from algorithms.HaLRTC import *
import copy
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from tensorly.decomposition import non_negative_parafac


# Problem Parameters
K = 3  # Dimension of the tensor
I = [20]*K
F = 3   # Rank of the tensor for algorithm
F_true = 3 # Rank for the tensor for generation
D = 10   # Dimension of side feature vector
gamma =15 # tensor latent factor distribution paramater
flag_NN_detection=1
flag_tensor_factorization=1
flag_NN_detection_linear=0
obs_count_fraction = 0.2
obs_feature_fraction = 0.3
obs_feature_equal_fraction = 0.2
obs_feature_equal_noise_dB = 40#float('inf')
g_function_type='cube'
file_name='_Omega_smaller_Xi'


# Network Paramaters
cuda_flag=0
hidden_layer_g = 3
hidden_unit_g = 20
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Simulation Parameters
no_of_trials = 1
no_of_BCD_iterations=2
no_of_epochs_theta=5
no_iteration_MM=5
no_inner_iter_MM=1
tol = 1e-6
learning_rate_g=0.001
mu = 2000 #regularization for features with no observation
flag_auto_mu_selection = 0
flag_normalized_cost=1

plot_dir = "plots/"

# Seed setting
seed=1  #9
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)
	
	
network_parameters = {
        'I':I,
        'K':K,
        'F':F,
        'F_true':F_true,
        'D':D,
        'gamma':gamma,
        'obs_count_fraction' : obs_count_fraction,
        'obs_feature_fraction':obs_feature_fraction,
        'obs_feature_equal_fraction':obs_feature_equal_fraction,
        'obs_feature_equal_noise_dB':obs_feature_equal_noise_dB,
        'g_function_type':g_function_type
}


simulation_paramaters = {'no_of_BCD_iterations': no_of_BCD_iterations,
                         'no_of_epochs_theta': no_of_epochs_theta,
                         'no_iteration_MM' : no_iteration_MM,
                         'no_inner_iter_MM' : no_inner_iter_MM,
                         'tol': tol,
                         'batch_size':batch_size,
                         'hidden_layer_g':hidden_layer_g,
                         'hidden_unit_g':hidden_unit_g,
                         'learning_rate_g':learning_rate_g,
                         'mu':mu,
                         'flag_auto_mu_selection':flag_auto_mu_selection,
                        'loss_type':'euclidean',
						'flag_normalized_cost':flag_normalized_cost}


flags = {'flag_NN_detection': flag_NN_detection,
        'flag_tensor_factorization':flag_tensor_factorization,
        'flag_NN_detection_linear':flag_NN_detection_linear}

network_parameters_NTF_KL = {
    'I':I,
    'K':K,
    'F':F}

simulation_paramaters_NTF_KL = {'no_of_BCD_iterations': no_of_BCD_iterations,
                         'no_inner_iter_MM' : no_inner_iter_MM,
                         'tol': tol}

dir_paths = {'plot_dir':plot_dir}


#output_paramaters=[]
cost = np.zeros((no_of_trials,no_of_BCD_iterations))
U_mse = np.zeros((no_of_trials,no_of_BCD_iterations))
p_mre=np.zeros((no_of_trials,no_of_BCD_iterations))
lambda_mre = np.zeros((no_of_trials,no_of_BCD_iterations))
timestamps=np.zeros((no_of_trials,no_of_BCD_iterations))
scaling_fact_p= np.zeros((no_of_trials,np.prod(I,dtype=int)))
scaling_fact_lambda= np.zeros((no_of_trials,np.prod(I,dtype=int)))

cost_linear = np.zeros((no_of_trials,no_of_BCD_iterations))
U_mse_linear = np.zeros((no_of_trials,no_of_BCD_iterations))
p_mre_linear=np.zeros((no_of_trials,no_of_BCD_iterations))
lambda_mre_linear = np.zeros((no_of_trials,no_of_BCD_iterations))
timestamps_linear=np.zeros((no_of_trials,no_of_BCD_iterations))
scaling_fact_p_linear= np.zeros((no_of_trials,np.prod(I,dtype=int)))
scaling_fact_lambda_linear= np.zeros((no_of_trials,np.prod(I,dtype=int)))

cost_NTF_KL = np.zeros((no_of_trials,no_of_BCD_iterations))
U_mse_NTF_KL = np.zeros((no_of_trials,no_of_BCD_iterations))
lambda_mre_NTF_KL = np.zeros((no_of_trials,no_of_BCD_iterations))
timestamps_NTF_KL=np.zeros((no_of_trials,no_of_BCD_iterations))


if obs_count_fraction==obs_feature_fraction:
    UNCLE_TC = UNCLE_TC_GLOBAL
else:
    UNCLE_TC=UNCLE_TC_GLOBAL
	
for i in range(no_of_trials):       
    # Generate groudtruth data
    observed_data=GroundTruth(network_parameters)
    observed_data.generate_observed_counts_Nonlinear_TF()
    input_parameters = {'flag_groundtruth' : 1,
                        'observed_data': observed_data}

    # Algorithm Initialization
    data_initialization=Initialization(network_parameters,simulation_paramaters,flags,observed_data.Z)
    data_initialization.initialize_TF() 
    init_parameters = { 'flag_init' : 1,
                        'M_init': data_initialization.T.clone().detach(),
                        'A_init': data_initialization.A1.copy(),
                        'Lambda_init':data_initialization.Lambda.clone().detach(),
                        'P_init':data_initialization.P.clone().detach(),
                        'GTHETA':data_initialization.GTHETA.clone().detach(),
                        'GTHETA_linear':data_initialization.GTHETA_linear.clone().detach(),
                      }
    
    ########### Run the proposed algorithm #########################################
    print('Running #trial = [{}/{}]'.format(i, no_of_trials))
    flags['flag_NN_detection_linear']=0
    trainer = UNCLE_TC(input_parameters,network_parameters,simulation_paramaters,flags,init_parameters)
    out=trainer.train_UNCLE_TC()
    cost[i,:]=out['cost']
    U_mse[i,:]=out['U_mse']
    p_mre[i,:]=out['p_mre']
    lambda_mre[i,:]=out['lambda_mre']
    timestamps[i,:]=out['timestamps']
    scaling_fact_p[i,:]=out['scaling_fact_p']
    scaling_fact_lambda[i,:]=out['scaling_fact_lambda']
    Y_pred=out['Y_pred']
    # Algorithm Initialization
    data_initialization=Initialization(network_parameters,simulation_paramaters,flags,observed_data.Z)
    data_initialization.initialize_TF() 
    init_parameters = { 'flag_init' : 1,
                        'M_init': data_initialization.T.clone().detach(),
                        'A_init': data_initialization.A1.copy(),
                        'Lambda_init':data_initialization.Lambda.clone().detach(),
                        'P_init':data_initialization.P.clone().detach(),
                        'GTHETA':data_initialization.GTHETA.clone().detach(),
                        'GTHETA_linear':data_initialization.GTHETA_linear.clone().detach(),
                      }
    
    ######### Run the linear version #########################################
    print('Running #trial = [{}/{}]'.format(i, no_of_trials))
    flags['flag_NN_detection_linear']=1
    trainer = UNCLE_TC(input_parameters,network_parameters,simulation_paramaters,flags,init_parameters)
    out=trainer.train_UNCLE_TC()
    cost_linear[i,:]=out['cost']
    U_mse_linear[i,:]=out['U_mse']
    p_mre_linear[i,:]=out['p_mre']
    lambda_mre_linear[i,:]=out['lambda_mre']
    timestamps_linear[i,:]=out['timestamps']
    scaling_fact_p_linear[i,:]=out['scaling_fact_p']
    scaling_fact_lambda_linear[i,:]=out['scaling_fact_lambda']
    Y_pred_linear=out['Y_pred']
    ############# Run baselines ####################################################
    # Algorithm Initialization
    data_initialization=Initialization(network_parameters,simulation_paramaters,flags,observed_data.Z)
    data_initialization.initialize_TF() 
    init_parameters = { 'flag_init' : 1,
                        'M_init': data_initialization.T.clone().detach(),
                        'A_init': data_initialization.A1.copy(),
                        'Lambda_init':data_initialization.Lambda.clone().detach(),
                        'P_init':data_initialization.P.clone().detach(),
                        'GTHETA':data_initialization.GTHETA.clone().detach(),
                        'GTHETA_linear':data_initialization.GTHETA_linear.clone().detach(),
                      }
    print('Running #trial = [{}/{}]'.format(i, no_of_trials))
    trainer1 = NTF_KL(input_parameters,network_parameters_NTF_KL,simulation_paramaters_NTF_KL,flags,init_parameters)
    out1=trainer1.train_NTF_KL()
    cost_NTF_KL[i,:]=out1['cost']
    U_mse_NTF_KL[i,:]=out1['U_mse']
    lambda_mre_NTF_KL[i,:]=out1['lambda_mre']
    timestamps_NTF_KL[i,:]=out1['timestamps'] 
    Y_pred_NTF=out1['Y_pred']


########################################################################################################
#######################Saving data########################################################   
np.save(plot_dir+'p_mre'+file_name+'.npy',p_mre)
np.save(plot_dir+'lambda_mre'+file_name+'.npy',lambda_mre)

np.save(plot_dir+'p_mre_linear'+file_name+'.npy',p_mre_linear)
np.save(plot_dir+'lambda_mre_linear'+file_name+'.npy',lambda_mre_linear)

np.save(plot_dir+'lambda_mre_NTF_KL'+file_name+'.npy',lambda_mre_NTF_KL)


# Calculate mean of the results
p_mre_avg=np.nanmean(p_mre,axis=0)  
p_mre_std=np.nanstd(p_mre,axis=0) 
lambda_mre_avg=np.nanmean(lambda_mre,axis=0) 
lambda_mre_std=np.nanstd(lambda_mre,axis=0) 

# Calculate mean of the results
p_mre_avg_linear=np.nanmean(p_mre_linear,axis=0)  
p_mre_std_linear=np.nanstd(p_mre_linear,axis=0)  
lambda_mre_avg_linear=np.nanmean(lambda_mre_linear,axis=0) 
lambda_mre_std_linear=np.nanstd(lambda_mre_linear,axis=0) 

# Calculate mean of the results
lambda_mre_avg_NTF_KL=np.nanmean(lambda_mre_NTF_KL,axis=0) 
lambda_mre_std_NTF_KL=np.nanstd(lambda_mre_NTF_KL,axis=0) 

result = [['UNCLE_TC',p_mre_avg[-1],\
            p_mre_std[-1],lambda_mre_avg[-1],lambda_mre_std[-1]],
          ['UNCLE_TC_linear',p_mre_avg_linear[-1],\
            p_mre_std_linear[-1],lambda_mre_avg_linear[-1],lambda_mre_std_linear[-1]],
          ['NTF-KL',0,\
            0,lambda_mre_avg_NTF_KL[-1],lambda_mre_std_NTF_KL[-1]]
         ]
Final_results = pd.DataFrame(result, columns=['Algorithms','MAE_p (mean)','MAE_p (std)','MAE_lambda (mean)', 'MAE_lambda (std)'])      
Final_results.to_csv(plot_dir+'final_results'+file_name+'.csv', index=False)


#################################################################################################################
#################################Plotting the results##########################################################
#plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(tight_layout=True)
ax.semilogy(p_mre_avg.flatten(),'go--', linewidth=2, markersize=5,markevery=3,label='UNCLE-TC')
ax.semilogy(p_mre_avg_linear.flatten(),'ro--', linewidth=2, markersize=5,markevery=3,label='UNCLE-TC (Linear)')

ax.set_xlabel('iterations')
ax.legend(loc='best')
ax.set_ylabel(r'$\text{MAE}_p$')
ax.grid(True)  
plt.savefig(plot_dir+'p_mre'+file_name+'.jpg')
plt.savefig(plot_dir+'p_mre'+file_name+'.eps')

#
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(tight_layout=True)
ax.semilogy(lambda_mre_avg.flatten(),'go--', linewidth=2, markersize=5,markevery=3,label='UNCLE-TC')
ax.semilogy(lambda_mre_avg_linear.flatten(),'ro--', linewidth=2, markersize=5,markevery=3,label='UNCLE-TC (Linear)')
ax.semilogy(lambda_mre_avg_NTF_KL.flatten(),'mo--', linewidth=2, markersize=5,markevery=5,label='NTF-CPD-KL')
ax.set_xlabel('iterations')
plt.legend(loc='best')
#ax.set_ylabel(r'MAE of $\lambda_{\boldsymbol{i}}$s')
ax.set_ylabel(r'$\text{MAE}_\lambda$')
ax.grid(True)  
plt.savefig(plot_dir+'lambda_mre'+file_name+'.jpg')
plt.savefig(plot_dir+'lambda_mre'+file_name+'.eps')




