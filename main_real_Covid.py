import torch
from torch import nn
from torch.nn import functional as Func
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import statistics
import import_ipynb
from helper_functions.models import *
from helper_functions.plots import *
from helper_functions.generate_data import *
from helper_functions.utils import *
from helper_functions.PPI_Dataset import *
from helper_functions.Covid_Dataset import *
from algorithms.UNCLE_TC_GLOBAL import *
from algorithms.NTF_KL import *
from algorithms.BPTF import *
from algorithms.HaLRTC import *
from helper_functions.metrics import *

import copy
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import non_negative_tucker
import pandas as pd
import time
import os
import os.path

# Problem Parameters
dataset = "Covid" #Avian, #Covid
if dataset=="PPI":
	K=3
	I = [50,50,37] # No of plants, No of pollinators, No of years
	F_list = [10]   # Rank of the tensor
	D = 8  # dimension of the detetcion feature vector
elif dataset=="Covid":
	K=3
	I = [43,80,60] # County latitude, County longitude, No of dates
	F_list = [15]   # Rank of the tensor
	D = 24  # dimension of the detetcion feature vector
else:
	raise Exception('Invalid dataset id')
	
	
#flags for Algorithms
flag_UNCLE_TC=1
flag_UNCLE_LINEAR=0
flag_NTF_KL=1
flag_BPTF=0
flag_HaLRTC=0
flag_NTF_LS=0
flag_NTF_Tucker_LS=0
	
gamma =1 # tensor latent factor distribution paramater
flag_NN_detection=1
flag_tensor_factorization=1
flag_NN_detection_linear=0


# Network Paramaters
cuda_flag=0
hidden_layer_list = [3]
hidden_unit_list = [20]
batch_size_list = [64]
learning_rate_list=[0.001]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Simulation Parameters
no_of_trials = 1
fold_num = 5
no_of_BCD_iterations=20
no_of_epochs_theta=1
no_iteration_MM=1
no_inner_iter_MM=1
tol = 1e-6
mu = 1000 #regularization for features with no observation
flag_auto_mu_selection = 0

plot_dir = "plots/"
result_dir = "results/"+dataset+"/"

# Seed setting
seed=7
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
	torch.cuda.manual_seed_all(seed)
	
network_parameters = {
		'I':I,
		'K':K,
		'F':None,
		'D':D,
		'gamma':gamma,
		'obs_count_fraction' : None,
		'obs_feature_fraction':None,
		}

flags = {'flag_NN_detection': flag_NN_detection,
		'flag_tensor_factorization':flag_tensor_factorization,
		'flag_NN_detection_linear':flag_NN_detection_linear
		}




# Get observed data from the dataset
if dataset=="PPI":
	observed_data=PPI_Dataset(network_parameters)
elif dataset=="Covid":
	observed_data=Covid_Dataset(network_parameters)
else:
	raise Exception('Invalid dataset id')

observed_data.load_data(0)
I_selected_indices = observed_data.I_selected_indices.numpy()
J_selected_indices = observed_data.J_selected_indices.numpy()
K_selected_indices = observed_data.K_selected_indices.numpy()
# Get k-Fold data
Y_train_w_nan, \
Y_train_wo_nan, \
Omega_Y_train, \
Omega_linear_Y_train, \
Y_val, \
ind_val, \
Y_test, \
ind_test = observed_data.partition_k_folds(fold_num)
network_parameters['obs_count_fraction']=observed_data.obs_count_fraction
network_parameters['obs_feature_fraction']=observed_data.obs_feature_fraction


# CSV files to write the results
UNCLE_TC_results = pd.DataFrame(columns=['type','# trial','# fold','rank',\
								  '# hidden layers','# hidden units','learning rate',\
								  'batch size','rRMSE','MAPE',\
								  'F1 Score','Hamming Loss','runtime (s)'])
UNCLE_TC_linear_results = pd.DataFrame(columns=['type','# trial','# fold','rank',\
								  '# hidden layers','# hidden units','learning rate',\
								  'batch size','rRMSE','MAPE',\
								  'F1 Score','Hamming Loss','runtime (s)'])
NTF_KL_results = pd.DataFrame(columns=['type','# trial','# fold','rank',\
								 'rRMSE','MAPE','F1 Score','Hamming Loss','runtime (s)'])
BPTF_results = pd.DataFrame(columns=['type','# trial','# fold','rank', 'alpha',\
								  'rRMSE','MAPE','F1 Score','Hamming Loss','runtime (s)'])
HaLRTC_results = pd.DataFrame(columns=['type','# trial','# fold', 'rho',\
								  'rRMSE','MAPE','F1 Score','Hamming Loss','runtime (s)'])
NTF_LS_results = pd.DataFrame(columns=['type','# trial','# fold','rank',\
								  'rRMSE','MAPE','F1 Score','Hamming Loss','runtime (s)'])
NTF_Tucker_LS_results = pd.DataFrame(columns=['type','# trial','# fold','rank',\
								  'rRMSE','MAPE','F1 Score','Hamming Loss','runtime (s)'])

if os.path.exists(result_dir+'UNCLE_TC_results.csv'):
	os.remove(result_dir+'UNCLE_TC_results.csv')
if os.path.exists(result_dir+'UNCLE_TC_linear_results.csv'):
	os.remove(result_dir+'UNCLE_TC_linear_results.csv')
if os.path.exists(result_dir+'NTF_KL_results.csv'):
	os.remove(result_dir+'NTF_KL_results.csv')
if os.path.exists(result_dir+'BPTF_results.csv'):
	os.remove(result_dir+'BPTF_results.csv')
if os.path.exists(result_dir+'HaLRTC_results.csv'):
	os.remove(result_dir+'HaLRTC_results.csv')
if os.path.exists(result_dir+'NTF_LS_results.csv'):
	os.remove(result_dir+'NTF_LS_results.csv')
if os.path.exists(result_dir+'NTF_Tucker_LS_results.csv'):
	os.remove(result_dir+'NTF_Tucker_LS_results.csv')




#Initialize result arrays
metrics_UNCLE_TC = [0]*4
metrics_UNCLE_TC[0] = np.zeros((no_of_trials,fold_num)) #rRMSE
metrics_UNCLE_TC[1] = np.zeros((no_of_trials,fold_num)) #MAPE
metrics_UNCLE_TC[2] = np.zeros((no_of_trials,fold_num)) #F1 score
metrics_UNCLE_TC[3] = np.zeros((no_of_trials,fold_num)) #Hamming Loss



metrics_UNCLE_TC_linear = [0]*4
metrics_UNCLE_TC_linear[0] = np.zeros((no_of_trials,fold_num))
metrics_UNCLE_TC_linear[1] = np.zeros((no_of_trials,fold_num))
metrics_UNCLE_TC_linear[2] = np.zeros((no_of_trials,fold_num))
metrics_UNCLE_TC_linear[3] = np.zeros((no_of_trials,fold_num))

metrics_NTF_KL = [0]*4
metrics_NTF_KL[0] = np.zeros((no_of_trials,fold_num))
metrics_NTF_KL[1] = np.zeros((no_of_trials,fold_num))
metrics_NTF_KL[2] = np.zeros((no_of_trials,fold_num))
metrics_NTF_KL[3] = np.zeros((no_of_trials,fold_num))

metrics_BPTF = [0]*4
metrics_BPTF[0] = np.zeros((no_of_trials,fold_num))
metrics_BPTF[1] = np.zeros((no_of_trials,fold_num))
metrics_BPTF[2] = np.zeros((no_of_trials,fold_num))
metrics_BPTF[3] = np.zeros((no_of_trials,fold_num))

metrics_HaLRTC = [0]*4
metrics_HaLRTC[0] = np.zeros((no_of_trials,fold_num))
metrics_HaLRTC[1] = np.zeros((no_of_trials,fold_num))
metrics_HaLRTC[2] = np.zeros((no_of_trials,fold_num))
metrics_HaLRTC[3] = np.zeros((no_of_trials,fold_num))


metrics_NTF_LS = [0]*4
metrics_NTF_LS[0] = np.zeros((no_of_trials,fold_num))
metrics_NTF_LS[1] = np.zeros((no_of_trials,fold_num))
metrics_NTF_LS[2] = np.zeros((no_of_trials,fold_num))
metrics_NTF_LS[3] = np.zeros((no_of_trials,fold_num))

metrics_NTF_Tucker_LS = [0]*4
metrics_NTF_Tucker_LS[0] = np.zeros((no_of_trials,fold_num))
metrics_NTF_Tucker_LS[1] = np.zeros((no_of_trials,fold_num))
metrics_NTF_Tucker_LS[2] = np.zeros((no_of_trials,fold_num))
metrics_NTF_Tucker_LS[3] = np.zeros((no_of_trials,fold_num))


UNCLE_TC = UNCLE_TC_GLOBAL


r1=0; r2=0; r3=0; r4=0; r5=0; r6=0; r7=0; r8=0
for i in range(no_of_trials):
	for k in range(fold_num):
		observed_data.Y = Y_train_w_nan[k]
		observed_data.Omega = Omega_Y_train[k]
		observed_data.Omega_linear = Omega_linear_Y_train[k]
		observed_data.Theta = Omega_Y_train[k]
		observed_data.Theta_linear = Omega_linear_Y_train[k]
		input_parameters = {'flag_groundtruth' : 0,
							'observed_data': observed_data}
		init_parameters = { 'flag_init' : 0}
################################################ Run the proposed algorithm with UNCLE_TC #########################################
################################################################################################################################
		if flag_UNCLE_TC:
			print('Start Training and Validation  for UNCLE_TC..............................')
			rRMSE_best=float('inf')
			for f,F in enumerate(F_list):
				network_parameters['F']=F
				for hl_g,hidden_layer_g in enumerate(hidden_layer_list):
					hidden_layer_h = hidden_layer_g
					for hu_g,hidden_unit_g in enumerate(hidden_unit_list):
						hidden_unit_h = hidden_unit_g
						for l_g,learning_rate_g in enumerate(learning_rate_list):
							learning_rate_h = learning_rate_g
							for b,batch_size in enumerate(batch_size_list):
								simulation_paramaters = {'no_of_BCD_iterations': no_of_BCD_iterations,
										 'no_of_epochs_theta': no_of_epochs_theta,
										 'no_iteration_MM' : no_iteration_MM,
										 'no_inner_iter_MM' : no_inner_iter_MM,
										 'tol': tol,
										 'batch_size':batch_size,
										 'hidden_layer_g':hidden_layer_g,
										 'hidden_unit_g':hidden_unit_g,
										 'learning_rate_g':learning_rate_g,
										 'ind_val':ind_val[k],
										 'Y_val' :Y_val[k],
										 'mu':mu,
										 'flag_auto_mu_selection':flag_auto_mu_selection,
										 'loss_type':'gen_KL'}
										
								flags['flag_NN_detection_linear']=0
								trainer = UNCLE_TC(input_parameters,network_parameters,simulation_paramaters,flags,init_parameters)
								out=trainer.train_UNCLE_TC()
								time_s = np.nansum(out['timestamps'])
								Y_full_pred=out['Y_pred']
								Y_val_pred = prediction(Y_full_pred,ind_val[k])
								e_metrics = evaluation(Y_val_pred,Y_val[k])
								rRMSE = e_metrics[0]
								##################################Recording P values###################################
								P_copy =out['P']
								file_P = open('P_with_indices_'+str(k)+'.txt','w')
								for ii,ind1 in enumerate(I_selected_indices):
									for jj,ind2 in enumerate(J_selected_indices):
										for kk,ind3 in enumerate(K_selected_indices):
											file_P.write( str(ind1)+'\t'+str(ind2)+'\t'+str(ind3)+'\t'+str(P_copy[ii,jj,kk])+'\n' )
								file_P.close()	
								##################################Recording P values###################################
								print('#################################################################################')
								print('VALIDATION - UNCLE-TC : #trial = [{}/{}], #fold [{}/{}], rank={}, hidden_layer={},'.format(i, no_of_trials, k,fold_num,F, hidden_layer_g))
								print('hidden_units={},learning_rate={:.4f}, batch_size={}'.format(hidden_unit_g,learning_rate_g, batch_size))
								print('rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}, runtime = {:.4f}'.format(e_metrics[0], \
									 e_metrics[1],e_metrics[2],e_metrics[3],time_s))
								print('#################################################################################')
								if rRMSE < rRMSE_best:
									UNCLE_TC_best_param = [F,hidden_layer_g,hidden_unit_g,learning_rate_g,batch_size]
									Y_sel = Y_full_pred
									rRMSE_best=rRMSE
									results_best = list(e_metrics)+[time_s]						  
								result = ['val',i, k, F, hidden_layer_g, hidden_unit_g, learning_rate_g,batch_size] + \
										  list(e_metrics)+[time_s]
								UNCLE_TC_results.loc[r1] = result
								UNCLE_TC_results.to_csv(result_dir+'UNCLE_TC_results.csv', index=False)
								r1=r1+1		
			print('Stop Training and Validation for UNCLE_TC..............................')
			print('Start Testing for UNCLE_TC..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k])   
			for j in range(4):
				metrics_UNCLE_TC[j][i,k] = results_test[j]
			print('##############################Test Results#####################################')
			print('TEST - UNCLE_TC : #trial = [{}/{}], #fold [{}/{}], rank={}, hidden_layer={},'.format(i, no_of_trials, k,\
											fold_num,UNCLE_TC_best_param[0], UNCLE_TC_best_param[1]))
			print('hidden_units={},learning_rate={:.4f}, batch_size={}'.format(UNCLE_TC_best_param[2],UNCLE_TC_best_param[3],\
																			   UNCLE_TC_best_param[4]))
			print('rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(results_test[0], \
									 results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')
			result = ['best',i, k]+ UNCLE_TC_best_param + results_best
			UNCLE_TC_results.loc[r1] = result
			UNCLE_TC_results.to_csv(result_dir+'UNCLE_TC_results.csv', index=False)
			r1=r1+1
			result = ['test',i, k]+ UNCLE_TC_best_param + list(results_test)+[0]
			UNCLE_TC_results.loc[r1] = result
			UNCLE_TC_results.to_csv(result_dir+'UNCLE_TC_results.csv', index=False)
			r1=r1+1
###############################################################################################################################
###############################################################################################################################


################################################ Run the proposed algorithm with UNCLE_TC with linear network #########################################
################################################################################################################################
		if flag_UNCLE_LINEAR:
			print('Start Training and Validation  for UNCLE_TC w linear net..............................')
			rRMSE_best=float('inf')
			for f,F in enumerate(F_list):
				network_parameters['F']=F
				for l_g,learning_rate_g in enumerate(learning_rate_list):
					learning_rate_h = learning_rate_g
					for b,batch_size in enumerate(batch_size_list):
						simulation_paramaters = {'no_of_BCD_iterations': no_of_BCD_iterations,
								 'no_of_epochs_theta': no_of_epochs_theta,
								 'no_iteration_MM' : no_iteration_MM,
								 'no_inner_iter_MM' : no_inner_iter_MM,
								 'tol': tol,
								 'batch_size':batch_size,
								 'hidden_layer_g':1,
								 'hidden_unit_g':5,
								 'learning_rate_g':learning_rate_g,
								 'ind_val':ind_val[k],
								 'Y_val' :Y_val[k],
								 'mu':mu,
								 'flag_auto_mu_selection':flag_auto_mu_selection,
								 'loss_type':'euclidean'
								}
						flags['flag_NN_detection_linear']=1
						trainer = UNCLE_TC(input_parameters,network_parameters,simulation_paramaters,flags,init_parameters)
						out=trainer.train_UNCLE_TC()
						time_s = np.nansum(out['timestamps'])
						Y_full_pred=out['Y_pred']
						Y_val_pred = prediction(Y_full_pred,ind_val[k])
						e_metrics = evaluation(Y_val_pred,Y_val[k])
						rRMSE = e_metrics[0]
						print('#################################################################################')
						print('VALIDATION - UNCLE_TC-linear : #trial = [{}/{}], #fold [{}/{}], rank={}, hidden_layer={},'.format(i, no_of_trials, k,fold_num,F, hidden_layer_g))
						print('hidden_units={},learning_rate={:.4f}, batch_size={}'.format(hidden_unit_g,learning_rate_g, batch_size))
						print('rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}, runtime = {:.4f}'.format(e_metrics[0], \
							 e_metrics[1],e_metrics[2],e_metrics[3],time_s))
						print('#################################################################################')
						if rRMSE < rRMSE_best:
							UNCLE_TC_linear_best_param = [F,hidden_layer_g,hidden_unit_g,learning_rate_g,batch_size]
							Y_sel = Y_full_pred
							rRMSE_best=rRMSE
							results_best = list(e_metrics)+[time_s]						  
						result = ['val',i, k, F, hidden_layer_g, hidden_unit_g, learning_rate_g,batch_size] + \
								  list(e_metrics)+[time_s]
						UNCLE_TC_linear_results.loc[r7] = result
						UNCLE_TC_linear_results.to_csv(result_dir+'UNCLE_TC_linear_results.csv', index=False)
						r7=r7+1		
			print('Stop Training and Validation for UNCLE_TC_linear..............................')
			print('Start Testing for UNCLE_TC_linear..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k])   
			for j in range(4):
				metrics_UNCLE_TC_linear[j][i,k] = results_test[j]
			print('##############################Test Results#####################################')
			print('TEST - UNCLE_TC-linear : #trial = [{}/{}], #fold [{}/{}], rank={}, hidden_layer={},'.format(i, no_of_trials, k,\
											fold_num,UNCLE_TC_linear_best_param[0], UNCLE_TC_linear_best_param[1]))
			print('hidden_units={},learning_rate={:.4f}, batch_size={}'.format(UNCLE_TC_linear_best_param[2],UNCLE_TC_linear_best_param[3],\
																			   UNCLE_TC_linear_best_param[4]))
			print('rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(results_test[0], \
									 results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')
			result = ['best',i, k]+ UNCLE_TC_linear_best_param + results_best
			UNCLE_TC_linear_results.loc[r7] = result
			UNCLE_TC_linear_results.to_csv(result_dir+'UNCLE_TC_linear_results.csv', index=False)
			r7=r7+1
			result = ['test',i, k]+ UNCLE_TC_linear_best_param + list(results_test)+[0]
			UNCLE_TC_linear_results.loc[r7] = result
			UNCLE_TC_linear_results.to_csv(result_dir+'UNCLE_TC_linear_results.csv', index=False)
			r7=r7+1
################################################################################################################################
################################################################################################################################


######################################################### Run NTF-KL ##############################################################
################################################################################################################################
		if flag_NTF_KL:
			print('Start Training and Validation for NTF-KL..............................')
			observed_data.Y = Y_train_w_nan[k]
			input_parameters = {'flag_groundtruth' : 0,
								'observed_data': observed_data}
			rRMSE_best=float('inf')
			simulation_paramaters = {'no_of_BCD_iterations': no_of_BCD_iterations,
									 'no_inner_iter_MM' : no_inner_iter_MM,
									 'tol': tol,
									 'ind_val':ind_val[k],
									 'Y_val' :Y_val[k]}
			for f,F in enumerate(F_list):
				network_parameters_ptf = {
				'I':I,
				'K':K,
				'F':F}
				trainer = NTF_KL(input_parameters,network_parameters_ptf,simulation_paramaters,flags,init_parameters)
				out=trainer.train_NTF_KL()
				time_s = np.nansum(out['timestamps'])
				Y_full_pred=out['Y_pred']
				Y_val_pred = prediction(Y_full_pred,ind_val[k])
				e_metrics = evaluation(Y_val_pred,Y_val[k])
				rRMSE = e_metrics[0]
				print('#################################################################################')
				print('VALIDATION- NTF-KL : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f},runtime = {:.4f}'.format(F,\
							e_metrics[0], e_metrics[1],e_metrics[2],e_metrics[3],time_s))
				print('#################################################################################')
				if rRMSE < rRMSE_best:
					NTF_KL_best_param = [F]
					Y_sel = Y_full_pred
					rRMSE_best=rRMSE
					results_best = list(e_metrics)+[time_s] 
				result = ['val',i, k, F]+list(e_metrics)+[time_s]
				NTF_KL_results.loc[r3] = result
				NTF_KL_results.to_csv(result_dir+'NTF_KL_results.csv', index=False)
				r3=r3+1
			print('Stop Training and Validation for NTF-KL..............................')
			print('Start testing for NTF-KL..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k])   
			for j in range(4):
				metrics_NTF_KL[j][i,k] = results_test[j]
			print('##############################Best Results#####################################')
			print('TEST- NTF-KL : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(F, \
					results_test[0], results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')
			result = ['best',i, k]+ NTF_KL_best_param + results_best
			NTF_KL_results.loc[r3] = result
			NTF_KL_results.to_csv(result_dir+'NTF_KL_results.csv', index=False)
			r3=r3+1
			
			result = ['test',i, k]+ NTF_KL_best_param +list(results_test)+[0]
			NTF_KL_results.loc[r3] = result
			NTF_KL_results.to_csv(result_dir+'NTF_KL_results.csv', index=False)
			r3=r3+1
################################################################################################################################
#################################################################################################################################


######################################################### Run BPTF ##############################################################
################################################################################################################################
		if flag_BPTF:
			# Training, Validating and testing on BPTF  
			alpha_list =[0.05,0.1,0.2]
			print('Start Training and Validation for BPTF..............................')
			BPTF_result_list=[]
			rRMSE_best=float('inf')
			for f,F in enumerate(F_list):
				for a,alpha0 in enumerate(alpha_list):
					ts1 = time.time()
					trainer = BPTF(n_modes=3, n_components=F, max_iter=100, tol=1e-4, smoothness=100,verbose=False,
							alpha=alpha0, debug=False)
					trainer.fit(np.array(Y_train_wo_nan[k]))
					ts2 = time.time()
					time_s=ts2-ts1
					Y_full_pred = trainer.reconstruct()
					Y_val_pred = prediction(torch.tensor(Y_full_pred),ind_val[k])
					e_metrics = evaluation(Y_val_pred,Y_val[k])
					rRMSE = e_metrics[0]
					print('#################################################################################')
					print('VALIDATION - BPTF : rank={}, alpha={:.4f}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f},runtime \
										= {:.4f}'.format(F, alpha0,\
					e_metrics[0], e_metrics[1],e_metrics[2],e_metrics[3],time_s))
					print('#################################################################################')
					if rRMSE < rRMSE_best:
						BPTF_best_param = [F,alpha0]
						Y_sel = torch.tensor(Y_full_pred)
						rRMSE_best=rRMSE
						results_best = list(e_metrics)+[time_s] 
					result = ['val',i, k, F, alpha0]+list(e_metrics)+[time_s]
					BPTF_results.loc[r4] = result
					BPTF_results.to_csv(result_dir+'BPTF_results.csv', index=False)
					r4=r4+1
			print('Stop Training and Validation for BPTF..............................')
			print('Start Testing for BPTF..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k])   
			for j in range(4):
				metrics_BPTF[j][i,k] = results_test[j]
			print('##############################Best Results#####################################')
			print('TEST - BPTF : rank={}, alpha={:.4f}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(BPTF_best_param[0],BPTF_best_param[1],\
					results_test[0], results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')  
			result = ['best',i, k]+ BPTF_best_param+ results_best
			BPTF_results.loc[r4] = result
			BPTF_results.to_csv(result_dir+'BPTF_results.csv', index=False)
			r4=r4+1
			result = ['test',i, k]+ BPTF_best_param+ list(results_test)+[0]
			BPTF_results.loc[r4] = result
			BPTF_results.to_csv(result_dir+'BPTF_results.csv', index=False)
			r4=r4+1
#################################################################################################################################
#################################################################################################################################


######################################################### Run HaLRTC ##############################################################
################################################################################################################################
		if flag_HaLRTC:
			# Training, Validating and testing on HaLRTC  
			rho_list =[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
			print('Start Training and Validation for HaLRTC..............................')
			rRMSE_best=float('inf')
			groundtruth_data = None
			observed_poisson_data=np.array(Y_train_wo_nan[k])
			for f,rho0 in enumerate(rho_list):
				trainer = HaLRTC(alpha = [1/3, 1/3, 1/3],rho=rho0,epsilon = 1e-3,maxiter = 100)
				ts1 = time.time()
				Y_full_pred=trainer.train_HaLRTC(groundtruth_data,observed_poisson_data,flag_groundtruth=0)
				ts2 = time.time()
				time_s = ts2-ts1
				Y_val_pred = prediction(torch.tensor(Y_full_pred),ind_val[k])
				e_metrics = evaluation(Y_val_pred,Y_val[k])
				rRMSE = e_metrics[0]
				print('#################################################################################')
				print('VALIDATION - HaLRTC : rho = {}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f},\
				runtime = {:.4f}'.format(rho0, e_metrics[0], e_metrics[1],e_metrics[2],e_metrics[3],time_s))
				print('#################################################################################')
				if rRMSE < rRMSE_best:
					HaLRTC_best_param = [rho0]
					Y_sel = torch.tensor(Y_full_pred)
					rRMSE_best=rRMSE
					results_best = list(e_metrics)+[time_s] 
				result = ['val',i, k, rho0]+list(e_metrics)+[time_s]
				HaLRTC_results.loc[r5] = result
				HaLRTC_results.to_csv(result_dir+'HaLRTC_results.csv', index=False)
				r5=r5+1
			
			print('Stop Training and Validation for HaLRTC..............................')
			print('Start Testing for HaLRTC..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k]) 
			for j in range(4):
				metrics_HaLRTC[j][i,k] = results_test[j]
			print('##############################Best Results#####################################')
			print('TEST - HaLRTC : rho={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(HaLRTC_best_param[0],\
			results_test[0], results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')   
			result = ['best',i, k]+ HaLRTC_best_param + results_best
			HaLRTC_results.loc[r5] = result
			HaLRTC_results.to_csv(result_dir+'HaLRTC_results.csv', index=False)
			r5=r5+1
			
			result = ['test',i, k]+ HaLRTC_best_param + list(results_test)+[0]
			HaLRTC_results.loc[r5] = result
			HaLRTC_results.to_csv(result_dir+'HaLRTC_results.csv', index=False)
			r5=r5+1
################################################################################################################################
################################################################################################################################


########################################################## Run NTF-LS ##############################################################
#################################################################################################################################
		if flag_NTF_LS:
			# Training, Validating and testing on NTF-LS  
			print('Start Training and Validation for NTF-LS..............................')
			rRMSE_best=float('inf')
			for f,F in enumerate(F_list):
				observed_poisson_data=np.array(Y_train_wo_nan[k])
				ts1 = time.time()
				factors = non_negative_parafac(observed_poisson_data, F, n_iter_max=100, init='random', svd='numpy_svd', tol=1e-06, random_state=None, verbose=0)
				Y_full_pred = torch.tensor(tl.cp_to_tensor(factors))  
				ts2 = time.time()
				time_s=ts2-ts1
				Y_val_pred = prediction(torch.tensor(Y_full_pred),ind_val[k])
				e_metrics = evaluation(Y_val_pred,Y_val[k])
				rRMSE = e_metrics[0]
				print('#################################################################################')
				print('VALIDATION - NTF-LS : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f},runtime = {:.4f}'.format(F,\
				e_metrics[0], e_metrics[1],e_metrics[2],e_metrics[3],time_s))
				print('#################################################################################')
				if rRMSE < rRMSE_best:
					NTF_LS_best_param = [F]
					Y_sel = torch.tensor(Y_full_pred)
					rRMSE_best=rRMSE
					results_best = list(e_metrics)+[time_s] 
				result = ['val',i, k, F]+list(e_metrics)+[time_s]
				NTF_LS_results.loc[r6] = result
				NTF_LS_results.to_csv(result_dir+'NTF_LS_results.csv', index=False)
				r6=r6+1
			print('Stop Training and Validation for NTF-LS..............................')
			print('Start Testing for NTF-LS..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k]) 
			for j in range(4):
				metrics_NTF_LS[j][i,k] = results_test[j]
			print('##############################Best Results#####################################')
			print('TEST - NTF-LS : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(NTF_LS_best_param[0], \
			results_test[0], results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')
			result = ['best',i, k]+ NTF_LS_best_param+results_best
			NTF_LS_results.loc[r6] = result
			NTF_LS_results.to_csv(result_dir+'NTF_LS_results.csv', index=False)
			r6=r6+1
			
			result = ['test',i, k]+ NTF_LS_best_param+list(results_test)+[0]
			NTF_LS_results.loc[r6] = result
			NTF_LS_results.to_csv(result_dir+'NTF_LS_results.csv', index=False)
			r6=r6+1
################################################################################################################################
################################################################################################################################


######################################################### Run NTF-Tucker-LS ##############################################################
################################################################################################################################
		if flag_NTF_Tucker_LS:
			# Training, Validating and testing on NTF-Tucker-LS  
			print('Start Training and Validation for NTF-Tucker-LS..............................')
			rRMSE_best=float('inf')
			for f,F in enumerate(F_list):
				observed_poisson_data=np.array(Y_train_wo_nan[k])
				ts1 = time.time()
				core, factors = non_negative_tucker(observed_poisson_data, F, n_iter_max=10, init='random', tol=0.0001, random_state=None, verbose=False)
				Y_full_pred = torch.tensor(tl.tucker_to_tensor((core,factors)))  
				Y_full_pred = Y_full_pred.contiguous()
				ts2 = time.time()
				time_s=ts2-ts1
				Y_val_pred = prediction(Y_full_pred,ind_val[k])
				e_metrics = evaluation(Y_val_pred,Y_val[k])
				rRMSE = e_metrics[0]
				print('#################################################################################')
				print('VALIDATION - NTF-Tucker-LS : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f},runtime = {:.4f}'.format(F,\
				e_metrics[0], e_metrics[1],e_metrics[2],e_metrics[3],time_s))
				print('#################################################################################')
				if rRMSE < rRMSE_best:
					NTF_Tucker_LS_best_param = [F]
					Y_sel = torch.tensor(Y_full_pred)
					rRMSE_best=rRMSE
					results_best = list(e_metrics)+[time_s] 
				result = ['val',i, k, F]+list(e_metrics)+[time_s]
				NTF_Tucker_LS_results.loc[r8] = result
				NTF_Tucker_LS_results.to_csv(result_dir+'NTF_Tucker_LS_results.csv', index=False)
				r8=r8+1
			print('Stop Training and Validation for NTF-Tucker-LS..............................')
			print('Start Testing for NTF-Tucker-LS..............................')
			Y_test_pred = prediction(Y_sel,ind_test[k])
			results_test = evaluation(Y_test_pred,Y_test[k]) 
			for j in range(4):
				metrics_NTF_Tucker_LS[j][i,k] = results_test[j]
			print('##############################Best Results#####################################')
			print('TEST - NTF-Tucker-LS : rank={}, rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(NTF_LS_best_param[0], \
			results_test[0], results_test[1],results_test[2],results_test[3]))
			print('##############################Best Results#######################################')
			result = ['best',i, k]+ NTF_Tucker_LS_best_param+results_best
			NTF_Tucker_LS_results.loc[r8] = result
			NTF_Tucker_LS_results.to_csv(result_dir+'NTF_Tucker_LS_results.csv', index=False)
			r8=r8+1
			
			result = ['test',i, k]+ NTF_Tucker_LS_best_param+list(results_test)+[0]
			NTF_Tucker_LS_results.loc[r8] = result
			NTF_Tucker_LS_results.to_csv(result_dir+'NTF_Tucker_LS_results.csv', index=False)
			r8=r8+1
#################################################################################################################################
#################################################################################################################################


if os.path.exists(result_dir+'final_results.csv'):
	os.remove(result_dir+'final_results.csv')
#############################################Final Results########################################################################
print('#################################################################################')
print('UNCLE_TC: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_UNCLE_TC[0]),\
			np.mean(metrics_UNCLE_TC[1]),np.mean(metrics_UNCLE_TC[2]),np.mean(metrics_UNCLE_TC[3])))
print('UNCLE_TC_linear: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_UNCLE_TC_linear[0]),\
			np.mean(metrics_UNCLE_TC_linear[1]),np.mean(metrics_UNCLE_TC_linear[2]),np.mean(metrics_UNCLE_TC_linear[3])))
print('NTF-KL:rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_KL[0]),\
			np.mean(metrics_NTF_KL[1]),np.mean(metrics_NTF_KL[2]),np.mean(metrics_NTF_KL[3])))
print('BPTF: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_BPTF[0]),\
			np.mean(metrics_BPTF[1]),np.mean(metrics_BPTF[2]),np.mean(metrics_BPTF[3])))
print('HaLRTC: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_HaLRTC[0]),\
			np.mean(metrics_HaLRTC[1]),np.mean(metrics_HaLRTC[2]),np.mean(metrics_HaLRTC[3])))
print('NTF-LS: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_LS[0]),\
			np.mean(metrics_NTF_LS[1]),np.mean(metrics_NTF_LS[2]),np.mean(metrics_NTF_LS[3])))
print('NTF-Tucker-LS: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_Tucker_LS[0]),\
			np.mean(metrics_NTF_Tucker_LS[1]),np.mean(metrics_NTF_Tucker_LS[2]),np.mean(metrics_NTF_Tucker_LS[3])))
print('#################################################################################') 
result = [['UNCLE_TC',np.mean(metrics_UNCLE_TC[0]),\
			np.mean(metrics_UNCLE_TC[1]),np.mean(metrics_UNCLE_TC[2]),np.mean(metrics_UNCLE_TC[3])],
		  ['UNCLE_TC_linear',np.mean(metrics_UNCLE_TC_linear[0]),np.mean(metrics_UNCLE_TC_linear[1]),\
		   np.mean(metrics_UNCLE_TC_linear[2]),np.mean(metrics_UNCLE_TC_linear[3])],
		  ['NTF-KL',np.mean(metrics_NTF_KL[0]),\
			np.mean(metrics_NTF_KL[1]),np.mean(metrics_NTF_KL[2]),np.mean(metrics_NTF_KL[3])],
		  ['BPTF',np.mean(metrics_BPTF[0]),\
			np.mean(metrics_BPTF[1]),np.mean(metrics_BPTF[2]),np.mean(metrics_BPTF[3])],
		  ['HaLRTC',np.mean(metrics_HaLRTC[0]),\
			np.mean(metrics_HaLRTC[1]),np.mean(metrics_HaLRTC[2]),np.mean(metrics_HaLRTC[3])],
		  ['NTF-LS',np.mean(metrics_NTF_LS[0]),\
			np.mean(metrics_NTF_LS[1]),np.mean(metrics_NTF_LS[2]),np.mean(metrics_NTF_LS[3])],
		  ['NTF-Tucker-LS',np.mean(metrics_NTF_Tucker_LS[0]),\
			np.mean(metrics_NTF_Tucker_LS[1]),np.mean(metrics_NTF_Tucker_LS[2]),np.mean(metrics_NTF_Tucker_LS[3])]	
		 ]
Final_results = pd.DataFrame(result, columns=['Algorithms','rRMSE','MAPE','F1 Score', 'Hamming Loss'])	  
Final_results.to_csv(result_dir+'final_results.csv', index=False)


##############################################Final Results########################################################################
#print('#################################################################################')
#print('UNCLE-TC: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_TensEconet[0]),\
#			np.mean(metrics_TensEconet[1]),np.mean(metrics_TensEconet[2]),np.mean(metrics_TensEconet[3])))
#print('TensEcoNet w/o hnet: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_TensEconet_wo_hnet[0]),\
#			np.mean(metrics_TensEconet_wo_hnet[1]),np.mean(metrics_TensEconet_wo_hnet[2]),np.mean(metrics_TensEconet_wo_hnet[3])))
#print('TensEcoNet linear: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_TensEconet_linear[0]),\
#			np.mean(metrics_TensEconet_linear[1]),np.mean(metrics_TensEconet_linear[2]),np.mean(metrics_TensEconet_linear[3])))
#print('NTF-KL:rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_KL[0]),\
#			np.mean(metrics_NTF_KL[1]),np.mean(metrics_NTF_KL[2]),np.mean(metrics_NTF_KL[3])))
#print('BPTF: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_BPTF[0]),\
#			np.mean(metrics_BPTF[1]),np.mean(metrics_BPTF[2]),np.mean(metrics_BPTF[3])))
#print('HaLRTC: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_HaLRTC[0]),\
#			np.mean(metrics_HaLRTC[1]),np.mean(metrics_HaLRTC[2]),np.mean(metrics_HaLRTC[3])))
#print('NTF-LS: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_LS[0]),\
#			np.mean(metrics_NTF_LS[1]),np.mean(metrics_NTF_LS[2]),np.mean(metrics_NTF_LS[3])))
#print('NTF-Tucker-LS: rRMSE={:.4f}, MAPE={:.4f},F1 Score={:.4f},Hamming Loss={:.4f}'.format(np.mean(metrics_NTF_Tucker_LS[0]),\
#			np.mean(metrics_NTF_Tucker_LS[1]),np.mean(metrics_NTF_Tucker_LS[2]),np.mean(metrics_NTF_Tucker_LS[3])))
#print('#################################################################################') 