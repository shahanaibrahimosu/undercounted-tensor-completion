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
class UNCLE_TC_GLOBAL:
	def __init__(self,input_parameters,network_parameters,simulation_paramaters,flags,init_parameters):
		self.I = network_parameters['I']
		self.K = network_parameters['K']
		self.F = network_parameters['F']
		self.D = network_parameters['D']
		self.flag_groundtruth = input_parameters['flag_groundtruth']
		self.observed_data = input_parameters['observed_data']
		self.obs_count_fraction = network_parameters['obs_count_fraction']
		self.obs_feature_fraction = network_parameters['obs_feature_fraction']
		
		self.no_of_BCD_iterations = simulation_paramaters['no_of_BCD_iterations']
		self.hidden_unit_g = simulation_paramaters['hidden_unit_g']
		self.hidden_layer_g = simulation_paramaters['hidden_layer_g']
		self.learning_rate_g = simulation_paramaters['learning_rate_g']
		self.batch_size = simulation_paramaters['batch_size']
		self.no_of_epochs_theta = simulation_paramaters['no_of_epochs_theta']
		self.no_iteration_MM = simulation_paramaters['no_iteration_MM']
		self.no_inner_iter_MM = simulation_paramaters['no_inner_iter_MM']	  
		self.tol = simulation_paramaters['tol']
		self.mu = simulation_paramaters['mu']
		self.flag_auto_mu_selection = simulation_paramaters['flag_auto_mu_selection']
		self.flag_normalized_cost = simulation_paramaters['flag_normalized_cost']
		self.loss_type=simulation_paramaters['loss_type']
		if not self.flag_groundtruth:
			self.ind_val = simulation_paramaters['ind_val']
			self.Y_val = simulation_paramaters['Y_val']
		
		self.flag_NN_detection = flags['flag_NN_detection']
		self.flag_NN_detection_linear = flags['flag_NN_detection_linear']
		self.flag_tensor_factorization=flags['flag_tensor_factorization']
		
		self.network_parameters=network_parameters
		self.simulation_paramaters=simulation_paramaters
		self.flags=flags
		
		self.init_parameters=init_parameters
		

	def train_UNCLE_TC(self):
		# Get observed data
		Y  = self.observed_data.Y.clone().detach()
		Z = self.observed_data.Z.clone().detach()
		Theta = self.observed_data.Theta.clone().detach()
		Omega = self.observed_data.Omega.clone().detach()
		ThetaminusOmega = self.observed_data.ThetaminusOmega.clone().detach()
		OmegaminusTheta = self.observed_data.OmegaminusTheta.clone().detach()
		if self.flag_groundtruth==1:
			A_g = self.observed_data.A_true.copy()
			P_g   = self.observed_data.P.clone().detach()
			P1_g = P_g.view(np.prod(self.I,dtype=np.int),1)
			Lambda_g   = self.observed_data.Lambda.clone().detach()
			M_g	  = self.observed_data.T.clone().detach()
		if self.flag_groundtruth==2:
			P_g   = self.observed_data.P.clone().detach()
			P1_g = P_g.view(np.prod(self.I,dtype=np.int),1)
		Y1 = Y.view(np.prod(self.I,dtype=np.int),1)
		
		# Get the linear indices corresponding to Theta 
		if self.flag_groundtruth==1:
			P_masked = self.observed_data.P_masked.clone().detach()
			P1 = P_masked.view(np.prod(self.I,dtype=np.int),1)
			P_isnannot = torch.logical_not(torch.isnan(P1.squeeze()))
			Theta_linear = torch.nonzero(P_isnannot)
			Theta_linear = Theta_linear.squeeze()
			Y_isnannot = torch.logical_not(torch.isnan(Y1.squeeze()))
			Omega_linear = torch.nonzero(Y_isnannot)
			Omega_linear = Omega_linear.squeeze()
		else:
			Theta_linear = self.observed_data.Theta_linear.clone().detach()
			Omega_linear = self.observed_data.Omega_linear.clone().detach()
			self.obs_count_fraction = Omega.size()[0]/np.prod(self.I)
			self.obs_feature_fraction = Theta.size()[0]/np.prod(self.I)
		

		

		
		# Train loader for updating theta
		train_loader_theta = torch.utils.data.DataLoader(Theta_linear,batch_size=self.batch_size,shuffle=False)
		num_observed_features = Theta.size()[0]
		print(num_observed_features)
		
		# Train loader for updating pi's
		if self.obs_count_fraction > self.obs_feature_fraction:
			estimation_set_for_P = Theta_linear
			train_loader_p = torch.utils.data.DataLoader(estimation_set_for_P,batch_size=self.batch_size,shuffle=False,drop_last=True)
			num_observed_data = Theta.size()[0]
		else:
			estimation_set_for_P = Omega_linear
			train_loader_p = torch.utils.data.DataLoader(Omega_linear,batch_size=self.batch_size,shuffle=False,drop_last=True)
			num_observed_data = Omega.size()[0]
		print(num_observed_data)
		
		if self.loss_type=='euclidean':
			loss_function = euclidean_loss
		else:
			loss_function = gen_KL_loss

		

		if self.init_parameters['flag_init']:
			M	  = self.init_parameters['M_init']
			A	  = self.init_parameters['A_init'].copy()
			Lambda = self.init_parameters['Lambda_init']
			Lambda1 = Lambda.view(np.prod(self.I,dtype=np.int),1)
			P	  = self.init_parameters['P_init']
			P1 = P.view(np.prod(self.I,dtype=np.int),1) 
			if self.flag_NN_detection_linear:
				GTHETA	  = self.init_parameters['GTHETA_linear']
				GTHETA1 = GTHETA.view(np.prod(self.I,dtype=np.int),1)
			else:
				GTHETA	  = self.init_parameters['GTHETA']
				GTHETA1 = GTHETA.view(np.prod(self.I,dtype=np.int),1)
		else:	   
			# Algorithm Initialization
			data_initialization=Initialization(self.network_parameters,self.simulation_paramaters,self.flags,Z)
			data_initialization.initialize_TF() 
			M	  = data_initialization.T.clone().detach()
			A	  = data_initialization.A1.copy()
			Lambda = data_initialization.Lambda.clone().detach()
			Lambda1 = Lambda.view(np.prod(self.I,dtype=np.int),1)
			P	  = data_initialization.P.clone().detach()
			P1 = P.view(np.prod(self.I,dtype=np.int),1)
			if self.flag_NN_detection_linear:
				GTHETA	  = data_initialization.GTHETA_linear.clone().detach()
				GTHETA1 = GTHETA.view(np.prod(self.I,dtype=np.int),1)
			else:
				GTHETA	  = data_initialization.GTHETA.clone().detach()
				GTHETA1 = GTHETA.view(np.prod(self.I,dtype=np.int),1)
 
		if self.flag_NN_detection_linear:
			model_g = NN_detetcion_linear(self.D)
			print("using linear network")
		else:
			model_g = NN_detetcion(self.D,self.hidden_unit_g,self.hidden_layer_g)
		model_p = probability_model(np.prod(self.I,dtype=np.int))
		# Optimizers
		optimizer_g = torch.optim.Adam(model_g.parameters(), lr=self.learning_rate_g, weight_decay=0)
		optimizer_p = torch.optim.Adam(model_p.parameters(), lr=self.learning_rate_g, weight_decay=0)


		#Initialize output arrays
		cost= [0]*self.no_of_BCD_iterations
		U_mse= [0]*self.no_of_BCD_iterations
		p_mre= [0]*self.no_of_BCD_iterations
		lambda_mre= [0]*self.no_of_BCD_iterations
		timestamps = [0]*self.no_of_BCD_iterations
		rRMSE = [0]*self.no_of_BCD_iterations
		


		# Calculate the metrics 
		Y_pred = Lambda*P
		Y_pred_prev=Y_pred
		P_sel_prev = P.clone().detach().numpy()
		Lambda_sel_prev = Lambda.clone().detach().numpy()
		P_sel_best = P.clone().detach().numpy()
		Lambda_sel_best = Lambda.clone().detach().numpy()
		Y_pred_best = Y_pred
		if self.flag_groundtruth==0:
			Y_val_pred = prediction(Y_pred,self.ind_val)
			rRMSE[0] = get_rRMSE(Y_val_pred,self.Y_val) 
			rRMSE_best=rRMSE[0]
			
		
		initial_cost1=get_totalcost(M,P,Y,Omega)
		initial_cost2=loss_function(P1[Theta_linear,0],GTHETA1[Theta_linear,0])/Theta_linear.size()[0]
		value =2*initial_cost1/initial_cost2
		power = int(math.log10(value))
		if self.flag_auto_mu_selection:
			self.mu = (int(value/10**power))*10**power+10**power
		print('Selected mu value = '+str(self.mu))
		cost[0] = get_totalcost(M,P,Y,Omega)+(self.mu/2)*loss_function(P1[Theta_linear,0],GTHETA1[Theta_linear,0])/Theta_linear.size()[0]
		#print('P1[Theta_linear,0]')
		#print(P1[Theta_linear,0])
		#print('GTHETA1[Theta_linear,0]')
		#print(GTHETA1[Theta_linear,0])
		if self.flag_groundtruth==1:
			U_mse[0] = 0#getMSE(A,A_g)
			p_mre[0] =  getMSE_entry(P1[estimation_set_for_P,0],P1_g[estimation_set_for_P,0])# torch.norm(P-P_g,p=1)/(I*J*K)
			lambda_mre[0] = getMSE_entry(Lambda,Lambda_g)#torch.norm(Lambda-Lambda_g,p=1)/(I*J*K)
			print('#################################################################################')
			print('BCD iter [{}/{}], cost:{:.4f}, MSE of U:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0], U_mse[0]))
			print('BCD iter [{}/{}], MRE of P:{:.4f}, MRE of Lambda:{:.4f}'.format(0, self.no_of_BCD_iterations, p_mre[0], lambda_mre[0]))
			print('#################################################################################')
		elif self.flag_groundtruth==0:		   
			print('#################################################################################')
			print('BCD iter [{}/{}], cost:{:.4f}, rRMSE_val:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0],rRMSE[0]))
			print('#################################################################################')
		else:
			p_mre[0] =  getMSE_entry(P,P_g)# torch.norm(P-P_g,p=1)/(I*J*K)
			print('#################################################################################')
			print('BCD iter [{}/{}], cost:{:.4f}, MRE of P:{:.4f}'.format(0, self.no_of_BCD_iterations, cost[0],p_mre[0]))
			print('#################################################################################')   
			
			
		
			
		#Start training for U,V,W...................
		ts1 = time.time()
		ts2 = time.time()
		timestamps[0] = ts2-ts1
		for k in range(self.K):
			A[k]=torch.tensor(A[k],dtype=torch.float)
		for iter in range(self.no_of_BCD_iterations-1):			
			#Start training for P................... 
			P_hat = GTHETA1[estimation_set_for_P,0].clone().detach()
			P_hat = P_hat.float()
			if self.flag_normalized_cost==1:
				Lambda_mod = Lambda1[estimation_set_for_P,0]/Omega.size()[0]
				Y_mod = Y1[estimation_set_for_P,0]/Omega.size()[0]
				mu_mod = self.mu/Theta.size()[0]
			else:
				Lambda_mod = Lambda1[estimation_set_for_P,0]
				Y_mod = Y1[estimation_set_for_P,0]
				mu_mod = self.mu
			if self.loss_type=='euclidean':
				#mu_temp=(self.mu)*P_hat-Lambda1[estimation_set_for_P,0]
				#val = (mu_temp+torch.sqrt(torch.square(mu_temp)+4*self.mu*Y1[estimation_set_for_P,0]))/(2*self.mu)
				#P1[estimation_set_for_P,0] = val.float()
				mu_temp=(mu_mod)*P_hat-Lambda_mod
				val = (mu_temp+torch.sqrt(torch.square(mu_temp)+4*mu_mod*Y_mod))/(2*mu_mod)
				P1[estimation_set_for_P,0] = val.float()
			else:
				#val = torch.div(Y1[estimation_set_for_P,0]+(self.mu/2)*P_hat,(self.mu/2)+Lambda1[estimation_set_for_P,0])
				val = torch.div(Y_mod+(mu_mod/2)*P_hat,(mu_mod/2)+Lambda_mod)
				P1[estimation_set_for_P,0] = val.float()
			P1[estimation_set_for_P,0] = torch.clamp(P1[estimation_set_for_P,0],min=1e-12,max=1)
			P1=P1.view(-1,1)
			P = P1.view(self.I)
			
			if self.obs_count_fraction > self.obs_feature_fraction:
				# This part will update the p values which are in Omega which are not in Theta
				P1[OmegaminusTheta,0]=torch.clamp(torch.div(Y1[OmegaminusTheta,0],Lambda1[OmegaminusTheta,0]),min=1e-12,max=(1-1e-12)).float()
				P1=P1.view(-1,1)
				#print(P1)
				P = P1.view(self.I) 
				
			if self.obs_count_fraction < self.obs_feature_fraction:
				P1[ThetaminusOmega,0] = GTHETA1[ThetaminusOmega,0].clone().detach()
				P1[ThetaminusOmega,0]=P1[ThetaminusOmega,0].float()
				P = P1.view(self.I)
		
			ts1 = time.time()   
			if self.flag_tensor_factorization:
				loss_prev=0
				eps=1e-12
				for epoch in range(self.no_iteration_MM): 
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
									V_hat[p] = Y[sparse_indices[p,0],sparse_indices[p,1],sparse_indices[p,2]]/torch.clamp(torch.dot(W[p,:],A[k][sparse_indices[p,k],:]),min=1e-6)
									Phi[jj,:] += V_hat[p]*W[p,:]
									Psi[jj,:] +=P[sparse_indices[p,0],sparse_indices[p,1],sparse_indices[p,2]]*W[p,:]						   
								A[k][jj,:] = torch.div(A[k][jj,:]*Phi[jj,:],torch.clamp(Psi[jj,:],min=1e-6))
						
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
					

				
			if self.flag_NN_detection: 
												 
				#Start training for theta...................
				model_g.train()
				loss_g_prev=0
				for epoch in range(self.no_of_epochs_theta):
					loss_g=0
					for i, indices_data in enumerate(train_loader_theta,0):						
						optimizer_g.zero_grad()
						g_theta = model_g.forward(Z[indices_data,:])
						g_theta = g_theta.view(-1)
						loss = loss_function(P1[indices_data,0],g_theta)#torch.sum((P1[indices_data,0]-g_theta)**2)
						loss.backward(retain_graph=True)
						loss_g=loss_g+loss
						optimizer_g.step()
					loss_g = loss_g/num_observed_features
					#print('epoch [{}/{}], theta loss function:{:.4f}'.format(epoch + 1, self.no_of_epochs_theta, loss_g))
					if (abs(loss_g-loss_g_prev)/abs(loss_g_prev)) < self.tol:
						break
					loss_g_prev = loss_g
				ts2 = time.time()
				

					
				model_g.eval()
				with torch.no_grad():
					for i, indices_data in enumerate(train_loader_theta,0): 
						g_theta = model_g.forward(Z[indices_data,:])
						GTHETA1[indices_data,0]=g_theta.view(-1)
					GTHETA1=GTHETA1.view(-1,1)
					#print(P1)
					GTHETA = GTHETA1.view(self.I)

					
						
			else:
				P1 = P.view(np.prod(self.I,dtype=np.int),1)
					

				
			
				
				

			# Computing total loss and MSE
			timestamps[iter+1] = ts2-ts1
			
		
			
			#print(A[0])
			#print(A_g[0])
			Y_pred = Lambda*P
			if self.flag_groundtruth==0:
				Y_val_pred = prediction(Y_pred,self.ind_val)
				rRMSE[iter+1] = get_rRMSE(Y_val_pred,self.Y_val) 
			with torch.no_grad(): 
				g_theta = model_g.forward(Z[Theta_linear,:])	
			#print(get_totalcost(M,P,Y,Omega))
			#print(loss_function(P1[Theta_linear,0],g_theta)/Theta_linear.size()[0])
			cost[iter+1]=get_totalcost(M,P,Y,Omega)+(self.mu/2)*loss_function(P1[Theta_linear,0],g_theta)/Theta_linear.size()[0]
			if self.flag_groundtruth==1:
				if self.flag_tensor_factorization:
					U_mse[iter+1] = 0#getMSE(A,A_g)
				else:
					U_mse[iter+1] =0
				#print(P1)
				#print(P1_g)
				p_mre[iter+1] = getMSE_entry(P1[estimation_set_for_P,0],P1_g[estimation_set_for_P,0]) #torch.norm(P-P_g,p=1)/(I*J*K)
				lambda_mre[iter+1] = getMSE_entry(Lambda,Lambda_g) #torch.norm(Lambda-Lambda_g,p=1)/(I*J*K)
				print('#################################################################################')
				print('BCD iter [{}/{}], total loss function:{:.4f}, MSE of U:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1], U_mse[iter+1]))
				print('BCD iter [{}/{}], MRE of P:{:.4f}, MRE of Lambda:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, p_mre[iter+1], lambda_mre[iter+1]))
				print('#################################################################################')
			elif self.flag_groundtruth==0:			  
				print('#################################################################################')
				print('BCD iter [{}/{}], cost:{:.4f}, rRMSE_val:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1],\
																			   rRMSE[iter+1]))
				print('#################################################################################')
			else:  
				p_mre[iter+1] = getMSE_entry(P,P_g) #torch.norm(P-P_g,p=1)/(I*J*K)
				print('#################################################################################')
				print('BCD iter [{}/{}], cost:{:.4f}, MRE of P:{:.4f}'.format(iter + 1, self.no_of_BCD_iterations, cost[iter+1],\
																			   p_mre[iter+1]))
				print('#################################################################################')

 
			
			if self.flag_groundtruth==0:
				stop_condition = iter > 5 and rRMSE[iter+1]>rRMSE[iter]
				#stop_condition = iter > 1 and abs(cost[iter]-cost[iter+1])/abs(cost[iter]) < self.tol
				if rRMSE[iter+1] < rRMSE_best:
					Y_pred_best = Y_pred
					P_sel_best = P.clone().detach().numpy()
					Lambda_sel_best = Lambda.clone().detach().numpy()
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
					p_mre[iter+2:]  =[float("nan")]*(self.no_of_BCD_iterations-iter-2)
				if self.flag_groundtruth==2:
					p_mre[iter+2:]  =[float("nan")]*(self.no_of_BCD_iterations-iter-2)				
				break

			Y_pred_prev = Y_pred
		
		
		if self.flag_groundtruth==1:
			scaling_fact_p = torch.div(P_g,P)
			scaling_fact_p = scaling_fact_p.view(1,torch.numel(P))
			scaling_fact_lambda = torch.div(Lambda_g,Lambda)
			scaling_fact_lambda = scaling_fact_lambda.view(1,torch.numel(P))				
			output_parameters = {'cost':cost,
								 'U_mse':U_mse,
								 'lambda_mre':lambda_mre,
								 'p_mre':p_mre,
								 'timestamps':timestamps,
								 'scaling_fact_p':scaling_fact_p,
								 'scaling_fact_lambda':scaling_fact_lambda,
								 'Y_pred':Y_pred
			}
		else: 
			output_parameters = {'cost':cost,
								 'timestamps':timestamps,
								 'p_mre':p_mre,
								 'Y_pred':Y_pred_best.clone().detach(),
								 'P':P_sel_best,
								 'Lambda':Lambda_sel_best,
								 'A':A.copy(),
			} 
			
		A =[]
				
		return output_parameters
	

def initialize_p(m,val):
		m.w.data = val
		
def euclidean_loss(x,y):
	z = torch.sum((x-y)**2)
	return z
	
def gen_KL_loss(x,y):
	z = torch.sum(x*torch.log(torch.div(x,y))-x+y)
	return z

