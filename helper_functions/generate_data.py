import torch
import numpy as np
import tensorly as tl
import math

import import_ipynb
from helper_functions.models import *

class GroundTruth:
	def __init__(self,generate_options):
		# Problem Parameters
		self.I = generate_options['I'] # Size of the tensor
		self.K = generate_options['K']  # Dimension of the tensor
		self.F = generate_options['F_true']   # Rank of the tensor
		self.D = generate_options['D']	# dimension of the feature vector
		self.gamma = generate_options['gamma']
		self.obs_count_fraction = generate_options['obs_count_fraction']
		self.obs_feature_fraction = generate_options['obs_feature_fraction']
		
		self.obs_feature_equal_fraction = generate_options['obs_feature_equal_fraction']
		self.obs_feature_equal_noise_dB = generate_options['obs_feature_equal_noise_dB']
		self.g_function_type = generate_options['g_function_type']
		

		# Output options
		self.Y = None
		self.Z = None
		self.A_true = []
		self.P =None
		self.Lambda=None
		self.T = None
		self.Omega=None
		self.Theta=None
		self.OmegaminusTheta=torch.empty(1)
		self.ThetaminusOmega=torch.empty(1)


	def generate_observed_counts_Nonlinear_TF(self):
	
		while(1):
			# Generate tensor factor matrices 
			for k in range(self.K):
				U = self.gamma*np.random.rand(self.I[k],self.F) # 
				self.A_true.append(U);

			# Generate side features 
			self.Z = torch.randn(np.prod(self.I,dtype=np.int),self.D)  

			# Generate Lambda tensor
			weights = None
			self.T = torch.tensor(tl.cp_to_tensor([weights,self.A_true]))
			self.Lambda = self.T

			
			# Generate True obsevations N
			N = torch.poisson(self.Lambda)
			   

			
			# Generate mask for Y; Only a fraction of Y's are observed\, i.e, y_i where i \in Omega 
			if self.obs_count_fraction==self.obs_feature_fraction:
				mask_Z = torch.bernoulli(torch.tensor(self.obs_feature_fraction*torch.ones(self.I)))
				index_unobserved_P = torch.nonzero(mask_Z==0)
				index_list_unobserved_P=[]
				for k in range(self.K):
					index_list_unobserved_P.append(index_unobserved_P[:,k])
				mask_Z[index_list_unobserved_P]=torch.tensor(float('NaN'))
				mask_Y = mask_Z
				self.Theta = torch.nonzero(mask_Z==1)
				self.Omega = torch.nonzero(mask_Y==1)
				
			elif self.obs_count_fraction<self.obs_feature_fraction:
				mask_Z = torch.bernoulli(torch.tensor(self.obs_feature_fraction*torch.ones(self.I)))
				index_unobserved_P = torch.nonzero(mask_Z==0)
				index_list_unobserved_P=[]
				for k in range(self.K):
					index_list_unobserved_P.append(index_unobserved_P[:,k])
				mask_Z[index_list_unobserved_P]=torch.tensor(float('NaN'))
				diff_fraction = self.obs_feature_fraction-self.obs_count_fraction
				mask_Y=mask_Z.clone()
				self.Theta = torch.nonzero(mask_Z==1)
				index = torch.nonzero(mask_Y==1)
				num_to_nan = int(diff_fraction*np.prod(self.I,dtype=np.int))
				nan_indices = torch.randint(0,index.size()[0],(num_to_nan,))
				index_list_omega=[]
				for k in range(self.K):
					list_nan = index[nan_indices,k] 
					index_list_omega.append(list_nan)
				mask_Y[index_list_omega]=torch.tensor(float('NaN'))
				self.Omega = torch.nonzero(mask_Y==1)
				
				
				#Get Theta-Omega set
				mask_Z_copy=mask_Z.clone()
				mask_Y_copy=mask_Y.clone()
				mask_Y_copy[torch.isnan(mask_Y_copy)]=0
				mask_Z_copy[torch.isnan(mask_Z_copy)]=0
				mask_Z_minus_Y = mask_Z_copy-mask_Y_copy
				mask_Z_minus_Y=mask_Z_minus_Y.view(np.prod(self.I,dtype=np.int),1)
				self.ThetaminusOmega = torch.nonzero(mask_Z_minus_Y==1)
				
			else:
				# Generate mask for features i.e, z_i where i \in Theta
				mask_Y = torch.bernoulli(torch.tensor(self.obs_feature_fraction*torch.ones(self.I)))
				index = torch.nonzero(mask_Y==0)
				index_list=[]
				for k in range(self.K):
					index_list.append(index[:,k])
				mask_Y[index_list]=torch.tensor(float('NaN'))
				self.Omega = torch.nonzero(mask_Y==1)
				
				diff_fraction = self.obs_count_fraction-self.obs_feature_fraction
				mask_Z=mask_Y.clone()
				index = torch.nonzero(mask_Z==1)
				num_to_nan = int(diff_fraction*np.prod(self.I,dtype=np.int))
				nan_indices = torch.randint(0,index.size()[0],(num_to_nan,))
				index_list_theta=[]
				for k in range(self.K):
					list_nan = index[nan_indices,k] 
					index_list_theta.append(list_nan)
				mask_Z[index_list_theta]=torch.tensor(float('NaN'))
				self.Theta = torch.nonzero(mask_Z==1)
				
				
				
				#Get Omega-Theta set
				mask_Z_copy=mask_Z.clone()
				mask_Y_copy=mask_Y.clone()
				mask_Y_copy[torch.isnan(mask_Y_copy)]=0
				mask_Z_copy[torch.isnan(mask_Z_copy)]=0
				mask_Y_minus_Z = mask_Y_copy-mask_Z_copy
				mask_Y_minus_Z=mask_Y_minus_Z.view(np.prod(self.I,dtype=np.int),1)
				self.OmegaminusTheta = torch.nonzero(mask_Y_minus_Z==1)
				
			
				
				
			
			
			# Selecting linear indices for equal features, i.e, z_i, i \in Theta
			if self.obs_feature_equal_fraction!=0:
				mask_Z1 = mask_Z.view(np.prod(self.I,dtype=np.int),1)
				index_observed_Z = torch.nonzero(mask_Z1==1)
				perm_indices = torch.randperm(index_observed_Z.size(0))
				len_theta = int(self.obs_feature_equal_fraction*np.prod(self.I,dtype=np.int))
				if len_theta > index_observed_Z.size(0):
					len_theta=index_observed_Z.size(0)
				sel_indices = perm_indices[0:len_theta]
				index_observed_Z=index_observed_Z.squeeze()
				index_equal_Z=index_observed_Z[sel_indices,0]
				val_Z_equal = torch.randn(1,self.D) 
				if math.isinf(self.obs_feature_equal_noise_dB):
					sigma=0
				else:
					sigma = torch.sqrt((torch.norm(val_Z_equal)**2)/(self.D*(10**(self.obs_feature_equal_noise_dB/10))))
				noise_vectors = torch.randn(len_theta,self.D) 
				self.Z[index_equal_Z,:]=val_Z_equal+ noise_vectors*sigma

			# Generate detection probability g(z) = <alpha,tanh(z)>
			alpha = torch.rand(self.D,1)
			if self.g_function_type=='tanh':
				G_THETA_m  = torch.sigmoid(torch.matmul((torch.tanh(self.Z))**3,alpha))
				#G_THETA_m = torch.sigmoid(torch.matmul(0.1*(torch.log(self.Z**2))+0.1*self.Z**2,alpha))
			elif self.g_function_type=='log':
				G_THETA_m = torch.sigmoid(torch.matmul(0.1*(torch.log(self.Z**2))+0.1*self.Z**2,alpha))
				#G_THETA_m = torch.sigmoid(torch.matmul(3*torch.log(self.Z)+0.2*self.Z,alpha))
			elif self.g_function_type=='cube':
				G_THETA_m = torch.sigmoid(torch.matmul(0.5*self.Z**3+0.2*self.Z,alpha))
			elif self.g_function_type=='tanh-log':
				G_THETA_m = torch.sigmoid(torch.einsum('ij,ij->i',(torch.tanh(self.Z))**3,torch.log(self.Z)))
			elif self.g_function_type=='gaussian':
				G_THETA_m = torch.sigmoid(0.3*torch.exp(-0.5*(torch.einsum('ij,ij->i',self.Z,self.Z)))+0.7*torch.exp(-0.5*(torch.einsum('ij,ij->i',self.Z-1,self.Z-1))))
			else:
				None
			#tt = torch.max(G_THETA_m)
			#G_THETA_m = G_THETA_m/tt
			G_THETA  = G_THETA_m.view(self.I)
			self.P = G_THETA
			self.P_masked = self.P*mask_Z
			
		 

			# Generate observations
			M = torch.distributions.binomial.Binomial(N,self.P)
			self.Y = M.sample()
			self.Y = self.Y*mask_Y
			
			# Get the linear indices corresponding to unobserved features and place NaN for corresponding features
			mask_Z1 = mask_Z.view(np.prod(self.I,dtype=np.int),1)
			index_unobserved_Z = torch.isnan(mask_Z1)
			index_unobserved_Z=index_unobserved_Z.squeeze()
			self.Z[index_unobserved_Z,:]=torch.tensor(float('NaN'))
			
			
			P1 = self.P_masked.view(np.prod(self.I,dtype=np.int),1)
			P_isnannot = torch.logical_not(torch.isnan(P1.squeeze()))
			Theta_linear = torch.nonzero(P_isnannot)
			Theta_linear = Theta_linear.squeeze()
			
			
			if(torch.mean(P1[Theta_linear,0])>=0.09 and torch.mean(P1[Theta_linear,0])<=0.98):
				break
			else:
				self.Y = None
				self.Z = None
				self.A_true = []
				self.P =None
				self.Lambda=None
				self.T = None
				self.Omega=None
				self.Theta=None
				self.OmegaminusTheta=torch.empty(1)
				self.ThetaminusOmega=torch.empty(1)
			
			
	 
			
		
		
		# Only a fraction of Y's are observed\, i.e, y_i where i \in Omega 

		#mask_Y = torch.bernoulli(torch.tensor(self.obs_count_fraction*torch.ones(self.I)))
		#index = torch.nonzero(mask_Y==0)
		#index_list=[]
		#for k in range(self.K):
		#	index_list.append(index[:,k])
		#mask_Y[index_list]=torch.tensor(float('NaN'))
		#self.Y = self.Y*mask_Y
		#
		#
		## Get the tensor indices of observed interactions
		#Yt_isnannot = torch.logical_not(torch.isnan(Y))
		#indices_tensor_obs = torch.nonzero(Yt_isnannot)
		#len_observed = indices_tensor_obs.size()
		#print(len_observed)
	   
		
	   
		
		
		
		
class Initialization:
	def __init__(self,generate_options,simulation_paramaters,flags,Z):
		# Get the Parameters
		# Problem Parameters
		self.I = generate_options['I'] # Size of the tensor
		self.K = generate_options['K']   # Dimension of the tensor
		self.F = generate_options['F']   # Rank of the tensor
		self.D = generate_options['D']	# dimension of the feature vector
		self.gamma = generate_options['gamma']

		# Network Paramaters
		self.hidden_unit_g = simulation_paramaters['hidden_unit_g']
		self.hidden_layer_g = simulation_paramaters['hidden_layer_g']
		self.flag_NN_detection_linear = flags['flag_NN_detection_linear']


		# Output options
		self.Z = Z
		self.A1 = []
		self.P =None
		self.P_masked =None
		self.Lambda=None
		self.T = None
		self.GTHETA=None
		self.GTHETA_linear=None


	def initialize_TF(self):
		
		# Generate tensor factor matrices 
		for k in range(self.K):
			U = np.random.rand(self.I[k],self.F) # Plant embeddings
			self.A1.append(U);


		# Generate Lambda tensor
		weights = None
		self.T = torch.tensor(tl.cp_to_tensor([weights,self.A1]))
		self.Lambda = self.T




		# Generate detection probbaility and observed counts
		model_g_theta_linear = NN_detetcion_linear(self.D)
		model_g_theta = NN_detetcion(self.D,self.hidden_unit_g,self.hidden_layer_g)
		# G_THETA = torch.zeros((self.I,self.J,self.K))
		with torch.no_grad():
			G_THETA_m  = model_g_theta.forward(self.Z)
			G_THETA  = G_THETA_m.view(self.I)
			G_THETA_m_l  = model_g_theta_linear.forward(self.Z)
			G_THETA_l  = G_THETA_m_l.view(self.I)
		self.GTHETA = G_THETA
		self.GTHETA_linear = G_THETA_l
		
		self.P = torch.rand(self.I)



