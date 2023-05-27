import torch
from torch import nn
from torch.nn import functional as Func
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
class PPI_Dataset:
	def __init__(self,network_parameters):
		self.obs_file ="dataset/PPI/obs.txt"
		self.feature_file_lam="dataset/PPI/lam_fea.txt"
		self.feature_file_det="dataset/PPI/det_fea.txt"
		self.I_orig = 69
		self.J_orig = 215
		self.K_orig = 37

		self.I_all = network_parameters['I']
		self.I = self.I_all[0]
		self.J = self.I_all[1]
		self.K = self.I_all[2]# No of plants
		self.R2 = network_parameters['D']	# dimension of the feature vector
		self.Y= torch.zeros(self.I,self.J,self.K)
		self.Z2 = torch.zeros(self.I*self.J*self.K,self.R2)
		self.Y_no_nan = torch.zeros(self.I,self.J,self.K)
		
		self.Omega=None
		self.Theta=None
		self.Z = None
		self.OmegaminusTheta=torch.empty(1)
		self.ThetaminusOmega=torch.empty(1)
		self.obs_count_fraction=None
		self.obs_feature_fraction=None
		self.Theta_linear=None
		self.Omega_linear=None
		self.I_selected_indices = None
		self.J_selected_indices = None
		self.K_selected_indices = None


	def load_data(self,data_threshold):
		# Get Y tensor
		with open("%s" % (self.obs_file), "r") as f:
			data = f.read().strip().split("\n")
			data = [i.split() for i in data]
			data = np.asarray(data,dtype=np.long)
		data = torch.tensor(data,dtype=torch.long)
		if data_threshold>0:
			ind_selected  = torch.nonzero(data[:,3] <= data_threshold)
		else:
			ind_selected = torch.nonzero(torch.logical_not(torch.isnan(data[:,3])))
		data = data[ind_selected[:,0],:]
		Y_all = torch.ones(self.I_orig,self.J_orig,self.K_orig)*torch.tensor(float('NaN'))
		Y_all[data[:,1]-1,data[:,0]-1,data[:,2]-1] = data[:,3].float()
		pollinator_sum=torch.nansum(Y_all,dim=[0,2])
		pollinator_sum_sorted, pollinator_sum_indices = torch.sort(pollinator_sum,descending=True)
		#pollinator_sum_indices = torch.tensor(pollinator_sum_indices,dtype=torch.long)
		Y_mid  = Y_all[:,pollinator_sum_indices[0:self.J],:]
		plant_sum=torch.nansum(Y_mid,dim=[1,2])
		plant_sum_sorted, plant_sum_indices = torch.sort(plant_sum,descending=True)
		self.Y = Y_mid[plant_sum_indices[0:self.I],:,:]
		#dates_sum = torch.nansum(Y_mid1,dim=[0,1])
		#dates_sum_sorted, dates_sum_indices = torch.sort(dates_sum,descending=True)
		#self.Y = Y_mid1[:,:,dates_sum_indices[0:self.K]]
		#self.Y[self.Y==0]=torch.tensor(float('NaN'))
		#indices_zero=torch.nonzero(self.Y==0)
		#print(indices_zero.size()[0]/torch.numel(self.Y))
		#index_sel = torch.randint(indices_zero.size()[0],(12988,))
		#index_sel_Y=[]
		#for k in range(3):
		#	index_sel_Y.append(indices_zero[index_sel,k])
		#self.Y[index_sel_Y]=torch.tensor(float('NaN'))
		#
		self.I_selected_indices = plant_sum_indices[0:self.I]
		self.J_selected_indices = pollinator_sum_indices[0:self.J]
		self.K_selected_indices = torch.arange(self.K)

		indices_zero=torch.nonzero(self.Y==0)
		print(indices_zero.size()[0]/torch.numel(self.Y))
		nonzero_per = torch.sum(torch.logical_or(self.Y == 0,torch.isnan(self.Y)))*100.0/torch.numel(self.Y)
		zero_per = torch.sum(self.Y == 0)*100.0/torch.numel(self.Y)
		print('#################################################################################')
		print(' Nonzero entries in Y (%):{:.4f}'.format(100-nonzero_per))
		print('#################################################################################')
		print('#################################################################################')
		print(' zero entries in Y (%):{:.4f}'.format(zero_per))
		print('#################################################################################')
		
		
		# get feature file data for detection prob. p
		with open("%s" % (self.feature_file_det), "r") as f:
			data2 = f.read().strip().split("\n")
			data2 = [i.split() for i in data2]
		data2= np.asarray(data2,dtype=np.float32)
		Z2_indices = np.asarray(data2[ind_selected[:,0],0:3],dtype=np.long)-1
		Z2_data	= data2[ind_selected[:,0],3:3+self.R2]#data[:,3+self.R1:3+self.R1+self.R2]
		Z2_t =torch.zeros(self.I,self.J,self.K,self.R2)



		# Assign Z2
		for count,indices in enumerate(Z2_indices):
			i=indices[0]
			j=indices[1]
			k=indices[2]
			if torch.sum(i==plant_sum_indices[0:self.I])!=0 and torch.sum(j==pollinator_sum_indices[0:self.J])!=0:
				i_index =torch.nonzero(i==plant_sum_indices[0:self.I])
				j_index = torch.nonzero(j==pollinator_sum_indices[0:self.J])
				#Normalize the data
				data_Z2 = torch.tensor(Z2_data[count,:])
				#print('************')
				#print(data_Z2)
				#data_Z2 = (data_Z2-torch.mean(data_Z2))/(torch.std(data_Z2))#(data_Z2-torch.min(data_Z2))/(torch.max(data_Z2)-torch.min(data_Z2))
				#print(data_Z2)
				Z2_t[i_index,j_index,k,:]=data_Z2 #torch.tensor(Z2_data[count,:])#
		self.Z2=Z2_t.view(self.I*self.J*self.K,self.R2)

		#sum_Z2 = torch.sum(self.Z2,0)
		#normalize the features
		#for r in range(self.R1):
		#	self.Z1[:,r] = (self.Z1[:,r]-torch.min(self.Z1[:,r])) /(torch.max(self.Z1[:,r])  -torch.min(self.Z1[:,r]))
		for r in range(self.R2):
			self.Z2[:,r]  = (self.Z2[:,r]-torch.min(self.Z2[:,r])) /(torch.max(self.Z2[:,r])  -torch.min(self.Z2[:,r]))
		
		sum_Z2 = torch.sum(self.Z2,1)
		#print(sum_Z2.size())
		index=torch.nonzero(sum_Z2)
		#print(index)
		#print(self.Z2[index[0:10],:])
		
		#self.Omega = torch.nonzero(torch.logical_not(torch.isnan(self.Y)))
		#print(self.Omega)
		#print(self.Omega.size())
		#self.Theta = self.Omega
		#index_list=[]
		#for k in range(3):
		#	index_list.append(self.Omega[:,k])
		#Y11 = self.Y[index_list]
		#print(Y11)
		#print(torch.isnan(Y11))
		#print(torch.nonzero(Y11))			   
		#Y1 = self.Y.view(self.I*self.J*self.K,1)
		#self.Omega_linear = torch.nonzero(torch.logical_not(torch.isnan(Y1.squeeze())))
		#self.Omega_linear = self.Omega_linear.squeeze()
		
		#self.Theta_linear=torch.nonzero(torch.logical_not(torch.isnan(sum_Z2)))#self.Omega_linear
		#self.Theta = self.Theta_linear
		
		self.Z = self.Z2.clone().detach()
		#print(self.Z[index[0:10],:])


	def partition_training_testing_data(self,val_frac,test_frac):
		Y1 = self.Y.view(1,self.I*self.J*self.K)
		Y_isnannot = torch.logical_not(torch.isnan(Y1.squeeze()))
		indices = torch.nonzero(Y_isnannot)
		indices = indices.squeeze()
		ind_train, ind_val_test = train_test_split(indices, test_size=(test_frac+val_frac),shuffle=True)
		ind_val, ind_test = train_test_split(ind_val_test, test_size=test_frac/(test_frac+val_frac), shuffle=True)
		Y_test = Y1[0,ind_test]
		Y_val = Y1[0,ind_val]
		Y1[0,ind_val_test] =torch.tensor(float('NaN'))
		Y1_no_nan = Y1
		Y1_isnan = torch.isnan(Y1)
		indices_tensor = torch.nonzero(Y1_isnan)
		Y1_no_nan[indices_tensor[:,0],indices_tensor[:,1]]=0
		self.Y =Y1.view(self.I,self.J,self.K) #training data
		self.Y_no_nan=Y1_no_nan.view(self.I,self.J,self.K)
		return(Y_val, ind_val , Y_test, ind_test)

	def partition_k_folds(self,fold_num):
		Y1 = self.Y.view(1,self.I*self.J*self.K)
		Y_isnannot = torch.logical_not(torch.isnan(Y1.squeeze()))
		indices = torch.nonzero(Y_isnannot)
		indices = indices.squeeze()   
		kf = KFold(n_splits=fold_num, shuffle=True)
		Y_train_w_nan=[0]*fold_num
		Y_train_wo_nan=[0]*fold_num
		Omega_Y_train = [0]*fold_num
		Omega_linear_Y_train = [0]*fold_num
		obs_count_fraction=[0]*fold_num
		obs_feature_fraction=[0]*fold_num
		Y_val =[0]*fold_num
		Y_test =[0]*fold_num
		ind_val=[0]*fold_num
		ind_test=[0]*fold_num
		k=0
		for train_index, val_index in kf.split(indices):
			Y1_i=Y1.clone().detach()
			# Get validation indices
			ind_val_i=indices[val_index]#torch.index_select(indices, 0, torch.tensor(test_index))
			Y_val[k]=Y1_i[0,ind_val_i]
			#Get train indices
			train_test_indices = indices[train_index]
			ind_train_i, ind_test_i = train_test_split(train_test_indices, test_size=(1/(fold_num-1)),shuffle=True)#torch.index_select(indices, 0, torch.tensor(test_index))
			Y_test[k]=Y1_i[0,ind_test_i]
			Y1_i[0,ind_val_i] =torch.tensor(float('NaN'))  
			Y1_i[0,ind_test_i] =torch.tensor(float('NaN'))   
			Y1_no_nan = Y1_i.clone().detach()
			Y1_isnan = torch.isnan(Y1_i)
			indices_tensor = torch.nonzero(Y1_isnan)
			Y1_no_nan[indices_tensor[:,0],indices_tensor[:,1]]=0  
			

					 
			#Y1_no_nan = Y1_i.clone().detach()
			#Y1_isnan = torch.isnan(Y1_i)
			#indices_tensor = torch.nonzero(Y1_isnan)
			#Y1_no_nan[indices_tensor[:,0],indices_tensor[:,1]]=0	  
			
			#Y_train_w_nan.append(Y1_i.view(self.I,self.J,self.K))
			#Y_train_wo_nan.append(Y1_no_nan.view(self.I,self.J,self.K))
			Y_train_w_nan[k]=Y1_i.view(self.I,self.J,self.K)
			Y_train_wo_nan[k]=Y1_no_nan.view(self.I,self.J,self.K)
			
			Omega_Y_train[k] = torch.nonzero(torch.logical_not(torch.isnan(Y_train_w_nan[k])))
			Omega_linear_Y_train[k] = torch.nonzero(torch.logical_not(torch.isnan(Y1_i.squeeze())))
			Omega_linear_Y_train[k] = Omega_linear_Y_train[k].squeeze()
 
			
			ind_test[k]=ind_test_i
			ind_val[k]=ind_val_i
			k=k+1

		return(Y_train_w_nan, 
			   Y_train_wo_nan, 
			   Omega_Y_train, 
			   Omega_linear_Y_train, 
			   Y_val, 
			   ind_val, 
			   Y_test, 
			   ind_test)
		