import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from scipy import stats

def getMSE_U(U,U_true):
	F = np.shape(U)[1]
	row_ind, col_ind  = linear_sum_assignment(-np.dot(np.transpose(U),U_true))
	U  = U[:,col_ind]
	err=0
	for f in range(F):
		d1=U[:,f]/np.linalg.norm(U[:,f])
		d2=U_true[:,f]/np.linalg.norm(U_true[:,f])
		err = err+np.linalg.norm(d1-d2)**2
	err = err/F
	return err

def getMSE(U,U_true):
	K = len(U)
	err=0
	for k in range(K):
		err+=getMSE_U(U[k],U_true[k])
	err = err/K
	return err

def getMSE_all_factors(A,A_g,K):
	for k in range(K):
		err = err+ getMSE(A[k].numpy(),A_g[k])
	err = err/K
	return err


def getMSE_entry(L,L_g):
	size_L=torch.numel(L)
	L=L.view(-1)
	L_g=L_g.view(-1)
	L = L/torch.mean(L)
	L_g = L_g/torch.mean(L_g)
	err = torch.sum(torch.abs(L-L_g))/size_L
	return err

def prediction(Y_est,test_indices):
	Y1=Y_est.view(1,torch.numel(Y_est))
	Y_test_est = Y1[0,test_indices]
	return Y_test_est

def get_rRMSE(Y_est,Y):
	#size_Y = list(Y1.size())
	#K = len(size_Y)
	#index_list=[]
	#for k in range(K):
	#	index_list.append(index[:,k])
	#Y = Y1[index_list]
	#Y_est = Y1_est[index_list]
	diff = (Y_est-Y)**2
	RMSE=torch.sqrt(torch.sum(diff)/Y.size()[0])
	rRMSE = RMSE/torch.mean(Y)
	return rRMSE.numpy()

# def get_AUROC(Y_est,Y):
#	 auroc = roc_auc_score(np.array(Y),np.array(Y_est),multi_class='ovr')
#	 fpr, tpr, thresholds = metrics.roc_curve(train_y_true, train_y_pred)
#	 auroc = metrics.auc(fpr, tpr)
#	 precision, recall, thresholds = \
#						 precision_recall_curve(train_y_true, train_y_pred)
#		 auprc = metrics.auc(recall, precision)
#	 return auroc

# def get_AUPRC(Y_est,Y):
#	 auprc = average_precision_score(np.array(Y),np.array(Y_est))
#	 return auprc

def get_MAPE(Y_est,Y):
	#ind = torch.nonzero(Y)
	#Y_est = Y_est[ind] 
	#Y = Y[ind] 
	mape = mean_absolute_error(Y.numpy(),Y_est.numpy())
	#mape = mape/torch.mean(Y)
	#mape = mean_absolute_percentage_error(Y.numpy(),Y_est.numpy())
	#mape = mape/torch.mean(Y)
	return mape

def get_f1score(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	Y_est = np.round(Y_est)
	Y= np.round(Y)
	Y_est[Y_est>0]=1
	Y[Y>0]=1
	f1score = f1_score(Y,Y_est,average='macro')
	return f1score

def get_hammingloss(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	hloss = hamming_loss(np.round(Y),np.round(Y_est))
	return hloss

def get_pearsoncorr(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	pc = stats.pearsonr(Y,Y_est)
	return pc

def get_roc_auc(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	Y_est = np.round(Y_est)
	Y= np.round(Y)
	Y_est[Y_est>0]=1
	Y[Y>0]=1
	roc_au = roc_auc_score(Y,Y_est,average='macro')
	return roc_au

def get_prc(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	Y_est = np.round(Y_est)
	Y= np.round(Y)
	Y_est[Y_est>0]=1
	Y[Y>0]=1
	f1score = average_precision_score(Y,Y_est,average='macro')
	return f1score

def get_auprc(Y_est,Y):
	Y_est = Y_est.numpy()
	Y = Y.numpy()
	Y_est = np.round(Y_est)
	Y= np.round(Y)
	Y_est[Y_est>0]=1
	Y[Y>0]=1
	precision, recall, thresholds = precision_recall_curve(Y,Y_est)
	auc_precision_recall = auc(recall, precision)
	return auc_precision_recall
	
	
def get_totalcost(M1,P1,Y1,index):
	size_M = list(M1.size())
	K = len(size_M)
	index_list=[]
	for k in range(K):
		index_list.append(index[:,k])
	M = M1[index_list]
	P = P1[index_list]
	Y = Y1[index_list]
	total_loss=torch.sum((M)*P-Y*torch.log(P)-Y*torch.log(M))/(index.size()[0])
	#print('M')
	#print(M)
	##print('Y')
	##print(Y)
	#print('P')
	#print(P)
	return(total_loss)

def get_tensorcost(M1,P1,Y1,index):
	size_M = list(M1.size())
	K = len(size_M)
	index_list=[]
	for k in range(K):
		index_list.append(index[:,k])
	M = M1[index_list]
	P = P1[index_list]
	Y = Y1[index_list]
	loss = torch.sum(P*M-Y * torch.log(M))/(index.size()[0])
	return(loss)


def evaluation(Y_est,Y):
	rRMSE=get_rRMSE(Y_est,Y)
	MAPE=get_MAPE(Y_est,Y)
	AUPRC=get_auprc(Y_est,Y)
	AUROC=get_roc_auc(Y_est,Y)
	#pearsoncorr=get_pearsoncorr(Y_est,Y)
	return rRMSE,MAPE,AUROC,AUPRC