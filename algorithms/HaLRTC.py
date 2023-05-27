import torch
from torch import nn
from torch.nn import functional as Func
import numpy as np
import tensorly as tl
import time

import copy

class HaLRTC:
    def __init__(self, alpha: list, rho: float, epsilon: float, maxiter: int):
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.maxiter = maxiter

        
    def ten2mat(self,tensor, mode):
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

    def mat2ten(self,mat, dim, mode):
        index = list()
        index.append(mode)
        for i in range(dim.shape[0]):
            if i != mode:
                index.append(i)
        return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

    def svt(self, mat, tau):
        u, s, v = np.linalg.svd(mat, full_matrices = False)
        vec = s - tau
        vec[vec < 0] = 0
        return np.matmul(np.matmul(u, np.diag(vec)), v) 
    
    def compute_mape(self,var, var_hat):
        return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

    def compute_rmse(self,var, var_hat):
        return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
    
    def train_HaLRTC(self, dense_tensor, sparse_tensor,flag_groundtruth):
        dim = np.array(sparse_tensor.shape)
        tensor_hat = sparse_tensor
        pos_missing = np.where(sparse_tensor == 0)
        pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
        B = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
        Y = [np.zeros(sparse_tensor.shape) for _ in range(len(dim))]
        last_ten = sparse_tensor.copy()
        snorm = np.linalg.norm(sparse_tensor)

        it = 0
        while True:
            for k in range(len(dim)):
                B[k] = self.mat2ten(self.svt(self.ten2mat(tensor_hat + Y[k] / self.rho, k), self.alpha[k] / self.rho), dim, k)
            tensor_hat[pos_missing] = ((sum(B) - sum(Y) / self.rho) / 3)[pos_missing]
            for k in range(len(dim)):
                Y[k] = Y[k] - self.rho * (B[k] - tensor_hat)
            tol = np.linalg.norm((tensor_hat - last_ten)) / snorm
            last_ten = tensor_hat.copy()
            it += 1
            if it % 1 == 0 and flag_groundtruth == 1:
                print('Iter: {}'.format(it))
                print('Tolerance: {:.6}'.format(tol))
                print('MAPE: {:.6}'.format(self.compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
                print('RMSE: {:.6}'.format(self.compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
                print()
            if (tol < self.epsilon) or (it >= self.maxiter):
                break
        
        if flag_groundtruth == 1:
            print('Total iteration: {}'.format(it))
            print('Tolerance: {:.6}'.format(tol))
            print('MAPE: {:.6}'.format(self.compute_mape(dense_tensor[pos_test], tensor_hat[pos_test])))
            print('RMSE: {:.6}'.format(self.compute_rmse(dense_tensor[pos_test], tensor_hat[pos_test])))
            print()

        return tensor_hat
