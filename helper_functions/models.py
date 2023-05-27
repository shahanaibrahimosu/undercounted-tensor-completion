import torch
from torch import nn
from torch.nn import functional as Func
import numpy as np


class NN_detetcion(nn.Module):
    def __init__(self, R, hidden_units_g, hidden_layer_g):
        super(NN_detetcion, self).__init__()
        layerlist=[]
        n_in = R
        for i in range(hidden_layer_g):
            layerlist.append(nn.Linear(n_in,hidden_units_g))
            layerlist.append(nn.ReLU(inplace=True))
            n_in = hidden_units_g
        layerlist.append(nn.Linear(hidden_units_g,1))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, z):
        x = self.layers(z)
        p = torch.sigmoid(x)
        return p
		
		
class probability_model(nn.Module):
    def __init__(self,size):
        super(probability_model, self).__init__()
        self.w = nn.Parameter(torch.rand(size,1), requires_grad=True)
    def forward(self,ind):
        #P = Func.sigmoid(self.w)
        P=self.w[ind,0]
        P=torch.clamp(P,min=1e-12,max=1)
        return P


class NN_detetcion_linear(nn.Module):
    def __init__(self, R):
        super(NN_detetcion_linear, self).__init__()
        self.theta = nn.Linear(R, 1)
    def forward(self, z):
        p = torch.sigmoid(self.theta(z))
        #p=torch.clamp(self.theta(z),min=1e-12,max=1)
        #print(self.theta.weight)
        #breakpoint()
        return p
		
class NN_detetcion_groundtruth(nn.Module):
    def __init__(self, R, hidden_units_g):
        super(NN_detetcion_groundtruth, self).__init__()
        self.theta = nn.Linear(R, hidden_units_g)
        self.w   = nn.Linear(hidden_units_g,1)

    def forward(self, z):
        x = torch.tanh((self.theta(z)))+0.5*z
        p = torch.sigmoid(self.w(x))
        return p