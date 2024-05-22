import torch.nn as nn
import torch

import numpy as np
import torch.nn.functional as F


class Direct(nn.Module):
    def __init__(self):
        super(Direct, self).__init__()

    def forward(self, x):
        return x
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hiden_dims):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.fcs=nn.ModuleList()
        
        last_dim = input_dim
        for hiden_dim in hiden_dims:
            _ = nn.Linear(last_dim,hiden_dim) 
            self.fcs.append(_)

            last_dim = hiden_dim

        _ = nn.Linear(last_dim,output_dim) 
        
        self.dropout = nn.Dropout(0.5)
        self.fcs.append(_)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for fc in self.fcs[:-1]:
            print(1)
            x = torch.relu(fc(x)) 
            print(x.shape)
        x = self.dropout(x)
        x = self.fcs[-1](x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class ProjectLayer(nn.Sequential):
    def __init__(self, embedding_dim, proj_dim, drop_proj=0.3):
        super(ProjectLayer, self).__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
        
    def forward(self, x):
        return x 

if __name__=='__main__':

    model = MLP(63*250,1024,[])

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)