from .opt_interface import * 
import torch 

## here we collect all the pytorch implemented stochastic optimization algorithms
class pytorch_opt(opt_interface):
    def __init__(self):
        self.optdict = {'SGD':torch.optim.SGD, 'Adam':torch.optim.Adam, 'RMSprop':torch.optim.RMSprop}
         
    def create_opt(self, opt_name):
        return self.optdict[opt_name]