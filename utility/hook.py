import torch.nn as nn 




def get_all_layers(net,hook_fn):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer,hook_fn)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)
