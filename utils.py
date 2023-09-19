import torch
import numpy as np
from collections import OrderedDict

def duplicate_channel_cv(image, channel=3, camera="D400"):
    if camera == "D400":
        image = np.reshape(image, (480, 640, 1))
    elif camera == "L500":
        image = np.reshape(image, (540, 960, 1))
    elif camera == "AzureKinect":
        image = np.reshape(image, (720, 1280, 1))
    else:
        raise ValueError('Expect camera product line: "D400" or "L500" or "AzureKinect".')

    image = np.repeat(image ,channel, -1)
    return image

def array_to_tensor(arr, dtype: str):
    
    assert dtype == "float" or dtype == "int" ,'dtype should be either float or int type' 

    if dtype == "float":
        if arr.ndim == 4: # NHWC
            tensor = torch.from_numpy(arr).permute(0,3,1,2).float()
        elif arr.ndim == 3: # HWC
            tensor = torch.from_numpy(arr).permute(2,0,1).float()
        else: # everything else
            tensor = torch.from_numpy(arr).float()
    
    elif dtype == "int":
        if arr.ndim == 4: # NHWC
            tensor = torch.from_numpy(arr).permute(0,3,1,2)
        elif arr.ndim == 3: # HWC
            tensor = torch.from_numpy(arr).permute(2,0,1)
        else: # everything else
            tensor = torch.from_numpy(arr)

    return tensor

def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict