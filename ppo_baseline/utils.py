import numpy as np
from copy import deepcopy


def grid_trans(np_array, trans_type):
    if trans_type == "none":
        new_array = np_array
    
    elif trans_type == "rot90":
        new_array = np.rot90(np_array, 1)
    
    elif trans_type == "rot180":
        new_array = np.rot90(np_array, 2)
        
    elif trans_type == "rot270":
        new_array = np.rot90(np_array, 3)
    
    elif trans_type == "fliplr":
        new_array = np.fliplr(np_array)
    
    elif trans_type == "rot90 + fliplr":
        new_array = np.rot90(np_array, 1)
        new_array = np.fliplr(new_array)
        
    elif trans_type == "rot180 + fliplr":
        new_array = np.rot90(np_array, 2)
        new_array = np.fliplr(new_array)
    
    elif trans_type == "rot270 + fliplr":
        new_array = np.rot90(np_array, 3)
        new_array = np.fliplr(new_array)
        
    return deepcopy(new_array)









