import multiprocessing
from multiprocessing import Pool
import numpy as np

def processor_procedure(processor_index, data, ranges, f):
    processed_data_dict = {}

    for index in ranges[processor_index]:
        single_processed = f(data[index])
        processed_data_dict[index] = single_processed

    return masked_data_dict

def Parallelize(data, f, num_cores=None):
    
    num_subjects = len(data)
    if(num_cores == None):
        num_cores = multiprocessing.cpu_count()
    
    num_cores = min(num_cores, num_subjects, multiprocessing.cpu_count())
    
    cores_to_use = num_cores - 1
    range_dict = {}
    for i in range(cores_to_use):
        range_dict[i] = np.arange(i*int(num_subjects/(cores_to_use)), (i+1)*int(num_subjects/(cores_to_use)))
    range_dict[cores_to_use] = np.arange(num_subjects - num_subjects%cores_to_use, num_subjects)
    
    
    pass_data = []
    
    for core_index in range(num_cores):
        pass_data.append((core_index, data, range_dict, f))
        
    #print("Start")
        
    with Pool(num_cores) as p:
        _returns = p.starmap(processor_procedure, pass_data)
    
    #print("End")
    
    final_dict = {}
    for d in _returns:
        final_dict = {**final_dict, **d}
    
    return np.array(list(final_dict.values()))
