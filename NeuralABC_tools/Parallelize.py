import multiprocessing
from multiprocessing import Pool
import numpy as np

def __processor_procedure(processor_index, ranges, f):
    """
    __processor_procedure() is a helper function for Paralellize() and there is no reason for the user to manually call this function
    One instance of this function is run per core. The number of cores can be specified in the parameters of Parallelize()
    
    Parameters:
    ----------------
    :param processor_index: (integer) required: The index of the processor which this function is running on
    :param ranges: (dictionary) required: The list of elements that the function is responsible for processing is contained in this dictionary
    :param f: (function) required: The function that the elements will be passed to
    """
    

    '''
    Reference the global variable defined in Parallelize(). This is a "hacky" trick to grab the input data without 
    directly passing a copy of it to each function, which would waste time and resources
    '''
    global __48162789682__
    data = __48162789682__
    
    #Create the dictionary of outputs
    __processed = {}
    
    #If one passes the "processor_index" variable to the dictionary "ranges" as a key, the value received is an array
    #of element indices that specify which elements in the data the function must process
    for index in ranges[processor_index]:
        #Process the data by running it through the f() function
        _out = f(data[index])
        __processed[index] = _out

    return __processed



def Parallelize(data, f, num_cores=None):
    """
    Parallelize() allows for the parallelization of for-loop when processing data. 
    
    Parameters:
    ----------------
    :param data: (Python array or Numpy array) required: the sequence of data to be processed
    :param f: (a function) required: A function that takes in an element of data and returns the desired processed output of that element
    :param num_cores: (integer) optional: The desired number of cores to allocate. If unspecified, the maximum possible will be used
    
    
    Returns:
    ----------------
    Returns a NumPy array of shape (number of elements to process, -1) where -1 depends on the user-chosen function f
    """
    
    #Get the number of "subjects" i.e. the length of the data
    num_subjects = len(data)
    if(num_cores == None): #If the number of cores in unspecified, get the maximum number of cores available
        num_cores = multiprocessing.cpu_count()
    
    #The number of cores we use cannot exceed the number of subjects there are
    num_cores = min(num_cores, num_subjects, multiprocessing.cpu_count())
    
    
    #Creating a dictionary that has the core indices as keys and ranges of data element indices as the values
    cores_to_use = num_cores - 1
    range_dict = {}
    for i in range(cores_to_use):
        range_dict[i] = np.arange(i*int(num_subjects/(cores_to_use)), (i+1)*int(num_subjects/(cores_to_use)))
    range_dict[cores_to_use] = np.arange(num_subjects - num_subjects%cores_to_use, num_subjects)
    
    #Checking that there is an entry of the dictionary for every core that we will use
    assert(num_cores == len(range_dict))
    
    #Creating a list of tuples. Each tuple will serve as the arguments passed to the __processor_procedure helper function
    pass_data = []
    
    for core_index in range(num_cores):
        pass_data.append((core_index, range_dict, f))
    
    #Create a global variable and copy the data to it. The variable name was chosen in a way such that it hopefully would not
    #overwrite a user-defined variable. By creating such a global variable, each of the helper functions can easily access the
    #data without having it explicitly passed to them, which would waste time and resources
    global __48162789682__
    __48162789682__ = data
    
    #Initialize a pool of workers. 
    with Pool(num_cores) as p:
        _returns = p.starmap(__processor_procedure, pass_data)
    
    #Delete the global variable that we defined
    del __48162789682__
    
    #Process the _returns dictionary into a NumPy array
    final_dict = {}
    for d in _returns:
        final_dict = {**final_dict, **d}
    
    return np.array(list(final_dict.values())).reshape(len(data),-1)
