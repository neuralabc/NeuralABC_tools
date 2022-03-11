"""
Author: Christopher J. Steele
Date: 2022-03-05
"""

import numpy as np
from scipy.stats import binned_statistic


def mm_norm(array):
    """
    Normalize n-dimensional numpy array to between 0 and 1

    :param array: ndarray
    :return: ndarray with range = {0,1}
    """
    mmin = array.min()
    mmax = array.max()
    return (array - mmin) / (mmax - mmin)


def map_vals_to_index(index_array, key_vals):
    """
    Map a set of values in vector [key_vals] into ndarray [index_array]. The shape of np.unique(index_array) must be
    the same as that of key_vals. The index order of values in key_vals must be the same as the increasing indices of
    index_array. We do not remove the 0 index, the value to fill ALL indices in index_array (i.e., np.unique()) must
    also be in key_vals.

    :param index_array: ndarray of type int
    :param key_vals: 1-d ndarray containing sorted order of values to map to index_array (if 0 in index_array, include 0)

    :return: ndarray of shape index_array.shape() with key_vals mapped into ordered indices of index_array
    """

    palette = np.unique(index_array)  # sorted order of values in index_array that we will map to
    index = np.digitize(index_array, palette, right=True)  # create an index of palette to index_array
    return key_vals[index]  # fill key_vals into index


# slower implementation for testing
def _loop_map_vals_to_index(index_array, key_vals):
    """
    Use for loop to map values into index array. For comparison with the faster map_vals_to_index and paranoid
    verification that the same results are returned. All integers, including 0, included.

    :param index_array: ndarray of indices (dtype int) for mapping key_vals into
    :param key_vals: 1d ndarray with values to map to index_array in increasing order
    :return:
    """

    index_array_vals = np.unique(index_array)  # returns ordered vector, equivalent to ordering of key_vals
    d_out = np.zeros(index_array.shape)
    for idx, val in enumerate(index_array_vals):
        d_out[index_array == val] = key_vals[idx]
    return d_out


def summed_statistics_by_index(index_array, val_array, statistic='mean', ignore_zero_index=True):
    """
    Wrapper for scipy.stats.binned_statistic
    
    :param index_array: np.ndarray with indices defining regions over which statistic will be calculated
    :param val_array: np.ndarray with values which will be summarized based on the index_array
    :param statistic: one of {'mean','std','median','count','sum','min','max',function}
    :return: 
    """

    if not (index_array.dtype == int):
        print('Your index array is not of type int, please fix!')
        return 0
    else:
        bins = np.unique(index_array)
        if ignore_zero_index and (bins[0] == 0):
            bins = bins[1:]
        return binned_statistic(index_array, val_array, bins=bins, statistic=statistic, range=(bins.min(), bins.max()))
