"""
Author: Christopher J. Steele
Date: 2022-03-05
"""

import numpy as np


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

    palette = np.unique(index_array) #sorted order of values in index_array that we will map to
    index = np.digitize(index_array, palette, right=True) #create an index of palette to index_array
    return key_vals[index] #fill key_vals into index


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