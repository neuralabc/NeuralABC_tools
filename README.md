# NeuralABC_tools
Collection of tools used in the NeuralABC lab

Parallelize() example usage:

```python

arr = [2,3,5,7,11,12]

def is_odd(n):
    return True if n % 2 == 1 else False

Parallelize(arr, is_odd, 10)

'''
Returns 
array([[False],
       [ True],
       [ True],
       [ True],
       [ True],
       [False]])
'''

```
