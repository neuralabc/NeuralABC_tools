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

A more involved example

```python


#pc_stack is an (1001, 119387) array containing the masked first PC data of the 1001 subjects
pc_stack = np.loadtxt("Out2.txt")

#Define a few masks
old_mask = "/data/neuralabc/pirami/preprocess/Group_averages/masks/T1_5tt_warped_multcon_cubic_mean_vol2_WM_thr0p9_bin.nii.gz"
CC_mask = "Pezzoli_region.nii.gz"
correct_img = nb.load(old_mask)
resampled_mask = resample_img(CC_mask, target_affine=correct_img.affine, 
                              interpolation="nearest",
                             target_shape=(147, 183, 144))


#Our goal is to constrain the PC data to only include the voxels belonging to the amygdala
def get_amygdala_data(single):
    _img_old = masking.unmask(single, old_mask)
    _amy_arr = masking.apply_mask(_img_old, resampled_mask)
    
    return _amy_arr


%%time
processed = Parallelize(pc_stack, get_amygdala_data, 100)

'''
CPU times: user 863 ms, sys: 2.69 s, total: 3.56 s
Wall time: 18.3 s
'''

print(processed.shape)
#(1001, 39203)

```
