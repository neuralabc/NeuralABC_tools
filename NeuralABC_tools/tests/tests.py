from ..arrays import (map_vals_to_index, _loop_map_vals_to_index)

import numpy as np

def test_map_vals_to_index():
    rng = np.random.default_rng()
    idx_array = rng.integers(0, 10 + 1, 10000).reshape((100, 100))
    key_vals = rng.random(np.unique(idx_array).shape)

    res_1 = map_vals_to_index(idx_array, key_vals)
    res_2 = _loop_map_vals_to_index(idx_array, key_vals)

    assert np.allclose(res_1, res_2)
