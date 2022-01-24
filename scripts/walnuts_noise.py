from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from dataset.walnuts_interface import (
        VOL_SZ, PROJS_COLS, get_projection_data,
        get_single_slice_ray_trafo, get_single_slice_ind, get_ground_truth)

ANGULAR_SUB_SAMPLING = 20
PROJ_COL_SUB_SAMPLING = 6
# ANGULAR_SUB_SAMPLING = 10
# PROJ_COL_SUB_SAMPLING = 1

num_proj_cols = ceil(PROJS_COLS / PROJ_COL_SUB_SAMPLING)

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = 'walnuts/'

walnut_ray_trafo = get_single_slice_ray_trafo(
        data_path=DATA_PATH,
        walnut_id=WALNUT_ID,
        orbit_id=ORBIT_ID,
        angular_sub_sampling=ANGULAR_SUB_SAMPLING,
        proj_col_sub_sampling=PROJ_COL_SUB_SAMPLING)

print('vol_slice_contributing_to_masked_projs',
        walnut_ray_trafo.get_vol_slice_contributing_to_masked_projs())
print('proj_slice_contributing_to_masked_vol',
        walnut_ray_trafo.get_proj_slice_contributing_to_masked_vol())

vol_in_mask = np.ones((1,) + VOL_SZ[1:])
vol_x = np.zeros((walnut_ray_trafo.num_slices,) + VOL_SZ[1:])
vol_x[walnut_ray_trafo.vol_mask_slice] = vol_in_mask
projs = walnut_ray_trafo.fp3d(vol_x)
backprojection_mask = walnut_ray_trafo.bp3d(
        walnut_ray_trafo.proj_mask.astype(np.float32))

# test FDK reconstruction from restricted projections
projs_full = get_projection_data(data_path=DATA_PATH,
                                 walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
                                 angular_sub_sampling=ANGULAR_SUB_SAMPLING,
                                 proj_col_sub_sampling=PROJ_COL_SUB_SAMPLING)
projs = walnut_ray_trafo.projs_from_full(projs_full)
flat_projs_in_mask = walnut_ray_trafo.flat_projs_in_mask(projs)

slice_ind = get_single_slice_ind(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID)
gt_slices = [
        get_ground_truth(data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
                         slice_ind=slice_ind) for slice_ind in range(
                (VOL_SZ[0] - 1) // 2 - (walnut_ray_trafo.num_slices - 1) // 2,
                (VOL_SZ[0] - 1) // 2 + (walnut_ray_trafo.num_slices - 1) // 2 + 1)]
gt_projs = walnut_ray_trafo.fp3d(gt_slices)

mask_index_array = np.expand_dims(np.argmax(walnut_ray_trafo.proj_mask, axis=0), axis=0)

projs = np.take_along_axis(projs, mask_index_array, axis=0)[0]
gt_projs = np.take_along_axis(gt_projs, mask_index_array, axis=0)[0]

scaling_factor = 14.  # cfg.data.walnut.scaling_factor
print('mse:', np.mean((scaling_factor * projs - scaling_factor * gt_projs)**2))
print(projs.shape)


titles = ['measurement', 'ground truth FP', 'diff']

projs = np.exp(-projs)
gt_projs = np.exp(-gt_projs)

sinograms = [projs, gt_projs, projs-gt_projs]

fig, ax = plt.subplots(1, len(sinograms))

for i, (sinogram, title) in enumerate(zip(sinograms, titles)):
    im = ax[i].imshow(sinogram.T)
    ax[i].set_title(title)
    plt.colorbar(im, ax=ax[i])

plt.show()

fig, ax = plt.subplots(1, 1)

ax.scatter(np.ravel(gt_projs), np.ravel(projs-gt_projs)**2, marker='.')
ax.set_xlabel('projection value')
ax.set_ylabel('squared error')

plt.show()
