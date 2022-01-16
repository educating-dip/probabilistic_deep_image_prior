from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from dataset.walnuts_interface import VOL_SZ, get_single_slice_ind, get_ground_truth
from scalable_linearised_laplace import get_image_block_masks

WALNUT_ID = 1
ORBIT_ID = 2

DATA_PATH = 'walnuts/'

block_size = 8
start_0 = 72
start_1 = 72
end_0 = 424
end_1 = 424
chunks = 8

slice_ind = get_single_slice_ind(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID)
gt = get_ground_truth(
        data_path=DATA_PATH, walnut_id=WALNUT_ID, orbit_id=ORBIT_ID,
        slice_ind=slice_ind)

block_masks = get_image_block_masks(gt.shape, block_size)

num_blocks_0 = gt.shape[0] // block_size
num_blocks_1 = gt.shape[1] // block_size
start_block_0 = start_0 // block_size
start_block_1 = start_1 // block_size
end_block_0 = ceil(end_0 / block_size)
end_block_1 = ceil(end_1 / block_size)

block_idx_list = [
    block_idx for block_idx in range(len(block_masks))
    if block_idx % num_blocks_0 in range(start_block_0, end_block_0) and
    block_idx // num_blocks_0 in range(start_block_1, end_block_1)]

blocks_per_chunk = ceil(len(block_idx_list) / chunks)
block_idx_list_per_chunk = [
    block_idx_list[blocks_per_chunk*chunk_idx:blocks_per_chunk*(chunk_idx+1)] for chunk_idx in range(chunks)]

def sort_key(block_idx):
    inds_0, inds_1 = np.nonzero(block_masks[block_idx].reshape(gt.shape))
    dist_to_center = np.mean(np.sqrt((inds_0 - (gt.shape[0]-1)//2)**2 + (inds_1 - (gt.shape[1]-1)//2)**2))
    return dist_to_center

block_idx_list_per_chunk = [sorted(b, key=sort_key) for b in block_idx_list_per_chunk]

for chunk_idx, chunk_block_idx_list in enumerate(block_idx_list_per_chunk):
    print('chunk {} (size {})'.format(chunk_idx, len(chunk_block_idx_list)))
    print('[' + ','.join(map(str, chunk_block_idx_list)) + ']')

alpha_masks_per_chunk = []
for chunk_idx, chunk_block_idx_list in enumerate(block_idx_list_per_chunk):
    alpha_mask = np.zeros(gt.shape, dtype=bool)
    for block_idx in chunk_block_idx_list:
        alpha_mask |= block_masks[block_idx].reshape(gt.shape)
    alpha_masks_per_chunk.append(alpha_mask)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()
ax.imshow(gt, cmap='gray')
for alpha_mask, color in zip(alpha_masks_per_chunk, colors):
    im_mask = np.ones(gt.shape + (4,))
    im_mask[:, :, 0:3] = mcolors.to_rgb(color)
    im_mask[:, :, 3] = alpha_mask
    ax.imshow(im_mask, alpha=0.5)

plt.show()
