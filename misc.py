import numpy as np

def remove_background(images_i, depths_i, ids, threshold):
    images = np.copy(images_i)
    depths = np.copy(depths_i)

    mask = np.zeros((*depths_i.shape,), dtype=bool)
    mask[depths_i > threshold] = True

    mask_ids = np.zeros((*ids.shape,), dtype=bool)
    mask_ids[ids < 128] = True

    mask = np.logical_or(mask, mask_ids)

    images[np.broadcast_to(mask[..., None], (*images.shape,))] = 0
    images = np.concatenate((images, 1 - np.float32(mask[..., None])), axis=-1)
    depths[mask] = np.inf

    return images, depths

