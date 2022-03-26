import numpy as np

def remove_background(images_i, depths_i, threshold):
    images = np.copy(images_i)
    depths = np.copy(depths_i)

    mask = np.zeros((*depths_i.shape,), dtype=bool)
    mask[depths_i > threshold] = True

    images[np.broadcast_to(mask[..., None], (*images.shape,))] = 0
    images = np.concatenate((images, 1 - np.float32(mask[..., None])), axis=-1)
    depths[mask] = np.inf

    return images, depths

