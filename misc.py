import numpy as np

from tqdm import tqdm

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


def extract_foreground(images_i, depths_i, ids, scene_ids):
    images = np.copy(images_i)
    depths = np.copy(depths_i)

    # ids_foreground = []
    ids_thresh = -1
    for id in scene_ids['ids']:
        for label in scene_ids['ids'][id]:
            if 'vine' in label:
                ids_thresh = int(id)
                
            if ids_thresh != -1:
                break
        if ids_thresh != -1:
            break


    mask = np.zeros(ids.shape, dtype=bool)
    mask[ids > ids_thresh] = True
    # a = ids[..., None].astype(int)
    # b = np.array(ids_foreground)[None, None, None, :]
    # c = np.any(a == b, axis=-1)
    # print(c.shape, c)
    # mask[c] = True
    # for id in tqdm(ids_foreground):
    #     mask[ids == id] = True
    
    images[~mask] = 0
    depths[~mask] = 0
        
    return images, depths
