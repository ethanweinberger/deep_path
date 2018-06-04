#Data augmentation methods

import numpy as np
from patch import Patch
from PIL import Image

def horizontal_flip(orig_patch):
    """
    Horizontally flips a patch

    Args:
        orig_patch (Patch): Patch object

    Returns:
        flipped_patch (Patch): New patch object resulting from flipping
                               the original patch horizontally
    
    """

    flipped_img = np.ndarray(orig_patch.img.shape, dtype=np.uint8)
    flipped_img[:,:,0] = np.fliplr(orig_patch.img[:,:,0])
    flipped_img[:,:,1] = np.fliplr(orig_patch.img[:,:,1])
    flipped_img[:,:,2] = np.fliplr(orig_patch.img[:,:,2])

    flipped_patch = Patch(flipped_img, orig_patch.coords)
    return flipped_patch

def rotate(orig_patch, degrees):
    """
    Rotate the image in a given patch by a specified number of degrees

    Args:
        orig_patch (Patch): Patch object

    Returns:
        rotated_patch (Patch): New patch object resulting from rotationg
                               the original patch by the specified number
                               of degrees

    """

    orig_img = Image.fromarray(orig_patch.img)
    rotated_img = orig_img.rotate(degrees)
    rotated_patch = Patch(np.array(rotated_img), orig_patch.coords) 
    return rotated_patch
