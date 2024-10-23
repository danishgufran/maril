"""
An implementation of the fingerprint-image transformer method from the following work:

D. Li, J. Xu, Z. Yang, Y. Lu, Q. Zhang and X. Zhang, 
"Train Once, Locate Anytime for Anyone: Adversarial Learning based Wireless Localization," 
IEEE Conference on Computer Communications, 2021

Sub-project refers to the implementation of AdLoc

Notes to contributors:
- Keep all dependencies within sub-project
- Usable functions should be importable from outside sub-project
- Enable high flexibility when possible from outside sub-project
- The model should have a contention layer (bottleneck) within it
"""


# TODO:
# Function(s) to build model as per paper
# Ability to provide custom model structure as input

import numpy as np
from tqdm.auto import tqdm

def fingerprint_transformer(x: np.array) -> np.array:
    """Transform fingerprints as given in iToLoc using
    fp[i] - fp[j] logic.

    Parameters
    ----------
    x : np.array
        2D numpy array where each row is a fingerprint

    Returns
    -------
    np.array
        transformed images in NHWC format; TF default
    """

    # assuming NCHW format for tensorflow
    # create array that will hold images
    imgs = np.full(
        (
            x.shape[0],  # number of rows in x
            x.shape[1],  # height
            x.shape[1],  # width
            1,  # ####### image is greyscale
        ),
        fill_value=np.nan,
        dtype=float,
    )

    # iterate over each fingerprint
    for img_num, fp in tqdm(enumerate(x), total=x.shape[0]):

        # iter over same fingerprint twice
        for i in range(fp.shape[0]):
            for j in range(fp.shape[0]):

                # take absolute difference to avoid negative numbers
                imgs[img_num, i, j, 0] = abs(fp[i] - fp[j])

    # check for safety
    assert np.isnan(imgs).any() == False, "imgs array may have NaN"

    return imgs
