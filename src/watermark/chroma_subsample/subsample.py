"""
Chroma subsampling really messes up DWT approximation coeffecient values
Causing watermarks to often get lost.

If we pre-chroma subsample an image before we watermark, then the DWT approximation 
coefficients are what they would have been post-subsampling, thus the
resulting watermark is more robust to chroma subsampling attacks.

WebP's chroma subsampling is 4:2:0. As such, we will apply the same.
"""
import numpy as np
from enum import Enum


class SubsampleOptions(Enum):
  FOUR_FOUR_TWO = 0
  FOUR_TWO_ZERO = 1


def subsample(
    yuv_img: np.ndarray,
    u=True,
    v=False,
    subsample_type: SubsampleOptions = SubsampleOptions.FOUR_TWO_ZERO,
  ):
  if subsample_type == SubsampleOptions.FOUR_FOUR_TWO:
    return ValueError("Not yet implemented")

  channels = []
  if u:
    channels.append(1)
  if v:
    channels.append(2)

  subsampled_image = yuv_img.copy()
  cols, rows, _ = yuv_img.shape
  for channel in channels:

    # Horizontal copy
    print("~~~~ Horizontal ~~~~~")
    print(subsampled_image[:, :, channel].shape)
    print(subsampled_image[:, 1:rows // 2 * 2:2, channel].shape)
    print(subsampled_image[:, ::2, channel].shape)
    last_source_row = rows // 2 * 2
    subsampled_image[:, 1::2, channel] = subsampled_image[:,
                                                          :last_source_row:2, channel]

    # if subsample_type == SubsampleOptions.FOUR_FOUR_TWO:
    #   continue

    # Vertical copy
    print("~~~~ Vertical ~~~~~")
    print(subsampled_image[1::2, :, channel].shape)
    print(subsampled_image[::2, :, channel].shape)
    last_source_col = cols // 2 * 2
    subsampled_image[1::2, :,
                     channel] = subsampled_image[:last_source_col:2, :, channel]

  return subsampled_image
