import numpy as np


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
  m = np.array([
      [0.29900, -0.14713, 0.615],
      [0.58700, -0.28886, -0.51499],
      [0.11400, 0.436, -0.10001]
    ])

  # rgb = rgb.astype(float)
  yuv = np.dot(rgb, m)
  yuv[:, :, 1:] += 127.5
  yuv = np.clip(yuv, 0, 255)
  # return yuv
  return np.round(yuv).astype(int)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
  yuv = yuv.astype(float)
  m = np.array([
      [1.000, 1.000, 1.000],
      [0.000, -0.39465, 2.03211],
      [1.13983, -0.58060, 0.000],
    ])

  yuv[:, :, 1:] -= 127.5
  rgb = np.dot(yuv, m)
  rgb = np.clip(rgb, 0, 255)
  return np.round(rgb).astype(int)
