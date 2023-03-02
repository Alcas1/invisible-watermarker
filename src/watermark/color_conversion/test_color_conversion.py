import unittest
import numpy as np
from PIL import Image
import cv2

from color_conversion import color_conversion


def original_image():
  img = cv2.imread("../data/original.jpg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def peppers_image():
  img = cv2.imread("../data/peppers.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def transpose(img):
  return np.transpose(img, (2, 0, 1))


class TestColorConversion(unittest.TestCase):
  def test_rgb_yuv_conversions(self):
    for image in [
      original_image(),
      peppers_image(),
    ]:
      Image.fromarray(np.uint8(image)).save("readin.png")

      yuv_expected = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
      yuv_actual = color_conversion.rgb_to_yuv(image)

      Image.fromarray(np.uint8(yuv_expected)).save("yuv_expected.png")
      Image.fromarray(np.uint8(yuv_actual)).save("yuv_actual.png")

      diff = yuv_expected - yuv_actual
      big_diff = np.where(abs(diff) > 1)

      print(diff[big_diff])
      for dim in range(3):
        self.assertTrue(big_diff[dim].size == 0)

      rgb_expected = np.array(cv2.cvtColor(yuv_expected, cv2.COLOR_YUV2RGB))
      rgb_actual = color_conversion.yuv_to_rgb(yuv_actual)

      Image.fromarray(np.uint8(rgb_expected)).save("rgb_expected.png")
      Image.fromarray(np.uint8(rgb_actual)).save("rgb_actual.png")

      diff = rgb_expected - rgb_actual
      big_diff = np.where(abs(diff) > 2)

      print(diff[big_diff])
      for dim in range(3):
        self.assertTrue(big_diff[dim].size == 0)
