import unittest
import numpy as np
import cv2
from PIL import Image

from encoder.watermark_encoder import WatermarkEncoder, WatermarkDecoder


def original_image():
  img = cv2.imread("../data/original.jpg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def peppers_image():
  img = cv2.imread("../data/peppers.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encoded_original_image():
  img = cv2.imread("../data/expected_encoded/encoded_original.jpg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encoded_peppers_image():
  img = cv2.imread("../data/expected_encoded/encoded_peppers.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def SDV2_encoder():
  return WatermarkEncoder(content=b"SDV2")


def SDV2_decoder():
  return WatermarkDecoder(wm_length=32)


class TestMaxDctEncode(unittest.TestCase):
  def test_watermark_encode_decode(self):
    encoder = WatermarkEncoder(content=b"SDV2")
    decoder = WatermarkDecoder(wm_length=encoder.get_length())

    for name, image, file_code, file_format in [
      ("original", original_image(), "jpg", "JPEG"),
      ("peppers", peppers_image(), "png", "PNG"),
    ]:
      encoded_img = encoder.max_dwt_encode(image)

      decoded = decoder.decode(encoded_img)

      self.assertEqual(b"SDV2", decoded)

      # encoded_img = cv2.cvtColor(np.uint8(encoded_img), cv2.COLOR_RGB2BGR)
      # cv2.imwrite(
      #   f"../data/expected_encoded/encoded_{name}.{file_code}", encoded_img)
