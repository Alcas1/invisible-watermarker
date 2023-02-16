import io
import unittest
import numpy as np
from PIL import Image
import cv2
from typing import Tuple

from apply_watermark import _bytes_to_nparray, apply_watermark, decode_watermark, Filetype


WATERMARK = "SDV2"


def original_image_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/original.jpg', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.JPEG


def peppers_image_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/peppers.png', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.PNG


def tiger_image_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/tiger.jpeg', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.JPEG


def tiger_png_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/tiger.png', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.PNG


def tall_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/tall.jpg', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.JPEG


def wide_bytes() -> Tuple[bytes, Filetype]:
  with open('../data/wide.jpg', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes, Filetype.JPEG


def expected_original_np_array():
  img = cv2.imread("../data/original.jpg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def expected_peppers_np_array():
  img = cv2.imread("../data/peppers.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def expected_tiger_np_array():
  img = cv2.imread("../data/tiger.jpeg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def expected_tiger_png_np_array():
  img = cv2.imread("../data/tiger.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def expected_original_encoded_bytes():
  with open('../data/expected_encoded/encoded_original.jpg', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes


def expected_peppers_encoded_bytes():
  with open('../data/expected_encoded/encoded_peppers.png', 'rb') as f:
    img_bytes = io.BytesIO(f.read()).read()
  return img_bytes


class TestColorConversion(unittest.TestCase):
  def test_bytes_to_nparray(self):
    for name, input_data, expected in [
      ("original", original_image_bytes(), expected_original_np_array()),
      ("peppers", peppers_image_bytes(), expected_peppers_np_array()),
      ("tiger", tiger_image_bytes(), expected_tiger_np_array()),
      ("tiger_png", tiger_png_bytes(), expected_tiger_png_np_array()),
    ]:
      img_bytes = input_data[0]
      nparray = _bytes_to_nparray(img_bytes)

      self.assertTrue(
        np.array_equal(nparray, expected), f"nparray != expected for {name}"
      )

  def test_encode(self):
    for name, input_data, expected_encoded in [
      ("original", original_image_bytes(), expected_original_encoded_bytes()),
      ("peppers", peppers_image_bytes(), expected_peppers_encoded_bytes()),
    ]:
      img_bytes = input_data[0]
      file_type = input_data[1]
      encoded_bytes = apply_watermark(
          img_bytes, file_type, watermark=WATERMARK)

      # self.assertTrue(np.array_equal(
      # encoded_bytes, expected_encoded), f"actual != expected for {name}")

      # write encoded bytes to file
      with open(f'../data/expected_encoded/encoded_{name}.{file_type.value}', 'wb') as f:
        f.write(encoded_bytes)
      watermark = decode_watermark(encoded_bytes, wm_length=len(WATERMARK) * 8)
      self.assertEqual(WATERMARK, watermark,
                       "watermark != expected for {name}")

  # def test_decode(self):
  #   for encoded_bytes in [
  #     expected_original_encoded_bytes(),
  #     expected_peppers_encoded_bytes(),
  #   ]:
  #     watermark = decode_watermark(encoded_bytes, wm_length=len(WATERMARK) * 8)
  #     self.assertEqual(WATERMARK, watermark)

  def test_encode_with_resize_for_social_media(self):
    for name, input_data, new_expected_size in [
      # Tall image is 3024 × 4032 originally
      ("tall", tall_bytes(), (1080, 1350)),
      # wide image is 6880 x 1776 px originally
      ("wide", wide_bytes(), (1079, 565)),
      # original image is 1920 x 1080 px originally
      ("original", original_image_bytes(), (1080, 607)),
    ]:
      img_bytes = input_data[0]
      file_type = input_data[1]
      encoded_bytes = apply_watermark(
          img_bytes, file_type, watermark=WATERMARK, resize_for_social_media=True)

      encoded_img = Image.open(io.BytesIO(encoded_bytes))

      width, height = encoded_img.size
      self.assertEqual(new_expected_size, (width, height))
