import unittest
import numpy as np
import cv2
from PIL import Image

from max_dct import max_dct_encoder


def bits_to_utf8(bits):
  # Convert the numpy array of bits into a string of '0' and '1' characters
  bits_string = ''.join(str(int(b)) for b in bits)

  # Split the string of bits into 8-bit chunks
  chunks = [bits_string[i:i + 8] for i in range(0, len(bits_string), 8)]

  # Convert each 8-bit chunk into a decimal value
  decimals = [int(chunk, 2) for chunk in chunks]

  # Convert the list of decimal values into a bytes object
  bytes_obj = bytes(decimals)

  # Decode the bytes object into a UTF-8 encoded string
  utf8_string = bytes_obj.decode('utf-8')

  return utf8_string


def original_image():
  img = cv2.imread("../data/original.jpg")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def peppers_image():
  img = cv2.imread("../data/peppers.png")
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def single_frame_test_data():
  return np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])


def one_embedder():
  return max_dct_encoder.EmbedMaxDct(watermarks=[1])


def zero_embedder():
  return max_dct_encoder.EmbedMaxDct(watermarks=[0])


def SDV2_embedder():
  wm = "SDV2".encode("utf-8")
  seq = np.array([n for n in wm], dtype=np.uint8)
  watermark = list(np.unpackbits(seq))
  return max_dct_encoder.EmbedMaxDct(watermarks=watermark)


def SDV2_decoder():
  return max_dct_encoder.DecodeMaxDct(wm_length=32)


class TestMaxDctEncode(unittest.TestCase):
  def test_encode_frame_with_one(self):
    encoder = one_embedder()
    test_data = single_frame_test_data()

    encoder.encode_frame(test_data, scale=4)

    # Expect lowest number to get embedded with n // 4 + 3
    expected = np.array([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 19],
    ])

    self.assertTrue(np.array_equal(test_data, expected))

  def test_encode_frame_with_zero(self):
    encoder = zero_embedder()
    test_data = single_frame_test_data()

    encoder.encode_frame(test_data, scale=4)

    # Expect highest number to get embedded with n // 4 + 1
    expected = np.array([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 17],
    ])

    self.assertTrue(np.array_equal(test_data, expected))

  def test_encoder_decoder(self):
    encoder = SDV2_embedder()
    decoder = SDV2_decoder()
    for image in [
      original_image(),
      peppers_image(),
    ]:
      encoded = encoder.encode_rgb(image)
      # Image.fromarray(np.uint8(encoded)).save("encoded_original.jpg", "JPEG")

      decoded = decoder.decode_rgb(encoded)
      self.assertEqual("SDV2", bits_to_utf8(decoded))
