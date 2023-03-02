from enum import Enum
from typing import BinaryIO
from PIL import Image
import numpy as np
import io
import logging
import time

from encoder.watermark_encoder import WatermarkEncoder, WatermarkDecoder

# set logging level to info
logging.basicConfig(level=logging.INFO)


class Filetype(Enum):
  JPEG = "jpeg"
  PNG = "png"
  GIF = "gif"
  UNKNOWN = "unknown"


def apply_watermark(
  img_buffer: BinaryIO,
  content_length: int,
  file_type: Filetype = Filetype.PNG,
  jpeg_quality: int = 75,
  watermark: str = "SDV2",
  resize_for_social_media: bool = False,
):
  """
  jpeg_quality: 0-100, will be applied only if file_type is JPEG, otherwise ignored. Determines the 
    quality of the encoded image that is sent back.
  """
  if jpeg_quality < 0 or jpeg_quality > 100:
    raise ValueError("jpeg_quality must be between 0 and 100")

  img_bytes = img_buffer.read(content_length)
  # Convert image bytes to numpy array
  img = _bytes_to_nparray(img_bytes, resize_for_social_media)

  # Encode watermark into image
  wm_encoder = WatermarkEncoder(watermark.encode('utf-8', 'replace'))

  encoded_img = wm_encoder.max_dwt_encode(img)
  # Convert numpy array to image bytes
  encoded_img = Image.fromarray(encoded_img.astype(np.uint8), 'RGB')

  encoded_img_bytes_png = io.BytesIO()
  encoded_img_bytes_jpg = io.BytesIO()

  encoded_img.save(encoded_img_bytes_jpg, format=Filetype.JPEG.value,
                   quality=jpeg_quality, subsampling=0)
  encoded_img.save(encoded_img_bytes_png, format=Filetype.PNG.value)

  return encoded_img_bytes_jpg, encoded_img_bytes_png


def decode_watermark(encoded_img_buffer: io.BytesIO, wm_length=32) -> str:
  encoded_img_bytes = encoded_img_buffer.getvalue()

  # Convert image bytes to numpy array
  encoded_img = _bytes_to_nparray(encoded_img_bytes)

  # Decode watermark from image
  wm_decoder = WatermarkDecoder(wm_length=wm_length)
  watermark = wm_decoder.decode(encoded_img)
  decoded = watermark.decode('utf-8', 'replace')
  return decoded


def _crop_if_necessary(img: np.array) -> np.ndarray:
  # Get the original aspect ratio
  width, height = img.size
  original_aspect = width / height

  # These are the minimum and maximum aspect ratios for Instagram
  min_aspect = 4 / 5
  max_aspect = 1.91

  # Image is too tall, so we need to crop the top and bottom
  if original_aspect < min_aspect:
    new_height = int(width / min_aspect)
    left = 0
    right = width
    top = int((height - new_height) / 2)
    bottom = int((height + new_height) / 2)
    img = img.crop((left, top, right, bottom))

  # Image is too wide, so we need to crop the left and right
  elif original_aspect > max_aspect:
    new_width = int(height * max_aspect)
    left = int((width - new_width) / 2)
    right = int((width + new_width) / 2)
    top = 0
    bottom = height
    img = img.crop((left, top, right, bottom))

  return img


def _resize_for_social_media(img: np.ndarray) -> np.ndarray:
  """
  Resize image to fit Instagram's aspect ratio requirements.
  Twitter has looser requirements, but we'll use the same ones for now.
  Ignoring Facebook. Fb has very specific requirements on size, so most images uploaded to FB 
    will get resized. We suspect most AI generated images will get resized and lose the watermark
    So intentionally having a watermark on FB may actually get used as a signal that the image is not generated.
  """
  img = _crop_if_necessary(img)
  width, height = img.size

  # Instagram wants images to have width between 320 and 1080 px
  if width < 320:
    img = img.resize((320, int(320 / width * height)))
  elif width > 1080:
    img = img.resize((1080, int(height / (width / 1080))))

  # In case the aspect ratio is off after resizing rounding, crop again
  return _crop_if_necessary(img)


def _bytes_to_nparray(bytes: bytes, resize_for_social_media: bool = False) -> np.array:
  start_time = time.time()
  # Convert image bytes to PIL image object
  img = Image.open(io.BytesIO(bytes))

  width, height = img.size
  # logging.info(
  #   f"image size: {width} x {height} = {(width * height):,} pixels")
  pixels = width * height

  if resize_for_social_media:
    img = _resize_for_social_media(img)

  if pixels > 4096 * 4096:
    raise ValueError("Image is too large. Max size is 4096 x 4096 pixels.")
  if pixels < 256 * 256:
    raise ValueError("Image is too small. Should be larger than 256x256 ")

  # preprocess_time = time.time()
  # logging.info(
  #   f"time to handle pre-processing: {preprocess_time - start_time}")

  # logging.info(f"image mode: {img.mode}")
  if img.mode != 'RGB':
    img = img.convert('RGB')

  # convert_time = time.time()
  # logging.info(f"time to convert to RGB:  {convert_time - preprocess_time}")

  # Convert PIL image object to numpy array
  return np.asarray(img)
