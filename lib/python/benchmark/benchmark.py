import io
import time
from PIL import Image

from watermark.apply_watermark import apply_watermark, Filetype


"""
Resize an image from
700 x 700
960 x 960
1080 x 1080
1200 x 1200
1440 x 1440
1920 x 1920
2048 x 2048
2880 x 2880
3840 x 3840
4096 x 4096

# Test how long it takes to encode an image
"""

SIZES = [
  700,
  960,
  1080,
  1200,
  1440,
  1920,
  2048,
  2880,
  3840,
  4096
]


def peppers_image_bytes(size: int):
  with Image.open('../data/peppers.png') as img:
    img = img.resize((size, size))

  img_bytes = io.BytesIO()
  img.save(img_bytes, format="png")
  return img_bytes.getvalue()


def run():
  # img_bytes = peppers_image_bytes()
  for size in SIZES:
    img = peppers_image_bytes(size)

    apply_watermark(img, Filetype.PNG)


if __name__ == "__main__":
  run()
