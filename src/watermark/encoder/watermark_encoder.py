
from max_dct.max_dct_encoder import EmbedMaxDct, DecodeMaxDct
import struct
import numpy as np


class WatermarkEncoder(object):
  def __init__(self, content=b''):
    seq = np.array([n for n in content], dtype=np.uint8)
    self._watermarks = list(np.unpackbits(seq))
    self._wmLen = len(self._watermarks)

  def get_length(self):
    return self._wmLen

  def max_dwt_encode(self, rgb: np.ndarray) -> np.ndarray:
    rows, columns, _ = rgb.shape

    if rows * columns < 256 * 256:
      raise RuntimeError(
          'image too small, should be larger than 256x256')

    embed = EmbedMaxDct(self._watermarks)
    return embed.encode_rgb(rgb)


class WatermarkDecoder(object):
  def __init__(self, wm_length=0):
    self._wmLen = wm_length

  def _reconstruct_bytes(self, bits):
    nums = np.packbits(bits)
    bstr = b''
    for i in range(self._wmLen // 8):
      bstr += struct.pack('>B', nums[i])
    return bstr

  def decode(self, rgb) -> bytes:
    rows, columns, _ = rgb.shape
    if rows * columns < 256 * 256:
      raise RuntimeError(
          'image too small, should be larger than 256x256')

    bits = []
    embed = DecodeMaxDct(wm_length=self._wmLen)
    bits = embed.decode_rgb(rgb)
    return self._reconstruct_bytes(bits)
