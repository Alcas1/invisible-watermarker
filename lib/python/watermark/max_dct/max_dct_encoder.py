import numpy as np
import pywt

from color_conversion import color_conversion


class EmbedMaxDct(object):
  def __init__(self, watermarks=[], scales=[0, 36, 36], block=4):
    self._watermarks = watermarks
    self._wmLen = len(watermarks)
    self._scales = scales
    self._block = block

  def encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
    yuv = color_conversion.rgb_to_yuv(rgb)
    encoded = self.encode_yuv(yuv)
    return color_conversion.yuv_to_rgb(encoded)

  def encode_yuv(self, yuv: np.ndarray) -> np.ndarray:
    rows, columns, _ = yuv.shape

    for channel in range(2):
      if self._scales[channel] <= 0:
        continue

      last_processed_row = rows // self._block * self._block
      last_processed_col = columns // self._block * self._block

      ca1, (h1, v1, d1) = pywt.dwt2(
          yuv[:last_processed_row, :last_processed_col, channel], 'haar')

      self.encode_frame(ca1, scale=self._scales[channel])

      yuv[:last_processed_row, :last_processed_col, channel, ] = pywt.idwt2(
          (ca1, (v1, h1, d1)), 'haar')

    return yuv

  def encode_frame(self, frame, scale):
    '''
    frame is a matrix (M, N)

    we get K (watermark bits size) blocks (self._block x self._block)

    For i-th block, we encode watermark[i] bit into it
    '''
    (row, col) = frame.shape
    num = 0

    for i in range(row // self._block):
      for j in range(col // self._block):
        block = frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block]
        wmBit = self._watermarks[(num % self._wmLen)]

        diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)

        frame[i * self._block: i * self._block + self._block,
              j * self._block: j * self._block + self._block] = diffusedBlock

        num = num + 1

  def diffuse_dct_matrix(self, block, wmBit, scale):
    """
    To embed a 1, add 0.75
    To embed a 0, add 0.25
    """
    pos = np.argmax(abs(block.flatten()[1:])) + 1
    i, j = pos // self._block, pos % self._block

    val = block[i][j]

    if val >= 0.0:
      block[i][j] = (val // scale + 0.25 + (0.5 * wmBit)) * scale
    else:
      val = abs(val)
      block[i][j] = -1.0 * (val // scale + 0.25 + (0.5 * wmBit)) * scale
    return block


class DecodeMaxDct(object):
  def __init__(self, wm_length, scales=[0, 36, 36], block=4):
    self._wmLen = wm_length
    self._scales = scales
    self._block = block

  def decode_rgb(self, rgb: np.ndarray) -> np.ndarray:
    rows, columns, __name__ = rgb.shape

    yuv = color_conversion.rgb_to_yuv(rgb)

    scores = [[] for i in range(self._wmLen)]
    for channel in range(2):
      if self._scales[channel] <= 0:
        continue

      last_processed_row = rows // self._block * self._block
      last_processed_col = columns // self._block * self._block

      ca1, (_, _, _) = pywt.dwt2(
          yuv[:last_processed_row, :last_processed_col, channel], 'haar')

      scores = self.decode_frame(ca1, self._scales[channel], scores)

    avgScores = list(map(lambda l: np.array(l).mean(), scores))

    bits = (np.array(avgScores) * 255 > 127)
    return bits

  def decode_frame(self, frame, scale, scores):
    (row, col) = frame.shape
    num = 0

    for i in range(row // self._block):
      for j in range(col // self._block):
        block = frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block]

        wmBit = num % self._wmLen
        score = self.infer_dct_matrix(block, scale)

        scores[wmBit].append(score)
        num = num + 1

    return scores

  def infer_dct_matrix(self, block, scale):
    pos = np.argmax(abs(block.flatten()[1:])) + 1
    i, j = pos // self._block, pos % self._block

    val = block[i][j]
    if val < 0:
      val = abs(val)

    if (val % scale) > 0.5 * scale:
      return 1
    else:
      return 0
