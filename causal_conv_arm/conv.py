#Copyright (c) 2025 Stefano Campanella
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Causal convolution layer."""

import chex
import haiku as hk
import jax.numpy as jnp


# Using haiku.Conv1D and haiku.pad one can do fancier stuff, like handling automatically the number of channels and
# proper padding. However, here we'll keep it simple.
class CausalConv1D(hk.Conv1D):
  """Causal, one-dimensional convolution (channel-last).

  This layer expects inputs in shape [N, W, C], where:
  - N: batch size
  - W: sequence length (time or width)
  - C: number of channels/features

  The layer applies left padding of length (kernel_size - 1) * dilation_rate to ensure causality and
  then performs a 1D convolution with VALID padding using Haiku's Conv1D (channel-last convention).
  """

  def __init__(self,
               output_channels: int,
               kernel_size: int,
               dilation_rate: int,
               **kwargs):
    """Initialize the causal convolution module.

    Args:
      output_channels: Number of output channels/features.
      kernel_size: Size of the 1D kernel (along the sequence dimension).
      dilation_rate: Dilation rate for the convolution (>= 1).
    """
    super().__init__(output_channels=output_channels,
                     kernel_shape=kernel_size,
                     rate=dilation_rate,
                     padding='VALID',
                     **kwargs)

    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate

  def __call__(self, xs: chex.Array, *args, **kwargs) -> chex.Array:
    """Applies the causal convolution.

    Args:
      xs: Input array of shape [N, W, C].

    Returns:
      Output array of shape [N, W, output_channels].
    """
    pad_len = (self.kernel_size - 1) * self.dilation_rate
    xs = jnp.pad(xs, ((0, 0), (pad_len, 0), (0, 0)))
    return super().__call__(xs, *args, **kwargs)
