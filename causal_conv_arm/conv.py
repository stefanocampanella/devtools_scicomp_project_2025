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
import jax


# Using haiku.Conv1D and haiku.pad one can do fancier stuff, like handling automatically the number of channels and
# proper padding. However, here we'll keep it simple.
class CausalConv(hk.Module):
  """Causal, one-dimensional convolution."""

  def __init__(self,
               output_channels: int,
               kernel_size: int,
               dilation_rate:int,
               depends_on_current_token=False,
               w_init:hk.initializers.Initializer | None = None,
               **kwargs):
    """Initialize the module

    Args:
      in_channels: number of input channels/features
      output_channels: number of output channels/features
      kernel_size: kernel size
      dilation_rate: dilation rate
      depends_on_current_token: dependency on current token.
    """
    super().__init__(**kwargs)

    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.depends_on_current_token = depends_on_current_token
    self.w_init = w_init or hk.initializers.RandomNormal()

  def __call__(self, inputs: chex.Array, **kwargs) -> chex.Array:
    """Connects ``CausalConv1D`` layer.

    Args:
      inputs: An array of shape ``[C, W]``, or ``[N, C, W]``, where ``C`` is the number of channels/features, ``W``
      the size along the (unique) spatial dimension, and ``N`` the batch size.

    Returns:
      An array of shape ``[C, W]`` or ``[N, C, W]``.
    """
    if inputs.ndim < 2:
      raise ValueError('inputs must have at least 2 dimensions')
    elif inputs.ndim == 2:
      inputs = jax.lax.expand_dims(inputs, (0,))
      unbatched = True
    elif inputs.ndim == 3:
      unbatched = False
    else:
      raise ValueError('inputs must have shape (C, W) or (N, C, W)')

    input_channels = inputs.shape[-2]
    kernel = hk.get_parameter("w",
                              shape=(self.output_channels, input_channels, self.kernel_size),
                              init=self.w_init)
    padding = (self.kernel_size - 1) * self.dilation_rate + self.depends_on_current_token * 1
    inputs = jax.lax.pad(inputs, padding_value=0., padding_config=[(0, 0, 0), (0, 0, 0), (padding, 0, 0)])
    outputs = jax.lax.conv_general_dilated(inputs,
                                           kernel,
                                           rhs_dilation=(self.dilation_rate,),
                                           window_strides=(1,),
                                           padding=[(0, 0)],
                                           **kwargs)

    if unbatched:
      outputs = jax.lax.squeeze(outputs, (0,))

    if self.depends_on_current_token:
      outputs = outputs[..., :-1]

    return outputs
