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
"""Autoregressive model using dilated causal convolutions, see: https://arxiv.org/abs/1609.03499."""
import dataclasses
from typing import Callable

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from causal_conv_arm.conv import CausalConv1D


@dataclasses.dataclass
class ResidualBlock(hk.Module):
  embed_dim: int
  res_channels: int
  skip_channels: int
  kernel_size: int = 2
  dilation_rate: int = 1
  name: None | str = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, xs):
    filter_net = CausalConv1D(output_channels=self.res_channels,
                              kernel_size=self.kernel_size,
                              dilation_rate=self.dilation_rate)
    f = filter_net(xs)

    gate_net = CausalConv1D(output_channels=self.res_channels,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate)
    g = gate_net(xs)
    h = jax.nn.tanh(f) * jax.nn.sigmoid(g)

    res_net = CausalConv1D(output_channels=self.embed_dim,
                           kernel_size=1,
                           dilation_rate=self.dilation_rate)
    res = res_net(h)

    skip_net = CausalConv1D(output_channels=self.skip_channels,
                            kernel_size=1,
                            dilation_rate=self.dilation_rate)
    skip = skip_net(h)

    return xs + res, skip


class PostProcessLayer(hk.Conv1D):

  def __init__(self, output_channels, activation_fn, **kwargs):
    super().__init__(
      output_channels=output_channels,
      kernel_shape=1,
      rate=1,
      padding='VALID',
      **kwargs)
    self.activation_fn = activation_fn

  def __call__(self, xs, *args, **kwargs):
    xs = self.activation_fn(xs)
    return super().__call__(xs, *args, **kwargs)


@dataclasses.dataclass
class ARM(hk.Module):

  n_classes: int = 256
  embed_dim: int = 64
  res_channels: int = 128
  skip_channels: int = 128
  kernel_size: int = 2
  n_groups: int = 3
  n_blocks_per_group: int = 8
  n_postprocess_layers: int = 2
  postprocess_activation_fn: Callable = jax.nn.relu
  eps: float = 1.0e-5
  name: None | str = None

  def __post_init__(self):
    super().__post_init__(name=self.name)

  def __call__(self, xs: chex.Array):
    batch_size, n_elements = xs.shape
    xs = jnp.pad(xs, ((0, 0), (1, 0)))
    xs = hk.Embed(vocab_size=self.n_classes, embed_dim=self.embed_dim)(xs)
    skip_acc = jnp.zeros(shape=(batch_size, n_elements + 1, self.skip_channels))
    for n in range(self.n_groups):
      with hk.name_scope(f"group_{n}"):
        for k in range(self.n_blocks_per_group):
          block = ResidualBlock(name=f"block_{k}",
                                embed_dim=self.embed_dim,
                                res_channels=self.res_channels,
                                skip_channels=self.skip_channels,
                                kernel_size=self.kernel_size,
                                dilation_rate=2 ** k)
          xs, skip = block(xs)
          skip_acc = skip_acc + skip
    postprocess = hk.Sequential([PostProcessLayer(name=f"postprocess_{n}",
                                                  output_channels=self.skip_channels if n < self.n_postprocess_layers - 1 else self.n_classes,
                                                  activation_fn=self.postprocess_activation_fn)
                                 for n in range(self.n_postprocess_layers)])
    xs = postprocess(skip_acc)
    xs = jax.nn.softmax(xs, axis=-1)
    xs = xs[:, :-1, :]
    return xs

  def loglikelihood(self, xs: chex.Array):
    ps = self(xs)
    ps = jnp.clip(ps, a_min=self.eps, a_max=1 - self.eps)
    one_hot = jax.nn.one_hot(xs, num_classes=ps.shape[-1])
    log_p = jnp.sum(one_hot * jnp.log(ps), axis=(1, 2))

    return log_p

  def sample(self, x_seed: chex.Array):

    def sample_pixel(xs, index):
      ps = self(xs)
      key = hk.next_rng_key()
      ys = jax.random.categorical(key, jnp.log(ps), axis=-1)
      xs = xs.at[:, index].set(ys[:, index])
      return xs, None

    xs, _ = jax.lax.scan(sample_pixel, x_seed, jnp.arange(x_seed.shape[-1], dtype=jnp.int32))

    return xs
