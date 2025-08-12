# Copyright (c) 2025 Stefano Campanella
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import pytest

from causal_conv_arm import conv

pytestmark = pytest.mark.parametrize("seed", [42])


@pytest.fixture(scope="module", params=[1, 32])
def batch_size(request):
  return request.param


@pytest.fixture(scope="module", params=[1, 2])
def input_channels(request):
  return request.param


@pytest.fixture(scope="module", params=[1, 2])
def output_channels(request):
  return request.param


@pytest.fixture(scope="module", params=[1, 4])
def kernel_size(request):
  return request.param


@pytest.fixture(scope="module", params=[1, 32])
def num_elements(request):
  return request.param


@pytest.fixture(scope="module", params=[1, 4])
def dilation_rate(request):
  return request.param


@pytest.fixture
def key(seed):
  return jax.random.key(seed)


def get_conv(output_channels, kernel_size, dilation_rate, **kwargs):
  @hk.without_apply_rng
  @hk.transform
  def _conv(xs):
    conv_layer = conv.CausalConv1D(output_channels=output_channels,
                                   kernel_size=kernel_size,
                                   dilation_rate=dilation_rate,
                                   **kwargs)
    return conv_layer(xs)

  return _conv


def test_shape(key,
               num_elements,
               batch_size,
               input_channels,
               output_channels,
               kernel_size,
               dilation_rate):
  # Channel-last format: [N, W, C]
  inputs = jnp.empty((batch_size, num_elements, input_channels))
  conv_fn = get_conv(output_channels, kernel_size, dilation_rate)
  params = conv_fn.init(key, inputs)
  outputs = conv_fn.apply(params, inputs)

  assert outputs.shape == (batch_size, num_elements, output_channels)


def test_ndim(key,
              num_elements,
              batch_size,
              input_channels,
              output_channels,
              kernel_size,
              dilation_rate):
  inputs = jnp.empty((batch_size, num_elements, input_channels))
  inputs_londims = jnp.empty((1,))
  inputs_hindims = jnp.empty((1, 1, 1, 1))
  conv_fn = get_conv(output_channels, kernel_size, dilation_rate)
  params = conv_fn.init(key, inputs)

  with pytest.raises(Exception):
    conv_fn.apply(params, inputs_londims)

  with pytest.raises(Exception):
    conv_fn.apply(params, inputs_hindims)


def test_edge_kernel(key,
                     batch_size,
                     num_elements):
  input_channels = 1
  output_channels = 1
  kernel_size = 2
  dilation = 1
  inputs = jax.random.uniform(key, (batch_size, num_elements, input_channels))
  conv_fn = get_conv(output_channels, kernel_size, dilation, name="cconv")
  # Initialize params then overwrite Conv1D kernel weights to implement a simple difference filter
  params = conv_fn.init(key, inputs)
  w = jnp.array([[-1.0], [1.0]])  # shape (kernel, in_channels)
  params['cconv']['w'] = w.reshape(kernel_size, input_channels, output_channels)
  outputs = conv_fn.apply(params, inputs)
  expected_outputs = jnp.diff(jnp.pad(inputs, ((0, 0), (1, 0), (0, 0))), axis=1)
  chex.assert_trees_all_equal(outputs, expected_outputs)
