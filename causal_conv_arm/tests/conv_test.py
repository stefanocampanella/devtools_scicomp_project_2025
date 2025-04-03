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
def dilation(request):
  return request.param


@pytest.fixture(scope="module", params=[True, False])
def depend_on_current_token(request):
  return request.param


@pytest.fixture
def key(seed):
  return jax.random.key(seed)


def get_conv(input_channels, output_channels, kernel_size, dilation, depend_on_current_token, **kwargs):
  @hk.without_apply_rng
  @hk.transform
  def _conv(xs):
    conv_layer = conv.CausalConv(input_channels,
                                   output_channels,
                                   kernel_size,
                                   dilation=dilation,
                                   A=depend_on_current_token,
                                   **kwargs)
    return conv_layer(xs)

  return _conv


def test_shape(key,
               num_elements,
               batch_size,
               input_channels,
               output_channels,
               kernel_size,
               dilation,
               depend_on_current_token):
  A = depend_on_current_token
  inputs = jnp.empty((batch_size, input_channels, num_elements))
  inputs_unbatched = jnp.empty((input_channels, num_elements))
  conv_fn = get_conv(input_channels, output_channels, kernel_size, dilation, A)
  params = conv_fn.init(key, inputs)
  outputs = conv_fn.apply(params, inputs)
  outputs_unbatched = conv_fn.apply(params, inputs_unbatched)

  assert outputs_unbatched.shape == (output_channels, num_elements - A)
  assert outputs.shape == (batch_size, output_channels, num_elements - A)


def test_ndim(key,
              num_elements,
              batch_size,
              input_channels,
              output_channels,
              kernel_size,
              dilation,
              depend_on_current_token):
  A = depend_on_current_token
  inputs = jnp.empty((batch_size, input_channels, num_elements))
  inputs_londims = jnp.empty((1,))
  inputs_hindims = jnp.empty((1, 1, 1, 1))
  conv = get_conv(input_channels, output_channels, kernel_size, dilation, A)
  params = conv.init(key, inputs)

  with pytest.raises(ValueError):
    conv.apply(params, inputs_londims)

  with pytest.raises(ValueError):
    conv.apply(params, inputs_hindims)


def test_edge_kernel(key,
                     batch_size,
                     num_elements):
  input_channels = 1
  output_channels = 1
  kernel_size = 2
  dilation = 1
  A = False
  inputs = jax.random.uniform(key, (batch_size, input_channels, num_elements))
  conv = get_conv(input_channels, output_channels, kernel_size, dilation, A, name="cconv")
  params = {'cconv': {'w': jnp.tile(jnp.array([-1.0, 1.0]), (output_channels, input_channels, 1))}}
  outputs = conv.apply(params, inputs)
  expected_outputs = jnp.diff(jnp.pad(inputs, ((0, 0), (0, 0), (1, 0))), axis=-1)
  chex.assert_trees_all_equal(outputs, expected_outputs)
