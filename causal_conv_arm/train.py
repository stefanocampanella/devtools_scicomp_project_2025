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
"""Training script for the ARM model."""
import functools
import logging
import pathlib
import sys
import tomllib

import click
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from causal_conv_arm.arm import ARM
from causal_conv_arm.utils import SequentialMNIST, plot_grid


@click.command()
@click.argument("config_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, file_okay=True, readable=True))
@click.argument("data_path",
                required=True,
                type=click.Path(path_type=pathlib.Path, dir_okay=True, readable=True))
@click.option('--jit/--no-jit', default=False, is_flag=True)
@click.option('--download/--no-download', default=False, is_flag=True)
@click.option('--log-level',
              default='info',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical'], case_sensitive=False))
def main(config_path: pathlib.Path,
         data_path: pathlib.Path,
         jit: bool = True,
         download: bool = False,
         log_level: str = 'info'):

  logging.basicConfig(stream=sys.stdout,
                      format='%(levelname)s - %(asctime)s: %(message)s',
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      level=getattr(logging, log_level.upper()))

  with config_path.open('rb') as file:
    config = tomllib.load(file)

  train_ds = SequentialMNIST(root=str(data_path), train=True, download=download, n_elements=config['n_elements'])
  train_dl = DataLoader(
    train_ds,
    collate_fn=lambda samples: jnp.stack(samples, axis=0),
    generator=torch.Generator().manual_seed(config['seed']),
    **config['train_dataloader'])

  validation_ds = SequentialMNIST(root=str(data_path), train=False, download=download, n_elements=config['n_elements'])

  @hk.without_apply_rng
  @hk.transform
  def loglikelihood(xs):
    model = ARM(**config['model'])
    return model.loglikelihood(xs)

  def loss_fn(params, xs):
    log_p = -loglikelihood.apply(params, xs)
    return jnp.mean(log_p)

  grads_fn = jax.value_and_grad(loss_fn)

  @hk.transform
  def sample(x_seed):
    model = ARM(**config['model'])
    return model.sample(x_seed)

  sample_fn = sample.apply

  def evaluate_fn(params):
    validation_array = jnp.stack([xs for xs in validation_ds], axis=0)
    validation_array = jnp.expand_dims(validation_array, axis=-2)
    _, log_p = jax.lax.scan(lambda _, xs: (None, loss_fn(params, xs)), None, validation_array)
    return jnp.mean(log_p)

  if jit:
    grads_fn = jax.jit(grads_fn)
    sample_fn = jax.jit(sample_fn)
    evaluate_fn = jax.jit(evaluate_fn)

  key = jax.random.key(config['seed'])
  params = loglikelihood.init(key, jnp.empty((1, config['n_elements']), dtype=jnp.int32))

  optimizer = optax.adam(**config['optimizer'])
  opt_state = optimizer.init(params)

  ckpt_path = data_path / "checkpoints"
  ckpt_path = ckpt_path.resolve()
  ckpt_path = ocp.test_utils.erase_and_create_empty(ckpt_path)

  # Check how to save only best model
  ckpt_mngr_options = ocp.CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: -1 * metrics['validation_nll'])
  ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_mngr_options)

  writer = SummaryWriter()

  best_nll = jnp.inf
  patience = 0
  global_step = 0
  sample_seed = jnp.full((config['generation_batch_size'], config['n_elements']), 0)

  for epoch in range(config['epochs']):

    validation_nll = evaluate_fn(params)

    logging.info(f"Epoch {epoch}, validation NLL {validation_nll}")
    ckpt_mngr.save(epoch, args=ocp.args.StandardSave(params), metrics={'validation_nll': validation_nll.item()})
    writer.add_scalar('nll/validation', validation_nll.item(), epoch)

    key, subkey = jax.random.split(key)
    generated_samples = sample_fn(params, subkey, sample_seed)
    writer.add_figure('samples', plot_grid(generated_samples), epoch)

    if validation_nll < best_nll:
      best_nll = validation_nll
      patience = 0
    else:
      patience += 1

    if patience > config['patience']:
      logging.info(f"Patience {patience} reached, stopping training")
      break

    for xs in tqdm(train_dl):
      value, grads = grads_fn(params, xs)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      writer.add_scalar('nll/train', value.item(), global_step)
      global_step += 1


if __name__ == '__main__':
  main()