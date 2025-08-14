import chex
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import jax.numpy as jnp
import matplotlib.pyplot as plt
from math import ceil, sqrt


class SequentialMNIST(Dataset):
  """MNIST as 1D sequences of integer pixel values.

  This dataset wraps torchvision.datasets.MNIST and returns each 28x28 image
  as a flattened 1D JAX array of length up to 784 (values in [0, 255]).

  Notes:
    - Labels are intentionally discarded; this dataset is for unconditional
      sequence modeling of pixel intensities.
    - The maximum number of returned elements can be limited via `n_elements`.
  """

  def __init__(self, root: str, train: bool = True, download: bool = True, n_elements: int = 784):
    """Initialize the SequentialMNIST dataset.

    Args:
      root: Filesystem path where the MNIST data will be stored/read from.
      train: If True, use the training split; otherwise the test split.
      download: If True, download the dataset if it is not found in `root`.
      n_elements: Maximum number of elements to return per sample (<= 784).
        Values greater than 784 are clipped to 784.

    Returns:
      None. Initializes internal torchvision dataset and configuration.
    """
    transform = transforms.Compose([
      lambda xs: jnp.array(xs, dtype=jnp.int32),
      lambda xs: jnp.ravel(xs)])
    self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    self.n_elements = min(784, n_elements)

  def __len__(self):
    """Number of samples in the split.

    Returns:
      int: The number of items in the underlying MNIST split.
    """
    return len(self.mnist)

  def __getitem__(self, idx):
    """Get a flattened MNIST image as a 1D sequence.

    Args:
      idx: Integer index of the sample.

    Returns:
      jax.numpy.ndarray: Shape (n,), dtype=int32, where n = min(784, n_elements).
        Pixel values are integers in [0, 255].
    """
    xs, _ = self.mnist[idx]
    return xs


def plot_grid(samples: chex.Array, cmap='gray', shape=(28, 28), dpi: int = 200, **kwargs):
  """Plot a batch of images in a square grid and return the Matplotlib figure.

  Args:
    samples: Array-like of images to plot. Can be JAX/NumPy array or similar.
      Expected to be of shape (B, H, W) or can be reshaped to (-1, H, W) using
      `shape`. Values are displayed as-is.
    cmap: Colormap used by imshow (default: 'gray').
    shape: Target (H, W) used to reshape each sample if input is flat or
      otherwise needs reshaping.
    dpi: Dots per inch for the created figure.
    **kwargs: Additional keyword arguments forwarded to `fig.add_subplot`.

  Returns:
    matplotlib.figure.Figure: The created figure with images arranged in a grid.

  Notes:
    - The grid is arranged as an nrows x nrows square where nrows = ceil(sqrt(B)).
    - Axes are turned off for a cleaner visualization.
  """
  samples = np.array(samples)
  samples = samples.reshape((-1,) + shape)
  batch_size, _, _ = samples.shape
  fig = plt.figure(dpi=dpi)
  nrows = ceil(sqrt(batch_size))
  for i in range(batch_size):
    sample = samples[i, :, :]
    ax = fig.add_subplot(nrows, nrows, i + 1, **kwargs)
    ax.imshow(sample, cmap=cmap, interpolation='none', resample=False)
    ax.axis('off')

  return fig