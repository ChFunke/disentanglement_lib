# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import PIL
from six.moves import range
from tensorflow import gfile


DSPRITES_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
SCREAM_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "scream", "scream.jpg")



class DSprites(ground_truth_data.GroundTruthData):
  """DSprites dataset.

  The data set was originally introduced in "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework" and can be downloaded from
  https://github.com/deepmind/dsprites-dataset.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    if latent_factor_indices is None:
      latent_factor_indices = list(range(6))
    self.latent_factor_indices = latent_factor_indices
    self.data_shape = [64, 64, 1]
    # Load the data so that we can sample from it.
    with gfile.Open(DSPRITES_PATH, "rb") as data_file:
      # Data was saved originally using python2, so we need to set the encoding.
      data = np.load(data_file, encoding="latin1", allow_pickle=True)
      self.images = np.array(data["imgs"])
      self.factor_sizes = np.hstack((np.array(
          data["metadata"][()]["latents_sizes"], dtype=np.int64), [6]))
    self.full_factor_sizes = [1, 3, 6, 40, 32, 32, 6]
    self.factor_bases = np.prod(self.factor_sizes[:6]) / np.cumprod(
        self.factor_sizes[:6])
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    return self.sample_observations_from_factors_no_color(factors, random_state)

  def sample_observations_from_factors_no_color(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors[:, :6], self.factor_bases), dtype=np.int64)
    return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.factor_sizes[i], size=num)


class ColorDSprites(DSprites):
  """Color DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the sprite is colored in a randomly sampled
  color.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None):
    DSprites.__init__(self, latent_factor_indices)
    self.data_shape = [64, 64, 3]

  def sample_observations_from_factors(self, factors, random_state):
    no_color_observations = self.sample_observations_from_factors_no_color(
        factors, random_state)
    observations = np.repeat(no_color_observations, 3, axis=3)
    color = np.repeat(
        np.repeat(
            random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]),
            observations.shape[1],
            axis=1),
        observations.shape[2],
        axis=2)
    return observations * color


class BackgroundColorDSprites(DSprites):
  """BackgroundColor DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the background is colored in one of six randomly sampled
  colors.

  The ground-truth factors of variation are (in the default setting):
  0 - colour (1 different values)
  1 - shape (3 different values)
  2 - scale (6 different values)
  3 - orientation (40 different values)
  4 - position x (32 different values)
  5 - position y (32 different values)
  6 - background color (6 different values)
  """

  def __init__(self, latent_factor_indices=None, col='six_random'):
    DSprites.__init__(self, latent_factor_indices)
    self.data_shape = [64, 64, 3]
    self.col = col

  def sample_observations_from_factors(self, factors, random_state):
    no_color_observations = self.sample_observations_from_factors_no_color(
        factors, random_state)
    observations = np.repeat(no_color_observations, 3, axis=3)
    
    if self.col == 'six_random':
        colors = np.array([[0.64030151, 0.6411753 , 0.66388316],
           [0.92666881, 0.95607118, 0.89676185],
           [0.99438935, 0.82573291, 0.96938611],
           [0.8245052 , 0.57488259, 0.58252164],
           [0.60907252, 0.96265136, 0.7916103 ],
           [0.69036633, 0.96064992, 0.71655041]])
        color_label = factors[:, self.latent_factor_indices.index(6)]
    elif self.col == 'random':
        colors = np.array([random_state.uniform(0.5, 1, 3)])
        color_label = 0
    else:
        colors = np.array([self.col])
        color_label = 0

    color = np.repeat(
        np.repeat(
            colors[color_label].reshape([1, 1, 1, 3]),
            observations.shape[1],
            axis=1),
        observations.shape[2],
        axis=2)
    return 1 - (1 - observations) * color


class NoisyDSprites(DSprites):
  """Noisy DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the background pixels are replaced with random
  noise.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None):
    DSprites.__init__(self, latent_factor_indices)
    self.data_shape = [64, 64, 3]

  def sample_observations_from_factors(self, factors, random_state):
    no_color_observations = self.sample_observations_from_factors_no_color(
        factors, random_state)
    observations = np.repeat(no_color_observations, 3, axis=3)
    color = random_state.uniform(0, 1, [observations.shape[0], 64, 64, 3])
    return np.minimum(observations + color, 1.)


class ScreamDSprites(DSprites):
  """Scream DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, a random patch of the Scream image is sampled as
  the background and the sprite is embedded into the image by inverting the
  color of the sampled patch at the pixels of the sprite.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None):
    DSprites.__init__(self, latent_factor_indices)
    self.data_shape = [64, 64, 3]
    with gfile.Open(SCREAM_PATH, "rb") as f:
      scream = PIL.Image.open(f)
      scream.thumbnail((350, 274, 3))
      self.scream = np.array(scream) * 1. / 255.

  def sample_observations_from_factors(self, factors, random_state):
    no_color_observations = self.sample_observations_from_factors_no_color(
        factors, random_state)
    observations = np.repeat(no_color_observations, 3, axis=3)

    for i in range(observations.shape[0]):
      x_crop = random_state.randint(0, self.scream.shape[0] - 64)
      y_crop = random_state.randint(0, self.scream.shape[1] - 64)
      background = (self.scream[x_crop:x_crop + 64, y_crop:y_crop + 64] +
                    random_state.uniform(0, 1, size=3)) / 2.
      mask = (observations[i] == 1)
      background[mask] = 1 - background[mask]
      observations[i] = background
    return observations


class CorrelatedDSprites(DSprites):
  """Same as DSprites, but two of the latent factors correlate."""
  def __init__(self, latent_factor_indices=None, corr_indices=[3, 4],
               corr_type='plane'):
    super(CorrelatedDSprites, self).__init__(latent_factor_indices)
    self.state_space = util.CorrelatedSplitDiscreteStateSpace(
      self.factor_sizes, self.latent_factor_indices, corr_indices, corr_type)


class CorrelatedColorDSprites(CorrelatedDSprites, ColorDSprites):
  """Same as ColorDSprites, but two of the latent factors correlate."""
  def __init__(self, latent_factor_indices=None, corr_indices=[3, 4],
               corr_type='plane'):
    super(CorrelatedColorDSprites, self).__init__(latent_factor_indices,
                                                  corr_indices, corr_type)

class CorrelatedBackgroundColorDSprites(CorrelatedDSprites, BackgroundColorDSprites):
  """Same as ColorDSprites, but two of the latent factors correlate."""
  def __init__(self, latent_factor_indices=None, corr_indices=[3, 4],
               corr_type='plane'):
    super(CorrelatedBackgroundColorDSprites, self).__init__(latent_factor_indices,
                                                  corr_indices, corr_type)

class CorrelatedNoisyDSprites(CorrelatedDSprites, NoisyDSprites):
  """Same as NoisyDSprites, but two of the latent factors correlate."""
  def __init__(self, latent_factor_indices=None, corr_indices=[3, 4],
               corr_type='plane'):
    super(CorrelatedNoisyDSprites, self).__init__(latent_factor_indices,
                                                  corr_indices, corr_type)


class CorrelatedScreamDSprites(CorrelatedDSprites, ScreamDSprites):
  """Same as ScreamDSprites, but two of the latent factors correlate."""
  def __init__(self, latent_factor_indices=None, corr_indices=[3, 4],
               corr_type='plane'):
    super(CorrelatedScreamDSprites, self).__init__(latent_factor_indices,
                                                   corr_indices, corr_type)

