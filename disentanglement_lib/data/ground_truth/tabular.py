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

"""Tabular datasets."""
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
import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf

# NOTE: the UCI Adult dataset train and test splits should be located at
# the following paths
ADULT_TRAIN_PATH = os.path.join(
  os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "tabular", "adult",
  "adult_train.npz")

ADULT_TEST_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "tabular", "adult",
    "adult_test.npz")


class Adult(ground_truth_data.GroundTruthData):
  """Adult dataset.

  The UCI Adult dataset.

  This is a real-world tabular dataset with no known ground truth factors of
  variation (FoV). For the purposes of our study we consider the (unobserved)
  sensitive attribute and target label as FoV and draw the remaining (observed)
  features conditioned on the choice of sensitive and target labels. The FoV
  are formatted as follows:
  0 - sex (2 different values; 0 indicates Female and 1 indicates Male)
  1 - income > 50k (2 different values)
  """

  validation_size = 500  # a few train data used as FoV labels for adaptation

  @gin.configurable("adult_details", blacklist=["latent_factor_indices"])
  def __init__(self, latent_factor_indices=None, split=gin.REQUIRED,
               seed=gin.REQUIRED):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    if split not in ('train', 'test', 'validation'):
      raise ValueError('Invalid split %s. Supported choices are {train, test, '
                       'validation}' % split)
    train_data = np.load(ADULT_TRAIN_PATH)
    test_data = np.load(ADULT_TEST_PATH)
    # Seed sets which portion of train data set aside as validation set.
    npr = np.random.RandomState(seed)
    if split in ('train', 'validation'):
      idx = np.arange(len(train_data['x']))
      npr.shuffle(idx)
    else:
      idx = np.arange(len(test_data['x']))
      npr.shuffle(idx)

    # Load the data so that we can sample from it.
    if split == 'train':
      self.x = train_data['x'][idx][:-self.validation_size]
      self.y = train_data['y'][idx][:-self.validation_size]
      self.a = train_data['a'][idx][:-self.validation_size]
    if split == 'validation':
      self.x = train_data['x'][idx][-self.validation_size:]
      self.y = train_data['y'][idx][-self.validation_size:]
      self.a = train_data['a'][idx][-self.validation_size:]
    if split == 'test':
      self.x = test_data['x'][idx]
      self.y = test_data['y'][idx]
      self.a = test_data['a'][idx]

    if latent_factor_indices is None:
      latent_factor_indices = list(range(3))
    self.latent_factor_indices = latent_factor_indices
    # NOTE: several features are one-hot encoded (> 1 dimension each)
    num_feature_dims = len(self.x.T)
    self.data_shape = [num_feature_dims]
    self.factor_sizes = [len(self.x), 2, 2]
    self.full_factor_sizes = [len(self.x), 2, 2]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)
    self.state_space = util.get_state_space(self.factor_sizes,
                                            self.latent_factor_indices,
                                            data_dict={
                                              'a': self.a, 'y': self.y,
                                              'x': self.x
                                            })


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
    x_idx = factors[:, 0]
    return self.x[x_idx]

  def _sample_factor(self, i, num, random_state):
    factors = self.sample_factors(num, random_state)
    return factors[:, i]



