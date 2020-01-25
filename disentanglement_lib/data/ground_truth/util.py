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

"""Various utilities used in the data set code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import tensorflow as tf
import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf
import math


@gin.configurable("correlation", blacklist=["factor_sizes", "latent_factor_indices"])
def get_state_space(factor_sizes, latent_factor_indices, active_correlation=False):
    if not active_correlation:
        return SplitDiscreteStateSpace(factor_sizes, latent_factor_indices)
    else:
        return CorrelatedSplitDiscreteStateSpace(factor_sizes, latent_factor_indices)


def tf_data_set_from_ground_truth_data(ground_truth_data, random_seed):
    """Generate a tf.data.DataSet from ground_truth data."""

    def generator():
        # We need to hard code the random seed so that the data set can be reset.
        random_state = np.random.RandomState(random_seed)
        while True:
            yield ground_truth_data.sample_observations(1, random_state)[0]

    return tf.data.Dataset.from_generator(
        generator, tf.float32, output_shapes=ground_truth_data.observation_shape)


class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors)
            if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state):
        """Sample a batch of the latent factors."""
        factors = np.zeros(
            shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(
            shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)

    def get_correlated_factors(self):
        return []



class CorrelatedSplitDiscreteStateSpace(SplitDiscreteStateSpace):
    """State space with two correlated latent factors.
  Pick two latent factor indices to be correlated
  Args:
    corr_indices: The two latent factor indices to correalte.
    corr_type: plane, ellipse, or line (cf. Chen et al 2018 section 6.1)
    corr_amount: Relative amount to corr the log probability.
  Raises:
    ValueError: if an invalid corr type or corr indices are provided.
  """

    @gin.configurable("correlation_details", blacklist=["factor_sizes", "latent_factor_indices"])
    def __init__(self, factor_sizes, latent_factor_indices, corr_indices=gin.REQUIRED,
                 corr_type=gin.REQUIRED):
        if corr_indices is None:
            corr_indices = [3, 4]
        lfi = latent_factor_indices
        super(CorrelatedSplitDiscreteStateSpace, self).__init__(factor_sizes, lfi)

        if len(corr_indices) != 2 or corr_indices[0] == corr_indices[1]:
            raise ValueError('Invalid corr indices given')

        for ci in corr_indices:
            if ci == 0:
                msg = 'Correlating the 0th factor (color) is not currently supported.'
                raise ValueError(msg)

            if not ci in latent_factor_indices:
                msg = 'Invalid corr_indices: one of the specified indices is not ' + \
                      'a member of the latent_factor_indices.'
                raise ValueError(msg)

        self.corr_indices = corr_indices
        self.corr_type = corr_type
        self.joint_prob = self._get_joint_prob()

    @gin.configurable("correlation_hyperparameter")
    def _get_joint_prob(self, bias_plane=None, line_width=None):
        corr_factor_sizes = [self.factor_sizes[self.corr_indices[0]], self.factor_sizes[self.corr_indices[1]]]

        if self.corr_type == 'plane':  # equivalent to Chen et al 2018 type B
            if bias_plane is None:
                raise ValueError("Invalid corr type.")
            else:
                unnormalized_marg_prob_0 = np.linspace(0., 1., corr_factor_sizes[0])
                unnormalized_marg_prob_0 += bias_plane
                unnormalized_marg_prob_1 = np.linspace(0., 1., corr_factor_sizes[1])
                unnormalized_marg_prob_1 += bias_plane
                unnormalized_joint_prob = np.outer(
                    unnormalized_marg_prob_0, unnormalized_marg_prob_1
            )

        elif self.corr_type == 'ellipse':  # equivalent to Chen et al 2018 type C
            import cv2

            for sfs in corr_factor_sizes:
                if sfs < 12:
                    from warnings import warn
                    msg = 'Ellipse corr type (C) not recommended for factors of variation ' + \
                          'with fewer than 12 categories.'
                    warn(msg)

            n_x, n_y = corr_factor_sizes

            unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.uint8)

            center_x = n_x // 2
            center_y = n_y // 2

            r_maj = n_x // 4
            r_min = n_y // 4

            width = 1  # this will get smoothed out later

            cv2.ellipse(unnormalized_joint_prob, (center_y, center_x), (r_maj, r_min), 90, 0, 360,
                        255, width)

            kernel_width = min(corr_factor_sizes) // 5
            if not kernel_width % 2:  # kernels widths must be odd
                kernel_width += 1

            unnormalized_joint_prob = cv2.GaussianBlur(
                unnormalized_joint_prob, (kernel_width, kernel_width), 0)
            unnormalized_joint_prob = unnormalized_joint_prob.astype(np.float_)

        elif self.corr_type == 'line':  # equivalent to Chen et al 2018 type D
            import cv2

            for sfs in corr_factor_sizes:
                if sfs < 12:
                    from warnings import warn
                    msg = 'Line corr type (C) not recommended for factors of variation ' \
                          + 'with fewer than 12 categories.'
                    warn(msg)

            # Create a black image
            unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.uint8)

            # width = 2  this will get smoothed out later
            if line_width >= 1.0: # choose uniform (independent) distribution if line_width >= 1
                unnormalized_joint_prob = np.ones(corr_factor_sizes)
            else:

                width = math.ceil(line_width * min(corr_factor_sizes))

                # offset = corr_factor_sizes[0] // 6
                offset = 0
                start = (0, offset)
                end = (corr_factor_sizes[1], corr_factor_sizes[0])
                # this is a speciality when correlating object type and azimuth in shaes 3d

                kernel_width = min(corr_factor_sizes) // 4
                if not kernel_width % 2:  # kernels widths must be odd
                    kernel_width += 1

                kernel_width_x = kernel_width
                kernel_width_y = kernel_width

                if corr_factor_sizes[1] == 4:
                    end = (corr_factor_sizes[1], corr_factor_sizes[0] + 5)
                    kernel_width_x = 1
                    kernel_width_y = kernel_width + 4


                cv2.line(unnormalized_joint_prob, start, end, 255, width)

                unnormalized_joint_prob = cv2.GaussianBlur(unnormalized_joint_prob, (kernel_width_x, kernel_width_y), 0)
                unnormalized_joint_prob = unnormalized_joint_prob.astype(np.float_)

        elif self.corr_type == 'power':  # based on Creager et al ICML 2019
            first_factor = 1.
            second_factor = 3.
            # NOTE: The order of the indices now matters since they are each raised to
            #       a different power in the unnormalized density.
            unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.float_)
            Ni, Nj = corr_factor_sizes[0], corr_factor_sizes[1]
            for i in range(Ni):
                for j in range(Nj):
                    unnormalized_joint_prob[i, j] = \
                        (float(i) / Ni) ** first_factor \
                        + (float(j) / Nj) ** second_factor

        else:
            raise ValueError("Invalid corr type.")

        # normalize
        joint_prob = unnormalized_joint_prob / unnormalized_joint_prob.sum()

        return joint_prob

    def _sample_correlated_factors(self, num, random_state):
        corr_factor_sizes = [self.factor_sizes[self.corr_indices[0]], self.factor_sizes[self.corr_indices[1]]]
        # corr_factor_sizes = self.factor_sizes[self.corr_indices]
        n_x, n_y = corr_factor_sizes
        pairs = np.indices(dimensions=(n_x, n_y))
        pairs = pairs.reshape(2, -1).T

        inds = random_state.choice(np.arange(n_x * n_y),
                                   p=self.joint_prob.reshape(-1),
                                   size=num, replace=True)
        samps = pairs[inds]

        return samps

    def sample_latent_factors(self, num, random_state):
        """Sample a batch of the latent factors."""
        factors = np.zeros(
            shape=(num, len(self.latent_factor_indices)), dtype=np.int64)

        correlated_samples = self._sample_correlated_factors(num, random_state)
        # figure out how to properly index the array factors
        idx = np.argwhere(np.isin(self.latent_factor_indices, self.corr_indices))
        idx = idx.flatten()
        factors[:, idx] = correlated_samples

        for pos, i in enumerate(self.latent_factor_indices):
            if not i in self.corr_indices:
                factors[:, pos] = self._sample_factor(i, num, random_state)

        return factors

    def get_correlated_factors(self):
        # this is a workaround to the problem that dsprites_full neglects color variable
        shift = len(self.factor_sizes) - len(self.latent_factor_indices)
        return [self.corr_indices[0] - shift, self.corr_indices[1] - shift]


class CorrelatedSplitDiscreteStateSpace(SplitDiscreteStateSpace):
  """State space with two correlated latent factors.

  Pick two latent factor indices to be correlated
  Args:
    corr_indices: The two latent factor indices to correalte.
    corr_type: plane, ellipse, or line (cf. Chen et al 2018 section 6.1)
    corr_amount: Relative amount to corr the log probability.

  Raises:
    ValueError: if an invalid corr type or corr indices are provided.
  """

  def __init__(self, factor_sizes, latent_factor_indices, corr_indices,
               corr_type):
    lfi = latent_factor_indices
    super(CorrelatedSplitDiscreteStateSpace, self).__init__(factor_sizes, lfi)

    if len(corr_indices) != 2 or corr_indices[0] == corr_indices[1]:
      raise ValueError('Invalid corr indices given.')

    for ci in corr_indices:
      if ci == 0:
        msg = 'Correlating the 0th factor (color) is not currently supported.'
        raise ValueError(msg)

      if not ci in latent_factor_indices:
        msg = 'Invalid corr_indices: one of the specified indices is not ' + \
          'a member of the latent_factor_indices.'
        raise ValueError(msg)

    self.corr_indices = corr_indices
    self.corr_type = corr_type
    self.joint_prob = self._get_joint_prob()

  def _get_joint_prob(self):
    corr_factor_sizes = self.factor_sizes[self.corr_indices]

    if self.corr_type == 'plane':  # equivalent to Chen et al 2018 type B
      bias = 0.5  # sets diff between min and max prob
      unnormalized_marg_prob_0 = np.linspace(0., 1., corr_factor_sizes[0])
      unnormalized_marg_prob_0 += bias
      unnormalized_marg_prob_1 = np.linspace(0., 1., corr_factor_sizes[1])
      unnormalized_marg_prob_1 += bias
      unnormalized_joint_prob = np.outer(
        unnormalized_marg_prob_0, unnormalized_marg_prob_1
      )

    elif self.corr_type == 'ellipse':  # equivalent to Chen et al 2018 type C
      import cv2

      for sfs in corr_factor_sizes:
        if sfs < 12:
          from warnings import warn
          msg = 'Ellipse corr type (C) not recommended for factors of variation ' + \
                'with fewer than 12 categories.'
          warn(msg)

      n_x, n_y = corr_factor_sizes

      unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.uint8)

      center_x = n_x // 2
      center_y = n_y // 2

      r_maj = n_x // 4
      r_min = n_y // 4

      width = 1  # this will get smoothed out later

      cv2.ellipse(unnormalized_joint_prob, (center_y, center_x), (r_maj, r_min), 90, 0, 360,
                  255, width)

      kernel_width = min(corr_factor_sizes) // 5
      if not kernel_width % 2:  # kernels widths must be odd
        kernel_width += 1

      unnormalized_joint_prob = cv2.GaussianBlur(
        unnormalized_joint_prob, (kernel_width, kernel_width),0)
      unnormalized_joint_prob = unnormalized_joint_prob.astype(np.float_)

    elif self.corr_type == 'line':  # equivalent to Chen et al 2018 type D
      import cv2

      for sfs in corr_factor_sizes:
        if sfs < 12:
          from warnings import warn
          msg = 'Line corr type (C) not recommended for factors of variation ' \
                + 'with fewer than 12 categories.'
          warn(msg)


      # Create a black image
      unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.uint8)

      width = 2  # this will get smoothed out later

      offset = corr_factor_sizes[0] // 6
      start = (0, offset)
      end = (corr_factor_sizes[1], corr_factor_sizes[0] - offset)

      cv2.line(unnormalized_joint_prob , start, end, 255, width)

      kernel_width = min(corr_factor_sizes) // 5
      if not kernel_width % 2:  # kernels widths must be odd
        kernel_width += 1

      unnormalized_joint_prob = cv2.GaussianBlur(unnormalized_joint_prob,(kernel_width, kernel_width),0)
      unnormalized_joint_prob = unnormalized_joint_prob.astype(np.float_)

    elif self.corr_type == 'power':  # based on Creager et al ICML 2019
      first_factor = 1.
      second_factor = 3.
      # NOTE: The order of the indices now matters since they are each raised to
      #       a different power in the unnormalized density.
      unnormalized_joint_prob = np.zeros(corr_factor_sizes, np.float_)
      Ni, Nj = corr_factor_sizes[0], corr_factor_sizes[1]
      for i in range(Ni):
        for j in range(Nj):
          unnormalized_joint_prob[i, j] = \
                  (float(i) / Ni) ** first_factor \
                  + (float(j) / Nj) ** second_factor

    else:
      raise ValueError("Invalid corr type.")

    # normalize
    joint_prob = unnormalized_joint_prob / unnormalized_joint_prob.sum()

    return joint_prob

  def _sample_correlated_factors(self, num, random_state):

    corr_factor_sizes = self.factor_sizes[self.corr_indices]
    n_x, n_y = corr_factor_sizes
    pairs = np.indices(dimensions=(n_x, n_y))
    pairs = pairs.reshape(2, -1).T

    inds = random_state.choice(np.arange(n_x * n_y),
                               p=self.joint_prob.reshape(-1),
                               size=num, replace=True)
    samps = pairs[inds]

    return samps

  def sample_latent_factors(self, num, random_state):
    """Sample a batch of the latent factors."""
    factors = np.zeros(
        shape=(num, len(self.latent_factor_indices)), dtype=np.int64)

    correlated_samples  = self._sample_correlated_factors(num, random_state)
    # figure out how to properly index the array factors
    idx = np.argwhere(np.isin(self.latent_factor_indices, self.corr_indices))
    idx = idx.flatten()
    factors[:, idx] = correlated_samples

    for pos, i in enumerate(self.latent_factor_indices):
      if not i in self.corr_indices:
        factors[:, pos] = self._sample_factor(i, num, random_state)

    return factors


class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """Creates the StateSpaceAtomIndex.

    Args:
      factor_sizes: List of integers with the number of distinct values for each
        of the factors.
      features: Numpy matrix where each row contains a different factor
        configuration. The matrix needs to cover the whole state space.
    """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.

    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.

    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
                np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)