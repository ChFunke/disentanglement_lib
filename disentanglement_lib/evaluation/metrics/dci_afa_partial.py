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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import gin.tf


@gin.configurable(
    "dci_afa_partial",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_dci(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                random_seed_fast_adaptation_step=gin.REQUIRED,
                num_labels_available=gin.REQUIRED,
                batch_size=16):
    """Computes the DCI scores according to Sec 2.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    random_seed_fast_adaptation_step: random seed for the sampling of the labeled data
    num_labels_available: Number of points available as labeled data
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with average disentanglement score, completeness and
      informativeness (train and test).
  """
    scores = {}
    del artifact_dir
    for num_labels in num_labels_available:
        random_state_for_few_labels = np.random.RandomState(random_seed_fast_adaptation_step)
        adjusted_representation_function, scores_fit = get_fast_adapted_representation_function(ground_truth_data,
                                                                                    representation_function,
                                                                                    random_state_for_few_labels,
                                                                                    num_labels)

        logging.info("Generating training set.")
        # mus_train are of shape [num_codes, num_train], while ys_train are of shape
        # [num_factors, num_train].
        mus_train, ys_train = utils.generate_batch_factor_code(
            ground_truth_data, adjusted_representation_function, num_train,
            random_state, batch_size)
        assert mus_train.shape[1] == num_train
        assert ys_train.shape[1] == num_train
        mus_test, ys_test = utils.generate_batch_factor_code(
            ground_truth_data, adjusted_representation_function, num_test,
            random_state, batch_size)

        # Compute DCI score
        importance_matrix, train_err, test_err = compute_importance_gbt(
            mus_train, ys_train, mus_test, ys_test)
        assert importance_matrix.shape[0] == mus_train.shape[0]
        assert importance_matrix.shape[1] == ys_train.shape[0]

        size_string = str(num_labels)
        scores[size_string + ':mean_squared_error'] = scores_fit['mean_squared_error']
        scores[size_string + ':r2_score'] = scores_fit['r2_score']
        scores[size_string + ':mean_absolute_error'] = scores_fit['mean_absolute_error']

        scores[size_string + ":informativeness_train"] = train_err
        scores[size_string + ":informativeness_test"] = test_err
        scores[size_string + ":disentanglement"] = disentanglement(importance_matrix)
        scores[size_string + ":completeness"] = completeness(importance_matrix)

    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def get_fast_adapted_representation_function(ground_truth_data, representation_function, random_state_for_few_labels,
                                             num_labels_available):
    scores = {}

    if num_labels_available == 0:
        scores['mean_squared_error'] = 0.0
        scores['r2_score'] = 0.0
        scores['mean_absolute_error'] = 0.0
        return representation_function, scores
    else:
        x_train, y_train = utils.generate_batch_factor_code(
            ground_truth_data, representation_function, num_labels_available,
            random_state_for_few_labels, batch_size=16)
        assert x_train.shape[1] == num_labels_available
        assert y_train.shape[1] == num_labels_available

        corr_factors = ground_truth_data.state_space.get_correlated_factors()
        num_corr_factors = len(corr_factors)
        if num_corr_factors < 2:
            raise ValueError('This dataset has no correlated pair of factors of variation')

        # Produce GBT feature importance only for the correlated variables
        num_factors = num_corr_factors
        num_codes = x_train.shape[0]
        importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                     dtype=np.float64)
        for i in range(num_factors):
            model = GradientBoostingClassifier()
            model.fit(x_train.T, y_train[corr_factors[i], :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)

        # Select dimensions
        max_codes = np.array([np.amax(importance_matrix[i, :]) for i in range(num_codes)])
        entangled_code_dimensions = max_codes.argsort()[-num_corr_factors:]

        # Disentangle the features by doing linear regression using the observations (train data)
        X = np.transpose(x_train[entangled_code_dimensions, :])
        y = y_train[corr_factors, :]
        y.resize(X.shape[1], X.shape[0], refcheck=False)
        y = np.transpose(y)
        reg = LinearRegression().fit(X, y)

        y_pred = reg.predict(X)
        scores['mean_squared_error'] = mean_squared_error(y, y_pred)
        scores['r2_score'] = r2_score(y, y_pred)
        scores['mean_absolute_error'] = mean_absolute_error(y, y_pred)

        def _fast_adapted_representation_function(x):
            representations = representation_function(x)
            old_dimension_encodings = representations[:, entangled_code_dimensions]
            adapted_dimension_encodings = reg.predict(old_dimension_encodings)
            representations[:, entangled_code_dimensions] = adapted_dimension_encodings
            return representations

        return _fast_adapted_representation_function, scores
