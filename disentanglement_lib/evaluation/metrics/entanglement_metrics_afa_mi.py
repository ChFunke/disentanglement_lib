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
import disentanglement_lib.evaluation.metrics.sap_score as sap
import numpy as np
import itertools

import sklearn
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import gin.tf


@gin.configurable(
    "entanglement_metrics_afa_mi",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_entanglement_metrics_afa(ground_truth_data, representation_function, random_state,
                                     artifact_dir=None,
                                     num_train=gin.REQUIRED,
                                     num_test=gin.REQUIRED,
                                     random_seed_fast_adaptation_step=gin.REQUIRED,
                                     num_labels_available=gin.REQUIRED,
                                     batch_size=16):
    """Computes the entanglement metrics after fast adaptation.

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
        adjusted_representation_function = get_fast_adapted_representation_function(ground_truth_data,
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

        # Compute GBT importance matrix, SVM predictablity matrix and MI matrix
        gbt_matrix, train_err, test_err = compute_importance_gbt(
            mus_train, ys_train, mus_test, ys_test)
        assert gbt_matrix.shape[0] == mus_train.shape[0]
        assert gbt_matrix.shape[1] == ys_train.shape[0]

        num_bins = 20
        discretized_mus = histogram_discretize(mus_train, num_bins)

        sap_matrix = sap.compute_score_matrix(mus_train, ys_train, mus_test, ys_test, False)
        mi_matrix = utils.discrete_mutual_info(discretized_mus, ys_train)

        gbt_threshold_line, gbt_independent_groups, gbt_discovered_factors, gbt_distance_matrix, gbt_discovered_factors_dict = get_entanglement_curves(
            gbt_matrix)
        sap_threshold_line, sap_independent_groups, sap_discovered_factors, sap_distance_matrix, sap_discovered_factors_dict = get_entanglement_curves(
            sap_matrix)
        mi_threshold_line, mi_independent_groups, mi_discovered_factors, mi_distance_matrix, mi_discovered_factors_dict = get_entanglement_curves(
            mi_matrix)

        gbt_threshold_matrix = get_thresholded_matrix(gbt_distance_matrix, gbt_threshold_line)
        sap_threshold_matrix = get_thresholded_matrix(sap_distance_matrix, sap_threshold_line)
        mi_threshold_matrix = get_thresholded_matrix(mi_distance_matrix, sap_threshold_line)

        size_string = str(num_labels)

        scores[size_string + ":gbt_threshold_line"] = list(gbt_threshold_line)
        scores[size_string + ":gbt_independent_groups"] = list(gbt_independent_groups)
        scores[size_string + ":gbt_discovered_factors"] = list(gbt_discovered_factors)
        scores[size_string + ":gbt_distance_matrix"] = list(gbt_distance_matrix)
        scores[size_string + ":gbt_discovered_factors_dict"] = {int(k): v for k, v in gbt_discovered_factors_dict.items()}
        scores[size_string + ":gbt_threshold_matrix"] = list(gbt_threshold_matrix)

        scores[size_string + ":sap_threshold_line"] = list(sap_threshold_line)
        scores[size_string + ":sap_independent_groups"] = list(sap_independent_groups)
        scores[size_string + ":sap_discovered_factors"] = list(sap_discovered_factors)
        scores[size_string + ":sap_distance_matrix"] = list(sap_distance_matrix)
        scores[size_string + ":sap_discovered_factors_dict"] = {int(k): v for k, v in sap_discovered_factors_dict.items()}
        scores[size_string + ":sap_threshold_matrix"] = list(sap_threshold_matrix)

        scores[size_string + ":mi_threshold_line"] = list(mi_threshold_line)
        scores[size_string + ":mi_independent_groups"] = list(mi_independent_groups)
        scores[size_string + ":mi_discovered_factors"] = list(mi_discovered_factors)
        scores[size_string + ":mi_distance_matrix"] = list(mi_distance_matrix)
        scores[size_string + ":mi_discovered_factors_dict"] = {int(k): v for k, v in mi_discovered_factors_dict.items()}
        scores[size_string + ":mi_threshold_matrix"] = list(mi_threshold_matrix)

    scores['evaluation.name'] = "entanglement_metrics_afa"
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


def get_fast_adapted_representation_function(ground_truth_data, representation_function, random_state_for_few_labels,
                                             num_labels_available):
    if num_labels_available == 0:
        return representation_function
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

        num_bins = 20
        discretized_mus = histogram_discretize(x_train, num_bins)

        importance_matrix = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):
                importance_matrix[i, j] = sklearn.metrics.mutual_info_score(y_train[j, :], discretized_mus[i, :])

        # Select dimensions based on mid-range value (max_val + min_val)/2
        dimension_threshold = (np.amax(importance_matrix) + np.amin(importance_matrix)) / 2
        entangled_code_dimensions = [dimension for dimension in range(num_codes) if
                                     np.any(importance_matrix[dimension, :] > dimension_threshold)]
        if len(entangled_code_dimensions) < num_corr_factors:
            max_codes = np.array([np.amax(importance_matrix[i, :]) for i in range(num_codes)])
            entangled_code_dimensions = max_codes.argsort()[-num_corr_factors:]

        # Disentangle the features by doing linear regression using the observations (train data)
        X = np.transpose(x_train)
        y = np.zeros((X.shape[1], X.shape[0]))
        y[entangled_code_dimensions[0:len(corr_factors)], :] = y_train[corr_factors, :]
        y = np.transpose(y)
        reg = LinearRegression().fit(X, y)

        def _fast_adapted_representation_function(x):
            representations = representation_function(x)
            adapted_dimension_encodings = reg.predict(representations)
            representations[:, entangled_code_dimensions[0:len(corr_factors)]] = adapted_dimension_encodings[:,
                                                                                 entangled_code_dimensions[
                                                                                 0:len(corr_factors)]]
            return representations

        return _fast_adapted_representation_function


def histogram_discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def get_entanglement_curves(factor_code_matrix):
    threshold_line = np.unique(np.sort(factor_code_matrix, axis=None))[::-1]

    independent_group_constituents = []

    independent_groups_line = np.zeros(len(threshold_line))
    discovered_factors_line = np.zeros(len(threshold_line))
    for threshold_index, threshold in enumerate(threshold_line):
        connectivity_matrix = (factor_code_matrix > threshold).astype(int)
        discovered_factors = (np.sum(connectivity_matrix, axis=0, dtype=np.float32) > 0).sum()
        independent_groups = 0
        independent_group_constituent = []
        while np.sum(connectivity_matrix) > 0:
            factor = (np.sum(connectivity_matrix, axis=0, dtype=np.float32) > 0).argmax()
            connectivity_matrix, deleted_edges, group_factors = delete_all_connected_edges(connectivity_matrix, factor,
                                                                                           is_factor=True)
            if deleted_edges >= 1:
                independent_groups += 1
                independent_group_constituent.append(group_factors)
        independent_group_constituents.append(independent_group_constituent)
        independent_groups_line[threshold_index] = independent_groups
        discovered_factors_line[threshold_index] = discovered_factors

    distance_matrix, discovered_factors_dict = get_distance_matrix(threshold_line, independent_group_constituents,
                                                                   factor_code_matrix.shape[1])

    return threshold_line, independent_groups_line, discovered_factors_line, distance_matrix, discovered_factors_dict


def get_thresholded_matrix(distance_matrix, threshold_line):
    thresholded_matrix = []
    for index in range(len(distance_matrix)):
        thresholded_matrix.append(threshold_line[int(distance_matrix[index])])

    return thresholded_matrix


def delete_all_connected_edges(factor_code_matrix, vertex, is_factor):
    group_factors = []
    if is_factor:
        group_factors.append(vertex)
        connected_codes = list(np.where(factor_code_matrix[:, vertex] == 1)[0])
        deleted_edges = len(connected_codes)
        if len(connected_codes) > 0:
            factor_code_matrix[:, vertex] = 0
            for code in connected_codes:
                factor_code_matrix, deleted_edges_local, group_factors_local = delete_all_connected_edges(
                    factor_code_matrix, vertex=code, is_factor=False)
                group_factors += group_factors_local
                deleted_edges += deleted_edges_local
    else:
        connected_factors = list(np.where(factor_code_matrix[vertex, :] == 1)[0])
        deleted_edges = len(connected_factors)
        if len(connected_factors) > 0:
            factor_code_matrix[vertex, :] = 0
            for factor in connected_factors:
                factor_code_matrix, deleted_edges_local, group_factors_local = delete_all_connected_edges(
                    factor_code_matrix, vertex=factor, is_factor=True)
                group_factors += group_factors_local
                deleted_edges += deleted_edges_local
    return factor_code_matrix, deleted_edges, list(set(group_factors))


def get_distance_matrix(threshold_line, independent_group_constituents, num_factors):
    discovered_factors_dict = {}
    pair_list = list(itertools.combinations(range(num_factors), 2))
    distance_matrix = np.full(len(pair_list), 10000.0)
    max_threshold = max(threshold_line)
    for threshold_index, threshold in enumerate(threshold_line):
        for igc_clusters in independent_group_constituents[threshold_index]:
            for factor in igc_clusters:
                if factor not in discovered_factors_dict.keys():
                    discovered_factors_dict[factor] = threshold_index
        if threshold_index > 0:
            for igc_cluster in independent_group_constituents[threshold_index]:
                if igc_cluster not in independent_group_constituents[threshold_index - 1]:
                    clusters_to_be_merged = []
                    for igc_cluster_prev in independent_group_constituents[threshold_index - 1]:
                        prev_cluster_in_new_cluster = all(elem in igc_cluster for elem in igc_cluster_prev)
                        if prev_cluster_in_new_cluster:
                            clusters_to_be_merged.append(igc_cluster_prev)
                    if len(clusters_to_be_merged) == 1:
                        factor_list = [fac for fac in igc_cluster if fac not in clusters_to_be_merged[0]]
                        for factor in factor_list:
                            for other_factor in clusters_to_be_merged[0]:
                                min_factor = min(factor, other_factor)
                                max_factor = max(factor, other_factor)
                                index = pair_list.index((min_factor, max_factor))
                                # distance_matrix[index] = max_threshold - threshold
                                distance_matrix[index] = threshold_index
                            for other_factor in [fac for fac in factor_list if fac != factor]:
                                min_factor = min(factor, other_factor)
                                max_factor = max(factor, other_factor)
                                index = pair_list.index((min_factor, max_factor))
                                # distance_matrix[index] = max_threshold - threshold
                                distance_matrix[index] = threshold_index
                    if len(clusters_to_be_merged) == 2:
                        for factor in clusters_to_be_merged[0]:
                            for other_factor in clusters_to_be_merged[1]:
                                min_factor = min(factor, other_factor)
                                max_factor = max(factor, other_factor)
                                index = pair_list.index((min_factor, max_factor))
                                # distance_matrix[index] = max_threshold - threshold
                                distance_matrix[index] = threshold_index
                    elif len(clusters_to_be_merged) > 2:
                        for cluster in clusters_to_be_merged:
                            other_clusters = [other_cluster_temp for other_cluster_temp in clusters_to_be_merged if
                                              other_cluster_temp != cluster]
                            for other_cluster in other_clusters:
                                for factor in cluster:
                                    for other_factor in other_cluster:
                                        min_factor = min(factor, other_factor)
                                        max_factor = max(factor, other_factor)
                                        index = pair_list.index((min_factor, max_factor))
                                        # distance_matrix[index] = max_threshold - threshold
                                        distance_matrix[index] = threshold_index

    return distance_matrix, discovered_factors_dict
