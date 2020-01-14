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

"""Visualization module for entanglements in representation codes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from disentanglement_lib.utils import results
from tensorflow import gfile
import tensorflow_hub as hub
import gin.tf
import os
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.visualize import visualize_util
import itertools

import disentanglement_lib.evaluation.metrics.dci as dci
import disentanglement_lib.evaluation.metrics.sap_score as sap
import disentanglement_lib.evaluation.metrics.utils as utils


def visualize(model_dir,
              output_dir,
              overwrite=False):
    """Takes trained model from model_dir and visualizes it in output_dir.

    Args:
      model_dir: Path to directory where the trained model is saved.
      output_dir: Path to output directory.
      overwrite: Boolean indicating whether to overwrite output directory.
    """

    # Fix the random seed for reproducibility.
    random_state = np.random.RandomState(0)

    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.
    # Obtain the dataset name from the gin config of the previous step.

    module_paths = [os.path.join(model_dir, "postprocessed", "mean", "tfhub"),
                    os.path.join(model_dir, "postprocessed", "sampled", "tfhub")]
    modes = ['mean', 'sampled']

    results_dir = os.path.join(output_dir, "entanglement_properties")

    if tf.gfile.IsDirectory(results_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(results_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    if not gfile.IsDirectory(results_dir):
        gfile.MakeDirs(results_dir)

    gin_config_file = os.path.join(model_dir, "model", "results", "gin", "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    dataset = gin_dict["dataset.name"].replace("'", "")
    gin.bind_parameter("dataset.name", dataset)
    num_bins = 20
    gin.bind_parameter("correlation.active_correlation",
                       bool(gin_dict["correlation.active_correlation"] == "True"))
    if bool(gin_dict["correlation.active_correlation"] == "True"):
        gin.bind_parameter("correlation_details.corr_indices",
                           list(map(int, gin_dict["correlation_details.corr_indices"][1:-1].split(","))))
        gin.bind_parameter("correlation_details.corr_type", gin_dict["correlation_details.corr_type"].replace(
            "'", ""))
        if gin.query_parameter("correlation_details.corr_type") == "plane":
            gin.bind_parameter("correlation_hyperparameter.bias_plane",
                               float(gin_dict["correlation_hyperparameter.bias_plane"].replace("'", "")))
        elif gin.query_parameter("correlation_details.corr_type") == "line":
            gin.bind_parameter("correlation_hyperparameter.line_width",
                               float(gin_dict["correlation_hyperparameter.line_width"].replace("'", "")))

    for module_path, mode in zip(module_paths, modes):
        with hub.eval_function_for_module(module_path) as f:
            def _representation_function(x):
                """Computes representation vector for input images."""
                output = f(dict(images=x), signature="representation", as_dict=True)
                return np.array(output["default"])

            for correlation_mode, identifier in zip([True, False], ['', '_uc']):
                gin.bind_parameter("correlation.active_correlation", correlation_mode)

                ground_truth_data = named_data.get_named_ground_truth_data()

                num_train = 10000
                num_test = 5000
                batch_size = 16
                mus_train, ys_train = utils.generate_batch_factor_code(
                    ground_truth_data, _representation_function, num_train,
                    random_state, batch_size)
                mus_test, ys_test = utils.generate_batch_factor_code(
                    ground_truth_data, _representation_function, num_test,
                    random_state, batch_size)

                discretized_mus = histogram_discretize(mus_train, num_bins)

                results_dict = {}

                gbt_matrix, train_err_gbt, test_err_gbt = dci.compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
                sap_matrix = sap.compute_score_matrix(mus_train, ys_train, mus_test, ys_test, False)
                mi_matrix = utils.discrete_mutual_info(discretized_mus, ys_train)

                visualize_util.fov_latent_code_plot(gbt_matrix, os.path.join(results_dir, "gbt_matrix_" + mode + ".jpg"),
                                                    dataset)
                visualize_util.fov_latent_code_plot(sap_matrix, os.path.join(results_dir, "sap_matrix_" + mode + ".jpg"),
                                                    dataset)
                visualize_util.fov_latent_code_plot(mi_matrix, os.path.join(results_dir, "mi_matrix_" + mode + ".jpg"),
                                                    dataset)

                threshold_line, independent_groups, discovered_factors, distance_matrix, discovered_factors_dict = get_entanglement_curves(gbt_matrix)
                visualize_util.entanglement_curves_plot(threshold_line, independent_groups, discovered_factors, os.path.join(results_dir, "gbt_matrix_curve_" + mode + identifier + ".jpg"))
                visualize_util.plot_tree(threshold_line, distance_matrix, discovered_factors_dict, dataset, os.path.join(results_dir, "gbt_dendrogram_" + mode + identifier + ".jpg"))
                distance_threshold_gbt = get_thresholded_matrix(distance_matrix, threshold_line)
                results_dict['distance_threshold_gbt'] = distance_threshold_gbt
                results_dict['distance_matrix_gbt'] = list(distance_matrix)
                results_dict['thresholds_gbt'] = list(threshold_line)

                threshold_line, independent_groups, discovered_factors, distance_matrix, discovered_factors_dict = get_entanglement_curves(sap_matrix)
                visualize_util.entanglement_curves_plot(threshold_line, independent_groups, discovered_factors, os.path.join(results_dir, "sap_matrix_curve_" + mode + identifier + ".jpg"))
                visualize_util.plot_tree(threshold_line, distance_matrix, discovered_factors_dict, dataset, os.path.join(results_dir, "sap_dendrogram_" + mode + identifier + ".jpg"))
                distance_threshold_sap = get_thresholded_matrix(distance_matrix, threshold_line)
                results_dict['distance_threshold_sap'] = distance_threshold_sap
                results_dict['distance_matrix_sap'] = list(distance_matrix)
                results_dict['thresholds_sap'] = list(threshold_line)

                threshold_line, independent_groups, discovered_factors, distance_matrix, discovered_factors_dict = get_entanglement_curves(mi_matrix)
                visualize_util.entanglement_curves_plot(threshold_line, independent_groups, discovered_factors, os.path.join(results_dir, "mi_matrix_curve_" + mode + identifier + ".jpg"))
                visualize_util.plot_tree(threshold_line, distance_matrix, discovered_factors_dict, dataset, os.path.join(results_dir, "mi_dendrogram_" + mode + identifier + ".jpg"))
                distance_threshold_mig = get_thresholded_matrix(distance_matrix, threshold_line)
                results_dict['distance_threshold_mig'] = distance_threshold_mig
                results_dict['distance_matrix_mig'] = list(distance_matrix)
                results_dict['thresholds_mig'] = list(threshold_line)

                results_dict['evaluation_on_biased_dataset'] = correlation_mode
                results_dict['evaluation.name'] = "entanglement_task" + identifier

                # Save the results (and all previous results in the pipeline) on disk.
                original_results_dir = os.path.join(model_dir, "postprocessed", mode, "results")
                results_dir_aggregate = os.path.join(model_dir, "metrics", mode, "entanglement_task" + identifier, "results")
                results.update_result_directory(results_dir_aggregate, "evaluation", results_dict, original_results_dir)


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
                            other_clusters = [other_cluster_temp for other_cluster_temp in clusters_to_be_merged if other_cluster_temp != cluster]
                            for other_cluster in other_clusters:
                                for factor in cluster:
                                    for other_factor in other_cluster:
                                        min_factor = min(factor, other_factor)
                                        max_factor = max(factor, other_factor)
                                        index = pair_list.index((min_factor, max_factor))
                                        # distance_matrix[index] = max_threshold - threshold
                                        distance_matrix[index] = threshold_index

    return distance_matrix, discovered_factors_dict


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

    distance_matrix, discovered_factors_dict = get_distance_matrix(threshold_line, independent_group_constituents, factor_code_matrix.shape[1])

    return threshold_line, independent_groups_line, discovered_factors_line, distance_matrix, discovered_factors_dict


def histogram_discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized
