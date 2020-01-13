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

"""Utility functions for the visualization code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from disentanglement_lib.utils import resources
import numpy as np
from PIL import Image
import scipy
from scipy.cluster import hierarchy
from six.moves import range
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import seaborn as sns


def save_image(image, image_path):
    """Saves an image in the [0,1]-valued Numpy array to image_path.

  Args:
    image: Numpy array of shape (height, width, {1,3}) with values in [0, 1].
    image_path: String with path to output image.
  """
    # Copy the single channel if we are provided a grayscale image.
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = np.ascontiguousarray(image)
    image *= 255.
    image = image.astype("uint8")
    with tf.gfile.Open(image_path, "wb") as path:
        img = Image.fromarray(image, mode="RGB")
        img.save(path)


def get_fov_labels_for_dataset(dataset):
    labels = []
    if dataset == 'dsprites_full':
        labels = ['shape', 'scale', 'orientation', 'position x', 'position y']
    elif dataset == 'shapes3d':
        labels = ['floor color', 'wall color', 'object color', 'object size', 'object type', 'azimuth']
    elif dataset == 'mpi3d_real':
        labels = ['Object color', 'Object shape', 'Object size', 'Camera height', 'Background colors', 'First DOF', 'Second DOF']
    return labels


def fov_latent_code_plot(matrix_encoding, image_path, dataset):
    """Generates an image of the matrix containing the relations between the factor of variation and latent codes

  Args:
    matrix_encoding: numpy array of shape (num_codes, num_factors)
    image_path: String with path to output image.
  """

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot

    x = np.zeros(matrix_encoding.shape[0] * matrix_encoding.shape[1])
    y = np.zeros(matrix_encoding.shape[0] * matrix_encoding.shape[1])
    size = np.zeros(matrix_encoding.shape[0] * matrix_encoding.shape[1])

    index = 0
    for factor_index in range(matrix_encoding.shape[1]):
        for code_index in range(matrix_encoding.shape[0]):
            x[index] = code_index
            y[index] = factor_index
            size[index] = matrix_encoding[code_index, factor_index]
            index += 1

    # Mapping from column names to integer coordinates
    x_labels = [str(v) for v in range(matrix_encoding.shape[0])]
    y_labels = get_fov_labels_for_dataset(dataset)
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    n_colors = 256  # Use 256 colors for the diverging color palette
    # palette = sns.palplot(sns.color_palette("Blues", n_colors))
    palette = sns.color_palette("Blues", n_colors=n_colors)
    color_min, color_max = [0, max(size)]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    ax.scatter(
        x=x,  # Use mapping for x
        y=y,  # Use mapping for y
        s=size * 500,  # Vector of square sizes, proportional to size parameter
        c=[value_to_color(size_item) for size_item in size],  # Vector of square colors, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )
    # Show column labels on the axes
    ax.set_xlabel('latent codes')
    ax.set_ylabel('factor of variation')
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_facecolor('white')  # Make background white
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right
    plt.savefig(image_path,bbox_inches='tight')
    plt.close()


def entanglement_curves_plot(threshold_line, independent_groups, discovered_factors, image_path):
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    plt.ylabel('Number of factors', fontsize=20)
    plt.xlabel('Threshold', fontsize=20)

    plt.yticks(fontsize=20)

    plt.plot(range(len(threshold_line)), discovered_factors, linestyle='dashed', color='orange', label='Discovered')
    plt.plot(range(len(threshold_line)), independent_groups, linestyle='solid', color='blue', label='Independent groups')
    ax.figure.tight_layout()

    tick_poistions = []
    tick_labels = []
    every_nth = int(len(threshold_line) / 5.)
    for i in range(len(threshold_line)):
        if i % every_nth == 0:
            tick_poistions.append(i)
            tick_labels.append(np.around(threshold_line[i], decimals=2).astype(str))

    plt.xticks(tick_poistions, labels=tick_labels, fontsize=20)
    plt.grid(True)
    plt.legend(loc='lower center', ncol=2, fontsize=20)
    plt.savefig(image_path)
    plt.close()


def plot_tree(threshold_line, distance_matrix, discovered_factors_dict, dataset, image_path):
    labels = np.array(get_fov_labels_for_dataset(dataset))
    plt.figure(figsize=(15, 10))
    Z = hierarchy.linkage(distance_matrix, 'single')
    dn = hierarchy.dendrogram(Z, labels=labels)

    plt.clf()
    ax = plt.gca()
    icoord = scipy.array(dn['icoord'])
    dcoord = scipy.array(dn['dcoord'])
    color_list = scipy.array(dn['color_list'])
    leave_list = scipy.array(dn['leaves'])
    ivl_list = scipy.array(dn['ivl'])
    ymin, ymax = icoord.min(), icoord.max()
    xmin, xmax = dcoord.min(), dcoord.max()
    for xs, ys, color in zip(icoord, dcoord, color_list,):
        for index, y in enumerate(ys):
            if y == 0:
                i = (xs[index] - 5.0) / 10.0
                if abs(i - int(i)) < 1e-5:
                    leave_list[int(i)]
                    if discovered_factors_dict[leave_list[int(i)]] > max(ys):
                        raise ValueError('discovered factors and independent groups not consistent')
                    else:
                        ys[index] = discovered_factors_dict[leave_list[int(i)]]
                else:
                    raise ValueError('could not figure out correct factor')
        plt.plot(ys, xs, color='b')
    plt.yticks(np.arange(start=5, stop=5 + 10*len(leave_list), step=10), labels=ivl_list, fontsize=20)
    plt.ylabel('Factors', fontsize=20)

    tick_poistions = []
    tick_labels = []
    every_nth = int(len(threshold_line) / 5.)
    for i in range(len(threshold_line)):
        if i % every_nth == 0:
            tick_poistions.append(i)
            tick_labels.append(np.around(threshold_line[i], decimals=2).astype(str))

    plt.xticks(tick_poistions, labels=tick_labels, fontsize=20)
    plt.xlabel('Threshold', fontsize=20)
    plt.ylim(ymin-10, ymax + 0.1*abs(ymax))
    plt.xlim(xmin, len(threshold_line))
    ax.figure.tight_layout()
    plt.grid(True)
    plt.savefig(image_path)
    plt.close()


def dendrogram_plot(threshold_line, distance_matrix, discovered_factors_dict, dataset, image_path):
    labels = np.array(get_fov_labels_for_dataset(dataset))
    Z = hierarchy.linkage(distance_matrix, 'single')
    plt.figure()
    ax = plt.gca()
    dn = hierarchy.dendrogram(Z, orientation='right', labels=labels)
    plt.ylabel('Factors')
    plt.xlabel('Threshold', fontsize=20)
    plt.xticks(range(len(threshold_line)), labels=np.around(threshold_line, decimals=2).astype(str), fontsize=20)
    plt.xlim(0, len(threshold_line))
    ax.figure.tight_layout()
    plt.locator_params(axis='x', nbins=5)
    plt.grid(True)
    plt.savefig(image_path)


def grid_save_images(images, image_path):
    """Saves images in list of [0,1]-valued np.arrays on a grid.

  Args:
    images: List of Numpy arrays of shape (height, width, {1,3}) with values in
      [0, 1].
    image_path: String with path to output image.
  """
    side_length = int(math.floor(math.sqrt(len(images))))
    image_rows = [
        np.concatenate(
            images[side_length * i:side_length * i + side_length], axis=0)
        for i in range(side_length)
    ]
    tiled_image = np.concatenate(image_rows, axis=1)
    save_image(tiled_image, image_path)


def padded_grid(images, num_rows=None, padding_px=10, value=None):
    """Creates a grid with padding in between images."""
    num_images = len(images)
    if num_rows is None:
        num_rows = best_num_rows(num_images)

    # Computes how many empty images we need to add.
    num_cols = int(np.ceil(float(num_images) / num_rows))
    num_missing = num_rows * num_cols - num_images

    # Add the empty images at the end.
    all_images = images + [np.ones_like(images[0])] * num_missing

    # Create the final grid.
    rows = [padded_stack(all_images[i * num_cols:(i + 1) * num_cols], padding_px,
                         1, value=value) for i in range(num_rows)]
    return padded_stack(rows, padding_px, axis=0, value=value)


def padded_stack(images, padding_px=10, axis=0, value=None):
    """Stacks images along axis with padding in between images."""
    padding_arr = padding_array(images[0], padding_px, axis, value=value)
    new_images = [images[0]]
    for image in images[1:]:
        new_images.append(padding_arr)
        new_images.append(image)
    return np.concatenate(new_images, axis=axis)


def padding_array(image, padding_px, axis, value=None):
    """Creates padding image of proper shape to pad image along the axis."""
    shape = list(image.shape)
    shape[axis] = padding_px
    if value is None:
        return np.ones(shape, dtype=image.dtype)
    else:
        assert len(value) == shape[-1]
        shape[-1] = 1
        return np.tile(value, shape)


def best_num_rows(num_elements, max_ratio=4):
    """Automatically selects a smart number of rows."""
    best_remainder = num_elements
    best_i = None
    i = int(np.sqrt(num_elements))
    while True:
        if num_elements > max_ratio * i * i:
            return best_i
        remainder = (i - num_elements % i) % i
        if remainder == 0:
            return i
        if remainder < best_remainder:
            best_remainder = remainder
            best_i = i
        i -= 1


def pad_around(image, padding_px=10, axis=None, value=None):
    """Adds a padding around each image."""
    # If axis is None, pad both the first and the second axis.
    if axis is None:
        image = pad_around(image, padding_px, axis=0, value=value)
        axis = 1
    padding_arr = padding_array(image, padding_px, axis, value=value)
    return np.concatenate([padding_arr, image, padding_arr], axis=axis)


def add_below(image, padding_px=10, value=None):
    """Adds a footer below."""
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, 2)
    if image.shape[2] != 3:
        raise ValueError("Could not convert image to have three channels.")
    with tf.gfile.Open(resources.get_file("disentanglement_lib.png"), "rb") as f:
        footer = np.array(Image.open(f).convert("RGB")) * 1.0 / 255.
    missing_px = image.shape[1] - footer.shape[1]
    if missing_px < 0:
        return image
    if missing_px > 0:
        padding_arr = padding_array(footer, missing_px, axis=1, value=value)
        footer = np.concatenate([padding_arr, footer], axis=1)
    return padded_stack([image, footer], padding_px, axis=0, value=value)


def save_animation(list_of_animated_images, image_path, fps):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(
            pad_around(add_below(padded_grid(list(single_images)))))
    imageio.mimwrite(image_path, full_size_images, fps=fps)


def cycle_factor(starting_index, num_indices, num_frames):
    """Cycles through the state space in a single cycle."""
    grid = np.linspace(starting_index, starting_index + 2 * num_indices,
                       num=num_frames, endpoint=False)
    grid = np.array(np.ceil(grid), dtype=np.int64)
    grid -= np.maximum(0, 2 * grid - 2 * num_indices + 1)
    grid += np.maximum(0, -2 * grid - 1)
    return grid


def cycle_gaussian(starting_value, num_frames, loc=0., scale=1.):
    """Cycles through the quantiles of a Gaussian in a single cycle."""
    starting_prob = scipy.stats.norm.cdf(starting_value, loc=loc, scale=scale)
    grid = np.linspace(starting_prob, starting_prob + 2.,
                       num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    grid = np.minimum(grid, 0.999)
    grid = np.maximum(grid, 0.001)
    return np.array([scipy.stats.norm.ppf(i, loc=loc, scale=scale) for i in grid])


def cycle_interval(starting_value, num_frames, min_val, max_val):
    """Cycles through the state space in a single cycle."""
    starting_in_01 = (starting_value - min_val) / (max_val - min_val)
    grid = np.linspace(starting_in_01, starting_in_01 + 2.,
                       num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val
