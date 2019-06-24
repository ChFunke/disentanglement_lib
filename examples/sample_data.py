# coding=utf-8

from absl import app
from absl import flags
import math
import os
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import gfile
from tensorflow.contrib.gan import eval
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
import gin.tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
  'gin_file', None, 'Path of config file.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')


base_path = "sample_data_output"
num_batches = 10


def plot_joint_prob(joint_prob, filename):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.imshow(joint_prob)
  plt.colorbar()
  plt.savefig(filename)
  plt.close('all')


def write_batch_as_jpg(batch, filename, pad=1):

#  import pdb
#  pdb.set_trace()


  # pad with zeros to make more visible
  batch = tf.pad(batch, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]),
                 constant_values=1.)

  batch_size = len(batch)
  grid_shape = int(math.sqrt(batch_size)), int(math.sqrt(batch_size))
  image_shape = batch.shape[1:3]
  num_channels = batch.shape[-1]

  batch = eval.image_grid(batch, grid_shape=grid_shape, image_shape=image_shape,
                          num_channels=num_channels)
  batch = tf.image.convert_image_dtype(batch, tf.uint8)
  batch = tf.squeeze(batch, 0)

  with gfile.Open(filename, 'wb+') as f:
    encoded_batch = tf.io.encode_jpeg(batch, name='batch')
    f.write(encoded_batch.numpy())

  return


@gin.configurable
def get_dataset_loader(batch_size, seed):
  dataset = named_data.get_named_ground_truth_data()
  loader = util.tf_data_set_from_ground_truth_data(dataset, seed)
  loader = loader.batch(batch_size, drop_remainder=True)
  return loader, dataset


def get_dirname():
  name = gin.query_parameter('dataset.name')
  if 'correlated' in name:
    try:
      corr_type = gin.query_parameter('dataset.corr_type')
      corr_indices = gin.query_parameter('dataset.corr_indices')
    except:
      msg = 'For correlated datasets the corr_type and corr_indices must' + \
        'be explicitly passed via the gin config file or overrides.'
      raise ValueError(msg)
    dirname = os.path.join(base_path, name, corr_type, str(corr_indices))
  else:
    dirname = os.path.join(base_path, name)

  return dirname


def main(unused_argv):
  gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

  dset_iter, dset = get_dataset_loader()
  dset_iter = iter(dset_iter)

  dirname = get_dirname()
  if not gfile.Exists(dirname):
    gfile.MakeDirs(dirname)

  for i in range(num_batches):
    batch = next(dset_iter)
    filename = os.path.join(dirname, 'batch{}.jpg'.format(i))
    write_batch_as_jpg(batch, filename)

  if 'correlated' in gin.query_parameter('dataset.name'):
    joint_prob = dset.state_space.joint_prob
    first_marg_prob = joint_prob.sum(1, keepdims=True)  # marginalize 2nd factor
    second_marg_prob = joint_prob.sum(0, keepdims=True)  # marginalize 1st fctor
    first_marg_basename = 'corr_marg_prob{}.jpg'.format(
      dset.state_space.corr_indices[0])
    second_marg_basename = 'corr_marg_prob{}.jpg'.format(
      dset.state_space.corr_indices[1])
    plot_joint_prob(joint_prob, os.path.join(dirname, 'corr_joint_prob.jpg'))
    plot_joint_prob(first_marg_prob, os.path.join(dirname, first_marg_basename))
    plot_joint_prob(second_marg_prob, os.path.join(dirname,
                                                   second_marg_basename))

  # log all gin params used
  with gfile.Open(os.path.join(dirname, 'config.txt'), 'w') as f:
    f.write(gin.operative_config_str())

  gin.clear_config()

  print(dirname)


if __name__ == "__main__":
  app.run(main)
