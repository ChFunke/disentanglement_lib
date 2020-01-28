# coding=utf-8
"""Asserts no nans in adult dataset loader"""

from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.data.ground_truth import util
import gin.tf
from tqdm import tqdm

# TODO(creager): implement sampler; check nan

FLAGS = flags.FLAGS
flags.DEFINE_string(
  'gin_file', None, 'Path of config file.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')


NUM_BATCHES = 2000  # exhausts all 49k examples even at small batch sizes
MIN_BATCH_SIZE = int(50e3 / NUM_BATCHES)


@gin.configurable
def get_dataset_loader(batch_size, seed):
  dataset = named_data.get_named_ground_truth_data()
  loader = util.tf_data_set_from_ground_truth_data(dataset, seed)
  loader = loader.batch(batch_size, drop_remainder=True)
  return loader, dataset, batch_size, seed


def main(unused_argv):

  del unused_argv  # unused

  gin.parse_config_files_and_bindings([FLAGS.gin_file], FLAGS.gin_param)

  if not gin.query_parameter("get_dataset_loader.batch_size") > MIN_BATCH_SIZE:
    raise ValueError("use a bigger batch size!")

  dset_iter, dset, batch_size, seed = get_dataset_loader()
  dset_iter = iter(dset_iter)

  msg = 'loaded split %s with %s examples' % (
    gin.query_parameter('adult_details.split'), len(dset.x)
  )
  print(msg)

  # test observations X are not Nan via dataset iterator
  for _ in tqdm(range(NUM_BATCHES)):
    observations = next(dset_iter)  # Tensor
    assert np.isnan(observations.numpy()).sum() == 0., 'NaN found!'

  # test observations X and factors of variation (U, A, Y) are not Nan via dset
  npr = np.random.RandomState(seed)
  for _ in tqdm(range(NUM_BATCHES)):
    factors = dset.sample_factors(batch_size, npr)  # Array
    observations = dset.sample_observations(batch_size, npr)  # Array
    for array in (factors, observations):
      assert np.isnan(array).sum() == 0., 'NaN found!'

  gin.clear_config()

  print('done')
  exit()


if __name__ == "__main__":
  app.run(main)
