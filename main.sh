#!/bin/bash

bin/dlib_train \
  --model_dir=/tmp/disentanglement_lib/adult \
  --overwrite=True \
  --gin_config=disentanglement_lib/config/correlated_factors_study_v1/model_configs/tabular.gin \
  --gin_bindings="dataset.name='adult'" \
  --gin_bindings="adult_details.split='train'" \
  --gin_bindings="adult_details.seed=0" \
  #--gin_bindings="correlation_details.corr_indices=(0, 1)" \
  #--gin_bindings="correlation_details.corr_type='line'" \
