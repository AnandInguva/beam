#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: skip-file

import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow_serving.apis import prediction_log_pb2

import apache_beam as beam
import tfx_bsl
from tfx_bsl.public.beam import RunInference
from tfx_bsl.public import tfxio
from tfx_bsl.public.proto import model_spec_pb2

import numpy as np

from typing import Dict, Text, Any, Tuple, List

from apache_beam.options.pipeline_options import PipelineOptions
_INPUT_SHAPE = (224, 224, 3)
# input_layer = keras.layers.Input(shape=_INPUT_SHAPE,
#                                  dtype=tf.float32,
#                                  name='x')
# mid_layer = keras.layers.Flatten()(input_layer)
# output_layer= keras.layers.Dense(1)(mid_layer)
# model = keras.Model(input_layer, output_layer)

model = tf.keras.applications.MobileNet(input_shape=_INPUT_SHAPE)

RAW_DATA_PREDICT_SPEC = {
    'x': tf.io.VarLenFeature(dtype=np.float32),
}


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
def serve_tf_examples_fn(serialized_tf_examples):
  """Returns the output to be used in the serving signature."""
  features = tf.io.parse_example(serialized_tf_examples, RAW_DATA_PREDICT_SPEC)
  features = tf.sparse.to_dense(features['x'], default_value=0)
  features = tf.reshape(features, [-1, 224, 224, 3])
  return model(features, training=False)


# NEW MODEL PATH
new_model_path = '/Users/anandinguva/Desktop/custom_model/2'
signature = {'serving_default': serve_tf_examples_fn}
tf.saved_model.save(model, new_model_path, signatures=signature)
