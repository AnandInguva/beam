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
import numpy as np

_INPUT_SHAPE = (224, 224, 3)

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


def run(args):
  path_to_save_model = args.path_to_save_model
  signature = {'serving_default': serve_tf_examples_fn}
  # save the model with Signature compatible with TFX RunInference
  tf.saved_model.save(model, path_to_save_model, signatures=signature)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--saved_model_path',
      default=None,
      help='Saved tf model without any configuration')
  parser.add_argument(
      '--path_to_save_model',
      required=True,
      help='Path to save the configured model')
