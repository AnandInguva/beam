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
import io
import logging
import os

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tfx_bsl

from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing import synthetic_pipeline
from google.cloud.storage import Client
from PIL import Image
from typing import Any
from typing import Iterable
from typing import List
from typing import Union
from tfx_bsl.public.beam import RunInference
from tfx_bsl.public.proto import model_spec_pb2
from tensorflow_serving.apis import prediction_log_pb2

#########################################
_SAMPLES_DIR = '/Users/anandinguva/Desktop/samples'
_SAMPLES = [os.path.join(_SAMPLES_DIR, i) for i in os.listdir(_SAMPLES_DIR)]
#########################################

_IMG_SIZE = (224, 224, 3)


def _read_image(path_to_file: str, file_dir: str = None) -> Any:
  # path_to_file = os.path.join(file_dir, path_to_file)
  with FileSystems.open(path_to_file, 'r') as file:
    data = Image.open(io.BytesIO(file.read()))
    return path_to_file, np.asarray(np.resize(data,
                                    new_shape=_IMG_SIZE,
                                    ), dtype=np.float32)


class ExampleProcessor:
  def create_examples(self, feature):
    return tf.train.Example(
        features=tf.train.Features(feature={'x': self.create_feature(feature)}))

  def create_feature(self, array: np.array):
    return tf.train.Feature(
        float_list=tf.train.FloatList(
            value=array.reshape(224 * 224 * 3).tolist()))


class PostProcessor(beam.DoFn):
  def process(self, element: prediction_log_pb2.PredictionLog):
    predict_log = element.predict_log
    # Not returning input_value because it's an array and doesn't make sense
    # input_value = tf.train.Example.FromString(predict_log.request.inputs['examples'].string_val[0])
    output_value = predict_log.response.outputs
    output_tensor = (
        tf.io.decode_raw(
            output_value['output_0'].tensor_content, out_type=tf.float32))
    max_index_output_tensor = tf.math.argmax(output_tensor, axis=0)
    yield max_index_output_tensor


def setup_pipeline(p: beam.Pipeline, args):
  """Sets up dataflow pipeline based on specified arguments"""
  filename_value_pair = (
      p | "Create list of image names" >> beam.Create(_SAMPLES)
      # p | 'Read the input file' >> beam.io.ReadFromText(args.input)
      | 'Parse and read files from the input_file' >> beam.Map(_read_image))

  filenames = (
      filename_value_pair
      | 'Get filenames' >> beam.MapTuple(lambda a, b: a))

  data_values = (
      filename_value_pair
      | 'Get arrays of the files' >> beam.MapTuple(lambda a, b: b))

  predictions = (
      data_values
      | 'Convert np.array to tf.train.example' >>
      beam.Map(lambda x: ExampleProcessor().create_examples(x))
      | 'TFX RunInference' >> RunInference(
          model_spec_pb2.InferenceSpecType(
              saved_model_spec=model_spec_pb2.SavedModelSpec(
                  model_path=args.model_path)))
      | "Parse output" >> beam.ParDo(PostProcessor())
      | beam.Map(print))

  (filenames, predictions) >> beam.Map(print)


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()

  # parser.add_argument(
  #   '--input',
  #   dest='input',
  #   required=True,
  #   help='A file containing images names and other metadata in columns')
  parser.add_argument(
      '--output', dest='output', help='Output path for output files.')
  parser.add_argument(
      '--model_path',
      dest='model_path',
      default='/Users/anandinguva/Desktop/custom_model/2',
      help='Path to load the model.')
  parser.add_argument(
      '--benchmark_type',
      dest='benchmark_type',
      default=None,
      help='Type of benchmark to run.')
  return parser.parse_known_args(argv)


def run(argv=None):
  """Entry point. Defines and runs the pipeline."""
  known_args, pipeline_args = parse_known_args(argv)
  with beam.Pipeline(argv=pipeline_args) as p:
    setup_pipeline(p, known_args)


if __name__ == '__main__':
  run()
