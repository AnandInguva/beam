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

import argparse
import io
import logging
import os
from typing import Iterator
from typing import Optional
from typing import Tuple

import apache_beam as beam
import tensorflow as tf
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult
from PIL import Image
from tfx_bsl.public.beam.run_inference import CreateModelHandler
from tfx_bsl.public.proto import model_spec_pb2

_IMG_SIZE = (224, 224)


def filter_empty_lines(text: str) -> Iterator[str]:
  if len(text.strip()) > 0:
    yield text


def read_image(image_file_name: str,
               path_to_dir: Optional[str] = None) -> Tuple[str, Image.Image]:
  if path_to_dir is not None:
    image_file_name = os.path.join(path_to_dir, image_file_name)
  with FileSystems().open(image_file_name, 'r') as file:
    data = Image.open(io.BytesIO(file.read())).convert('RGB')
    return image_file_name, data


def preprocess_image(data):
  # Note: Converts the image dtype from uint8 to float32
  # https://www.tensorflow.org/api_docs/python/tf/image/resize
  image = tf.image.resize(data, _IMG_SIZE)
  return image


class ExampleProcessor:
  def create_example(self, element):
    feature_of_bytes = self.create_feature_vector(element)
    features_for_example = {'image': feature_of_bytes}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features_for_example))
    return example_proto.SerializeToString()

  def create_feature_vector(self, element):
    serialized_non_scalar = tf.io.serialize_tensor(element)
    feature_of_bytes = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_non_scalar.numpy()]))
    return feature_of_bytes

  def create_feature_int(self, element):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[element]))


class PostProcessor(beam.DoFn):
  def process(self, element):
    filename, predict_log = element[0], element[1].predict_log
    output_value = predict_log.response.outputs
    output_tensor = (
        tf.io.decode_raw(
            output_value['output_0'].tensor_content, out_type=tf.float32))
    max_index_output_tensor = tf.math.argmax(output_tensor, axis=0)
    yield filename, tf.get_static_value(max_index_output_tensor)


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      required=True,
      help='Path to the text file containing image names.')
  parser.add_argument(
      '--output',
      dest='output',
      required=True,
      help='Path where to save output predictions.'
      ' text file.')
  parser.add_argument(
      '--model_path',
      dest='model_path',
      required=True,
      help="Path to the model.")
  parser.add_argument(
      '--images_dir',
      default=None,
      help='Path to the directory where images are stored.'
      'Not required if image names in the input file have absolute path.')
  return parser.parse_known_args(argv)


def run(
    argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
  """
  Args:
    argv: Command line arguments defined for this example.
    save_main_session: Used for internal testing.
    test_pipeline: Used for internal testing.
  """
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  saved_model_spec = model_spec_pb2.SavedModelSpec(
      model_path=known_args.model_path)
  inferece_spec_type = model_spec_pb2.InferenceSpecType(
      saved_model_spec=saved_model_spec)
  model_handler = CreateModelHandler(inferece_spec_type)
  # create a keyedModelHandler to accommodate image names as keys.
  keyed_model_handler = KeyedModelHandler(model_handler)

  pipeline = test_pipeline
  if not test_pipeline:
    pipeline = beam.Pipeline(options=pipeline_options)

  filename_value_pair = (
      pipeline
      | 'ReadImageNames' >> beam.io.ReadFromText(known_args.input)
      | 'FilterEmptyLines' >> beam.ParDo(filter_empty_lines)
      | 'ReadImageData' >> beam.Map(
          lambda image_name: read_image(
              image_file_name=image_name, path_to_dir=known_args.images_dir))
      | 'ReshapeImage' >> beam.Map(lambda x: (x[0], preprocess_image(x[1]))))

  predictions = (
      filename_value_pair
      | 'ConvertToExampleProto' >>
      beam.Map(lambda x: (x[0], ExampleProcessor().create_example(x[1])))
      | 'TFXRunInference' >> RunInference(keyed_model_handler)
      | 'PostProcess' >> beam.ParDo(PostProcessor()))
  _ = (
      predictions
      | "WriteOutputToGCS" >> beam.io.WriteToText(
          known_args.output,
          shard_name_template='',
          append_trailing_newlines=True))

  result = pipeline.run()
  result.wait_until_finish()
  return result


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
