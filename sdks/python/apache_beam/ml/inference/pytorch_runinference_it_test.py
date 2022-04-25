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
import os

import apache_beam as beam
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.api import RunInference
from apache_beam.ml.inference.pytorch import PytorchModelLoader
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from functools import partial
from PIL import Image
from typing import Any

#########################################
_SAMPLES_DIR = 'gs://clouddfe-anandinguva/imagenet_data/train'
# _SAMPLES_DIR = '/Users/anandinguva/Desktop/train'
#########################################

_IMG_SIZE = (3, 224, 224)


def _read_image(path_to_file: str, file_dir: str) -> Any:

  path_to_file = os.path.join(file_dir, path_to_file)
  with FileSystems().open(path_to_file, 'r') as file:
    data = Image.open(io.BytesIO(file.read()))
    return path_to_file, data


def _resize_image(data):
  return data[0], torch.from_numpy(np.asarray(np.resize(data[1],
                             new_shape=_IMG_SIZE),
                             dtype=np.float32))


class PostProcessor(beam.DoFn):
  def process(self, element):
    filename, prediction_result = element
    prediction = torch.argmax(prediction_result.inference, dim=0)
    yield filename, prediction


def setup_pipeline(options: PipelineOptions, args):
  """Sets up dataflow pipeline based on specified arguments"""
  model = torchvision.models.vgg16(pretrained=False)
  model_loader = PytorchModelLoader(
      state_dict_path=args.model_path,
      model_class=model,
  )

  with beam.Pipeline(options=options) as p:
    filename_value_pair = (
        p | 'Read the input file' >> beam.io.ReadFromText(
            args.input, skip_header_lines=1)
        | 'Get file name to read' >> beam.Map(
            lambda x: x.split(',')[0])  # refactor csv file
        | 'Parse and read files from the input_file' >> beam.Map(
            partial(_read_image, file_dir=_SAMPLES_DIR))
        | 'Resize the data' >> beam.Map(_resize_image))

    predictions = (
        filename_value_pair
        | 'PyTorch RunInference' >> RunInference(model_loader)
        | 'Process output' >> beam.ParDo(PostProcessor()))

    predictions | "Write output to GCS" >> beam.io.WriteToText(
        args.output,
        file_name_suffix='.csv',
        shard_name_template='',
        append_trailing_newlines=True)


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--input',
      dest='input',
      required=True,
      help='A file containing images names and other metadata in columns')
  parser.add_argument(
      '--output', dest='output', help='Output path for output files.')
  parser.add_argument(
      '--model_path', dest='model_path', help='Path to load the model.')
  parser.add_argument(
      '--benchmark_type',
      dest='benchmark_type',
      default=None,
      help='Type of benchmark to run.')
  return parser.parse_known_args(argv)


def run(argv=None, save_main_session=True):
  """Entry point. Defines and runs the pipeline."""
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args=pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  setup_pipeline(pipeline_options, known_args)


if __name__ == '__main__':
  run()
