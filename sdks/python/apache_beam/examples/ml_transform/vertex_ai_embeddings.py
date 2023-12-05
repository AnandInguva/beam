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
import logging

import apache_beam as beam
from apache_beam.ml.transforms.base import MLTransform
from apache_beam.ml.transforms.embeddings.vertex_ai import VertexAITextEmbeddings
from apache_beam.options.pipeline_options import PipelineOptions




text_column = 'x'
numeric_column = 'y'
ARTIFACT_LOCATION = 'gs://anandinguva-test/artifacts'
MODEL_NAME = 'textembedding-gecko@001'

# dummy data
# TODO: replace with better data
data = [
    {text_column: 'Hello world', numeric_column: 1},
    {text_column: 'I am ready', numeric_column: 20}
] * 100

def run_tft_pipeline(artifact_location, pipeline_args):
    options = PipelineOptions(pipeline_args)
    embedding_config = VertexAITextEmbeddings(
       model_name=MODEL_NAME,
       columns=[text_column])
    with beam.Pipeline(options=options) as p:
        data_pcoll = (
            p
            | beam.Create(data)
        )

        mltransform_pcoll = (
            data_pcoll
            | MLTransform(write_artifact_location=artifact_location
                          ).with_transform(embedding_config
        ))
        mltransform_pcoll | beam.Map(logging.info)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--artifact_location', type=str, default=ARTIFACT_LOCATION)
  return parser.parse_known_args()

if __name__ == '__main__':
   logging.getLogger().setLevel(logging.INFO)
   args, pipeline_args = parse_args()
   artifact_location = args.artifact_location
   run_tft_pipeline(artifact_location, pipeline_args=pipeline_args)


