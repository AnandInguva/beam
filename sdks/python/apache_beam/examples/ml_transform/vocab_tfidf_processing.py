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

"""
This example uses Large Movie Review Dataset: http://ai.stanford.edu/~amaas/data/sentiment/ # pylint:disable=line-too-long
to preprocess the input text data and generate TF-IDF scores for each word of
the input text.

Workflow:
1. The input text is split into words using a delimiter.
2. The words are then converted to a vocabulary index using
    ComputeAndApplyVocabulary.
3. The vocabulary index is then converted to TF-IDF scores using TFIDF.
4. The output of the pipeline is a Tuple of
    (input_text, [(vocab_index, tfidf_score)].

To run this pipeline, download the Large Movie Review Dataset and
place it in a directory. Pass the directory path to --input_data_dir.
The pipeline will read the data from the directory and write the
transformed data to --output_dir. To save the artifacts, such as the
vocabulary file generated by ComputeAndApplyVocabulary will be saved to
the --artifact_location.

In this pipeline, we only preprocess the train data(25000 samples).
"""

import argparse
import logging
import os

import apache_beam as beam
from apache_beam.ml.transforms.base import MLTransform
from apache_beam.ml.transforms.tft import TFIDF
from apache_beam.ml.transforms.tft import ComputeAndApplyVocabulary

RAW_DATA_KEY = 'raw_data'
REVIEW_COLUMN = 'review'
LABEL_COLUMN = 'label'
DELIMITERS = '.,!?() '
VOCAB_SIZE = 20000


# pylint: disable=invalid-name
@beam.ptransform_fn
def Shuffle(pcoll):
  """Shuffles a PCollection.  Collection should not contain duplicates."""
  return (
      pcoll
      | 'PairWithHash' >> beam.Map(lambda x: (hash(x), x))
      | 'GroupByHash' >> beam.GroupByKey()
      | 'DropHash' >> beam.FlatMap(lambda hash_and_values: hash_and_values[1]))


class ReadAndShuffleData(beam.PTransform):
  def __init__(self, pos_file_pattern, neg_file_pattern):
    self.pos_file_pattern = pos_file_pattern
    self.neg_file_pattern = neg_file_pattern

  def expand(self, pcoll):

    negative_examples = (
        pcoll
        | "ReadNegativeExample" >> beam.io.ReadFromText(self.neg_file_pattern)
        | 'PairWithZero' >> beam.Map(lambda review: (review, 0))
        | 'DistinctNeg' >> beam.Distinct())

    positive_examples = (
        pcoll
        | "ReadPositiveExample" >> beam.io.ReadFromText(self.pos_file_pattern)
        | 'PairWithOne' >> beam.Map(lambda review: (review, 1))
        | 'DistinctPos' >> beam.Distinct())

    all_examples = ((negative_examples, positive_examples)
                    | 'FlattenPColls' >> beam.Flatten())

    shuffled_examples = (all_examples | 'Shuffle' >> Shuffle())

    # tag with column names for MLTransform
    return (
        shuffled_examples
        | beam.Map(
            lambda label_review: {
                REVIEW_COLUMN: label_review[0],
                LABEL_COLUMN: label_review[1],
                RAW_DATA_KEY: label_review[0]
            }))


def preprocess_data(
    file_patterns,
    pipeline_args,
    artifact_location,
    output_dir,
    test_pipeline=None  # used for testing purposes.
):
  positive_pattern, negative_pattern = file_patterns
  options = beam.options.pipeline_options.PipelineOptions(pipeline_args)
  pipeline = test_pipeline
  if not test_pipeline:
    pipeline = beam.Pipeline(options=options)
  data_pcoll = (
      pipeline
      |
      'ReadTrainData' >> ReadAndShuffleData(positive_pattern, negative_pattern))
  ml_transform = MLTransform(
      write_artifact_location=artifact_location,
  ).with_transform(
      ComputeAndApplyVocabulary(
          top_k=VOCAB_SIZE,
          frequency_threshold=10,
          columns=[REVIEW_COLUMN],
          split_string_by_delimiter=DELIMITERS)).with_transform(
              TFIDF(columns=[REVIEW_COLUMN], vocab_size=VOCAB_SIZE))
  data_pcoll = data_pcoll | 'MLTransform' >> ml_transform

  data_pcoll = (
      data_pcoll | beam.ParDo(MapTFIDFScoreToVocab(artifact_location)))

  #   _ = (data_pcoll | beam.io.WriteToText(output_dir))

  _ = data_pcoll | beam.Map(logging.info)

  result = pipeline.run()
  result.wait_until_finish()
  return result


class MapTFIDFScoreToVocab(beam.DoFn):
  def __init__(self, artifact_location):
    self.artifact_location = artifact_location

  def process(self, element):
    index_column_name = REVIEW_COLUMN + '_vocab_index'
    weight_column_name = REVIEW_COLUMN + '_tfidf_weight'
    element = element.as_dict()
    raw_data = element[RAW_DATA_KEY]

    vocab_index = element[index_column_name]
    weights = element[weight_column_name]

    vocabs_with_weights = [(vocab_index[i], weights[i])
                           for i in range(len(vocab_index))]

    return [(
        raw_data,
        vocabs_with_weights,
    )]


def parse_known_args(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_data_dir', help='path to directory containing input data.')
  parser.add_argument(
      '--artifact_location',
      help='path to directory to hold artifacts such as vocab files.')
  parser.add_argument(
      '--output_dir', help='path to directory to hold transformed data.')
  return parser.parse_known_args(argv)


def run(argv=None):
  args, pipeline_args = parse_known_args(argv)
  neg_filepatterm = os.path.join(args.input_data_dir, 'train/neg/*')
  pos_filepattern = os.path.join(args.input_data_dir, 'train/pos/*')
  artifact_location = args.artifact_location

  _ = preprocess_data((pos_filepattern, neg_filepatterm),
                      pipeline_args,
                      artifact_location,
                      output_dir=args.output_dir)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
