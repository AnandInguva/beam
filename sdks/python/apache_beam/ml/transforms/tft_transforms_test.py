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

from typing import List
from typing import NamedTuple

import unittest
import numpy as np
from parameterized import parameterized

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

# pylint: disable=wrong-import-order, wrong-import-position
try:
  from apache_beam.ml.transforms import base
  from apache_beam.ml.transforms import tft_transforms
  from apache_beam.ml.transforms import handlers
except ImportError:
  tft_transforms = None

skip_if_tft_not_available = unittest.skipIf(
    tft_transforms is None, 'tensorflow_transform is not installed.')


class MyTypesUnbatched(NamedTuple):
  x: List[int]


class MyTypesBatched(NamedTuple):
  x: List[int]


z_score_expected = {'x_mean': 3.5, 'x_var': 2.9166666666666665}


def assert_z_score_artifacts(element):
  element = element.as_dict()
  assert 'x_mean' in element
  assert 'x_var' in element
  assert element['x_mean'] == z_score_expected['x_mean']
  assert element['x_var'] == z_score_expected['x_var']


def assert_scale_to_0_1_artifacts(element):
  element = element.as_dict()
  assert 'x_min' in element
  assert 'x_max' in element
  assert element['x_min'] == 1
  assert element['x_max'] == 6


def assert_bucketize_artifacts(element):
  element = element.as_dict()
  assert 'x_quantiles' in element
  assert np.array_equal(
      element['x_quantiles'], np.array([3, 5], dtype=np.float32))


@skip_if_tft_not_available
class ScaleZScoreTest(unittest.TestCase):
  def test_z_score_unbatched(self):
    unbatched_data = [{
        'x': 1
    }, {
        'x': 2
    }, {
        'x': 3
    }, {
        'x': 4
    }, {
        'x': 5
    }, {
        'x': 6
    }]

    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      unbatched_result = (
          p
          | "unbatchedCreate" >> beam.Create(unbatched_data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesUnbatched)
          | "unbatchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Scale_To_ZScore(columns=['x'])))
      _ = (unbatched_result | beam.Map(assert_z_score_artifacts))

  def test_z_score_batched(self):
    batched_data = [{'x': [1, 2, 3]}, {'x': [4, 5, 6]}]
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      batched_result = (
          p
          | "batchedCreate" >> beam.Create(batched_data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesBatched)
          | "batchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Scale_To_ZScore(columns=['x'])))
      _ = (batched_result | beam.Map(assert_z_score_artifacts))


@skip_if_tft_not_available
class ScaleTo01Test(unittest.TestCase):
  def test_scale_to_0_1_batched(self):
    batched_data = [{'x': [1, 2, 3]}, {'x': [4, 5, 6]}]
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      batched_result = (
          p
          | "batchedCreate" >> beam.Create(batched_data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesBatched)
          | "batchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Scale_To_0_1(columns=['x'])))
      _ = (batched_result | beam.Map(assert_scale_to_0_1_artifacts))

      expected_output = [
          np.array([0, 0.2, 0.4], dtype=np.float32),
          np.array([0.6, 0.8, 1], dtype=np.float32)
      ]
      actual_output = (batched_result | beam.Map(lambda x: x.x))
      assert_that(
          actual_output, equal_to(expected_output, equals_fn=np.array_equal))

  def test_scale_to_0_1_unbatched(self):
    unbatched_data = [{
        'x': 1
    }, {
        'x': 2
    }, {
        'x': 3
    }, {
        'x': 4
    }, {
        'x': 5
    }, {
        'x': 6
    }]
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      unbatched_result = (
          p
          | "unbatchedCreate" >> beam.Create(unbatched_data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesUnbatched)
          | "unbatchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Scale_To_0_1(columns=['x'])))

      _ = (unbatched_result | beam.Map(assert_scale_to_0_1_artifacts))
      expected_output = (
          np.array([0], dtype=np.float32),
          np.array([0.2], dtype=np.float32),
          np.array([0.4], dtype=np.float32),
          np.array([0.6], dtype=np.float32),
          np.array([0.8], dtype=np.float32),
          np.array([1], dtype=np.float32))
      actual_output = (unbatched_result | beam.Map(lambda x: x.x))
      assert_that(
          actual_output, equal_to(expected_output, equals_fn=np.array_equal))


@skip_if_tft_not_available
class BucketizeTest(unittest.TestCase):
  def test_bucketize_unbatched(self):
    unbatched = [{'x': 1}, {'x': 2}, {'x': 3}, {'x': 4}, {'x': 5}, {'x': 6}]
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      unbatched_result = (
          p
          | "unbatchedCreate" >> beam.Create(unbatched)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesUnbatched)
          | "unbatchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Bucketize(columns=['x'], num_buckets=3)))
      _ = (unbatched_result | beam.Map(assert_bucketize_artifacts))

      transformed_data = (unbatched_result | beam.Map(lambda x: x.x))
      expected_data = [
          np.array([0]),
          np.array([0]),
          np.array([1]),
          np.array([1]),
          np.array([2]),
          np.array([2])
      ]
      assert_that(
          transformed_data, equal_to(expected_data, equals_fn=np.array_equal))

  def test_bucketize_batched(self):
    batched = [{'x': [1, 2, 3]}, {'x': [4, 5, 6]}]
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      batched_result = (
          p
          | "batchedCreate" >> beam.Create(batched)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesBatched)
          | "batchedMLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Bucketize(columns=['x'], num_buckets=3)))
      _ = (batched_result | beam.Map(assert_bucketize_artifacts))

      transformed_data = (
          batched_result
          | "TransformedColumnX" >> beam.Map(lambda ele: ele.x))
      expected_data = [
          np.array([0, 0, 1], dtype=np.int64),
          np.array([1, 2, 2], dtype=np.int64)
      ]
      assert_that(
          transformed_data, equal_to(expected_data, equals_fn=np.array_equal))

  @parameterized.expand([
      (range(1, 10), [4, 7]),
      (range(9, 0, -1), [4, 7]),
      (range(19, 0, -1), [10]),
      (range(1, 100), [25, 50, 75]),
      # similar to the above but with odd number of elements
      (range(1, 100, 2), [25, 51, 75]),
      (range(99, 0, -1), range(10, 100, 10))
  ])
  def test_bucketize_boundaries(self, test_input, expected_boundaries):
    # boundaries are outputted as artifacts for the Bucketize transform.
    data = [{'x': [i]} for i in test_input]
    num_buckets = len(expected_boundaries) + 1
    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      result = (
          p
          | "Create" >> beam.Create(data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesUnbatched)
          | "MLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.Bucketize(columns=['x'], num_buckets=num_buckets)))
      actual_boundaries = (
          result
          | beam.Map(lambda x: x.as_dict())
          | beam.Map(lambda x: x['x_quantiles']))

      def assert_boundaries(actual_boundaries):
        assert np.array_equal(actual_boundaries, expected_boundaries)

      _ = (actual_boundaries | beam.Map(assert_boundaries))


@skip_if_tft_not_available
class ApplyBucketsTest(unittest.TestCase):
  @parameterized.expand([
      (range(1, 100), [25, 50, 75]),
      (range(1, 100, 2), [25, 51, 75]),
  ])
  def test_apply_buckets(self, test_inputs, bucket_boundaries):
    with beam.Pipeline() as p:
      data = [{'x': [i]} for i in test_inputs]
      process_handler = handlers.TFTProcessHandlerSchema()
      result = (
          p
          | "Create" >> beam.Create(data)
          | beam.Map(lambda x: MyTypesBatched(**x)).with_output_types(
              MyTypesUnbatched)
          | "MLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.ApplyBuckets(
                  columns=['x'], bucket_boundaries=bucket_boundaries)))
      expected_output = []
      bucket = 0
      for x in sorted(test_inputs):
        # Increment the bucket number when crossing the boundary
        if (bucket < len(bucket_boundaries) and x >= bucket_boundaries[bucket]):
          bucket += 1
        expected_output.append(np.array([bucket]))

      actual_output = (result | beam.Map(lambda x: x.x))
      assert_that(
          actual_output, equal_to(expected_output, equals_fn=np.array_equal))


class ComputeAndVocabUnbatchedInputType(NamedTuple):
  x: str


class ComputeAndVocabBatchedInputType(NamedTuple):
  x: List[str]


@skip_if_tft_not_available
class ComputeAndApplyVocabTest(unittest.TestCase):
  def test_compute_and_apply_vocabulary_unbatched_inputs(self):
    batch_size = 100
    num_instances = batch_size + 1
    input_data = [{
        'x': '%.10i' % i,  # Front-padded to facilitate lexicographic sorting.
    } for i in range(num_instances)]

    expected_data = [{
        'x': (len(input_data) - 1) - i, # Due to reverse lexicographic sorting.
    } for i in range(len(input_data))]

    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      actual_data = (
          p
          | "Create" >> beam.Create(input_data)
          | beam.Map(lambda x: ComputeAndVocabUnbatchedInputType(**x)
                     ).with_output_types(ComputeAndVocabUnbatchedInputType)
          | "MLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.ComputeAndApplyVocabulary(columns=['x'])))
      actual_data |= beam.Map(lambda x: x.as_dict())

      assert_that(actual_data, equal_to(expected_data))

  def test_compute_and_apply_vocabulary_batched(self):
    batch_size = 100
    num_instances = batch_size + 1
    input_data = [
        {
            'x': ['%.10i' % i, '%.10i' % (i + 1), '%.10i' % (i + 2)],
            # Front-padded to facilitate lexicographic sorting.
        } for i in range(0, num_instances, 3)
    ]

    # since we have 3 elements in a single batch, multiply with 3 for
    # each iteration i on the expected output.
    excepted_data = [
        np.array([(len(input_data) * 3 - 1) - i,
                  (len(input_data) * 3 - 1) - i - 1,
                  (len(input_data) * 3 - 1) - i - 2],
                 dtype=np.int64)  # Front-padded to facilitate lexicographic
        # sorting.
        for i in range(0, len(input_data) * 3, 3)
    ]

    with beam.Pipeline() as p:
      process_handler = handlers.TFTProcessHandlerSchema()
      result = (
          p
          | "Create" >> beam.Create(input_data)
          | beam.Map(lambda x: ComputeAndVocabBatchedInputType(**x)
                     ).with_output_types(ComputeAndVocabBatchedInputType)
          | "MLTransform" >>
          base.MLTransform(process_handler=process_handler).with_transform(
              tft_transforms.ComputeAndApplyVocabulary(columns=['x'])))
      actual_output = (result | beam.Map(lambda x: x.x))
      assert_that(
          actual_output, equal_to(excepted_data, equals_fn=np.array_equal))


if __name__ == '__main__':
  unittest.main()
