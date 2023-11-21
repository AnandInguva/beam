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
# pytype: skip-file

import shutil
import tempfile
import typing
import unittest
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from parameterized import param
from parameterized import parameterized

import apache_beam as beam
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.transforms import base
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports
try:
  from apache_beam.ml.transforms import tft
  from apache_beam.ml.transforms.tft import TFTOperation
except ImportError:
  tft = None  # type: ignore

try:

  class _FakeOperation(TFTOperation):
    def __init__(self, name, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.name = name

    def apply_transform(self, inputs, output_column_name, **kwargs):
      return {output_column_name: inputs}
except:  # pylint: disable=bare-except
  pass


class BaseMLTransformTest(unittest.TestCase):
  def setUp(self) -> None:
    self.artifact_location = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.artifact_location)

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_appends_transforms_to_process_handler_correctly(self):
    fake_fn_1 = _FakeOperation(name='fake_fn_1', columns=['x'])
    transforms = [fake_fn_1]
    ml_transform = base.MLTransform(
        transforms=transforms, write_artifact_location=self.artifact_location)
    ml_transform = ml_transform.with_transform(
        transform=_FakeOperation(name='fake_fn_2', columns=['x']))

    self.assertEqual(len(ml_transform._process_handler.transforms), 2)
    self.assertEqual(
        ml_transform._process_handler.transforms[0].name, 'fake_fn_1')
    self.assertEqual(
        ml_transform._process_handler.transforms[1].name, 'fake_fn_2')

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_on_dict(self):
    transforms = [tft.ScaleTo01(columns=['x'])]
    data = [{'x': 1}, {'x': 2}]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              write_artifact_location=self.artifact_location,
              transforms=transforms))
      expected_output = [
          np.array([0.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
      ]
      actual_output = result | beam.Map(lambda x: x.x)
      assert_that(
          actual_output, equal_to(expected_output, equals_fn=np.array_equal))

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_on_list_dict(self):
    transforms = [tft.ScaleTo01(columns=['x'])]
    data = [{'x': [1, 2, 3]}, {'x': [4, 5, 6]}]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              transforms=transforms,
              write_artifact_location=self.artifact_location))
      expected_output = [
          np.array([0, 0.2, 0.4], dtype=np.float32),
          np.array([0.6, 0.8, 1], dtype=np.float32),
      ]
      actual_output = result | beam.Map(lambda x: x.x)
      assert_that(
          actual_output, equal_to(expected_output, equals_fn=np.array_equal))

  @parameterized.expand([
      param(
          input_data=[{
              'x': 1,
              'y': 2.0,
          }],
          input_types={
              'x': int, 'y': float
          },
          expected_dtype={
              'x': typing.Sequence[np.float32],
              'y': typing.Sequence[np.float32],
          },
      ),
      param(
          input_data=[{
              'x': np.array([1], dtype=np.int64),
              'y': np.array([2.0], dtype=np.float32),
          }],
          input_types={
              'x': np.int32, 'y': np.float32
          },
          expected_dtype={
              'x': typing.Sequence[np.float32],
              'y': typing.Sequence[np.float32],
          },
      ),
      param(
          input_data=[{
              'x': [1, 2, 3], 'y': [2.0, 3.0, 4.0]
          }],
          input_types={
              'x': List[int], 'y': List[float]
          },
          expected_dtype={
              'x': typing.Sequence[np.float32],
              'y': typing.Sequence[np.float32],
          },
      ),
      param(
          input_data=[{
              'x': [1, 2, 3], 'y': [2.0, 3.0, 4.0]
          }],
          input_types={
              'x': typing.Sequence[int],
              'y': typing.Sequence[float],
          },
          expected_dtype={
              'x': typing.Sequence[np.float32],
              'y': typing.Sequence[np.float32],
          },
      ),
  ])
  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_dict_output_pcoll_schema(
      self, input_data, input_types, expected_dtype):
    transforms = [tft.ScaleTo01(columns=['x'])]
    with beam.Pipeline() as p:
      schema_data = (
          p
          | beam.Create(input_data)
          | beam.Map(lambda x: beam.Row(**x)).with_output_types(
              beam.row_type.RowTypeConstraint.from_fields(
                  list(input_types.items()))))
      transformed_data = schema_data | base.MLTransform(
          write_artifact_location=self.artifact_location, transforms=transforms)
      for name, typ in transformed_data.element_type._fields:
        if name in expected_dtype:
          self.assertEqual(expected_dtype[name], typ)

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_fail_for_non_global_windows_in_produce_mode(self):
    transforms = [tft.ScaleTo01(columns=['x'])]
    with beam.Pipeline() as p:
      with self.assertRaises(RuntimeError):
        _ = (
            p
            | beam.Create([{
                'x': 1, 'y': 2.0
            }])
            | beam.WindowInto(beam.window.FixedWindows(1))
            | base.MLTransform(
                transforms=transforms,
                write_artifact_location=self.artifact_location,
            ))

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transform_on_multiple_columns_single_transform(self):
    transforms = [tft.ScaleTo01(columns=['x', 'y'])]
    data = [{'x': [1, 2, 3], 'y': [1.0, 10.0, 20.0]}]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              transforms=transforms,
              write_artifact_location=self.artifact_location))
      expected_output_x = [
          np.array([0, 0.5, 1], dtype=np.float32),
      ]
      expected_output_y = [np.array([0, 0.47368422, 1], dtype=np.float32)]
      actual_output_x = result | beam.Map(lambda x: x.x)
      actual_output_y = result | beam.Map(lambda x: x.y)
      assert_that(
          actual_output_x,
          equal_to(expected_output_x, equals_fn=np.array_equal))
      assert_that(
          actual_output_y,
          equal_to(expected_output_y, equals_fn=np.array_equal),
          label='y')

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_ml_transforms_on_multiple_columns_multiple_transforms(self):
    transforms = [
        tft.ScaleTo01(columns=['x']),
        tft.ComputeAndApplyVocabulary(columns=['y'])
    ]
    data = [{'x': [1, 2, 3], 'y': ['a', 'b', 'c']}]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              transforms=transforms,
              write_artifact_location=self.artifact_location))
      expected_output_x = [
          np.array([0, 0.5, 1], dtype=np.float32),
      ]
      expected_output_y = [np.array([2, 1, 0])]
      actual_output_x = result | beam.Map(lambda x: x.x)
      actual_output_y = result | beam.Map(lambda x: x.y)

      assert_that(
          actual_output_x,
          equal_to(expected_output_x, equals_fn=np.array_equal))
      assert_that(
          actual_output_y,
          equal_to(expected_output_y, equals_fn=np.array_equal),
          label='actual_output_y')

  @unittest.skipIf(tft is None, 'tft module is not installed.')
  def test_mltransform_with_counter(self):
    transforms = [
        tft.ComputeAndApplyVocabulary(columns=['y']),
        tft.ScaleTo01(columns=['x'])
    ]
    data = [{'x': [1, 2, 3], 'y': ['a', 'b', 'c']}]
    with beam.Pipeline() as p:
      _ = (
          p | beam.Create(data)
          | base.MLTransform(
              transforms=transforms,
              write_artifact_location=self.artifact_location))
    scale_to_01_counter = MetricsFilter().with_name('BeamML_ScaleTo01')
    vocab_counter = MetricsFilter().with_name(
        'BeamML_ComputeAndApplyVocabulary')
    mltransform_counter = MetricsFilter().with_name('BeamML_MLTransform')
    result = p.result
    self.assertEqual(
        result.metrics().query(scale_to_01_counter)['counters'][0].result, 1)
    self.assertEqual(
        result.metrics().query(vocab_counter)['counters'][0].result, 1)
    self.assertEqual(
        result.metrics().query(mltransform_counter)['counters'][0].result, 1)

  def test_non_ptransfrom_provider_class_to_mltransform(self):
    class Add:
      def __call__(self, x):
        return x + 1

    with self.assertRaisesRegex(
        TypeError, 'transform must be a subclass of PTransformProvider'):
      with beam.Pipeline() as p:
        _ = (
            p
            | beam.Create([{
                'x': 1
            }])
            | base.MLTransform(
                write_artifact_location=self.artifact_location).with_transform(
                    Add()))


class FakeModel:
  def __call__(self, example: List[str]) -> List[str]:
    for i in range(len(example)):
      example[i] = example[i][::-1]
    return example


class FakeModelHandler(ModelHandler):
  def run_inference(
      self,
      batch: List[str],
      model: Any,
      inference_args: Optional[Dict[str, Any]] = None):
    return model(batch)

  def load_model(self):
    return FakeModel()


class _YieldPTransform(beam.PTransform):
  def expand(self, pcoll):
    def yield_elements(elements):
      for element in elements:
        yield element

    return pcoll | beam.ParDo(yield_elements)


class FakeEmbeddingsManager(base.EmbeddingsManager):
  def __init__(self, columns):
    super().__init__(columns=columns)

  def get_model_handler(self) -> ModelHandler:
    return FakeModelHandler()

  def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
    return (
        RunInference(model_handler=base._TextEmbeddingHandler(self))
        | _YieldPTransform())


class TextEmbeddingHandlerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.embedding_conig = FakeEmbeddingsManager(columns=['x'])
    self.artifact_location = tempfile.mkdtemp()

  def tearDown(self) -> None:
    shutil.rmtree(self.artifact_location)

  def test_handler_with_incompatible_datatype(self):
    text_handler = base._TextEmbeddingHandler(
        embeddings_manager=self.embedding_conig)
    data = [
        ('x', 1),
        ('x', 2),
        ('x', 3),
    ]
    with self.assertRaises(TypeError):
      text_handler.run_inference(data, None, None)

  def test_handler_with_dict_inputs(self):
    data = [
        {
            'x': "Hello world"
        },
        {
            'x': "Apache Beam"
        },
    ]
    expected_data = [{key: value[::-1]
                      for key, value in d.items()} for d in data]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              write_artifact_location=self.artifact_location).with_transform(
                  self.embedding_conig))
      assert_that(
          result,
          equal_to(expected_data),
      )

  def test_handler_with_batch_sizes(self):
    self.embedding_conig.max_batch_size = 100
    self.embedding_conig.min_batch_size = 10
    data = [
        {
            'x': "Hello world"
        },
        {
            'x': "Apache Beam"
        },
    ] * 100
    expected_data = [{key: value[::-1]
                      for key, value in d.items()} for d in data]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              write_artifact_location=self.artifact_location).with_transform(
                  self.embedding_conig))
      assert_that(
          result,
          equal_to(expected_data),
      )

  def test_handler_on_multiple_columns(self):
    self.embedding_conig.columns = ['x', 'y']
    data = [
        {
            'x': "Hello world", 'y': "Apache Beam", 'z': 'unchanged'
        },
        {
            'x': "Apache Beam", 'y': "Hello world", 'z': 'unchanged'
        },
    ]
    self.embedding_conig.columns = ['x', 'y']
    expected_data = [{
        key: (value[::-1] if key in self.embedding_conig.columns else value)
        for key,
        value in d.items()
    } for d in data]
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create(data)
          | base.MLTransform(
              write_artifact_location=self.artifact_location).with_transform(
                  self.embedding_conig))
      assert_that(
          result,
          equal_to(expected_data),
      )


if __name__ == '__main__':
  unittest.main()
