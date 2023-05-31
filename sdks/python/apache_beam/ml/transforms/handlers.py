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

import tempfile
import typing
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

import apache_beam as beam
from apache_beam.ml.transforms.base import MLTransformOutput
from apache_beam.ml.transforms.base import ProcessHandler
from apache_beam.ml.transforms.base import ProcessInputT
from apache_beam.ml.transforms.base import ProcessOutputT
from apache_beam.ml.transforms.tft_transforms import _TFTOperation
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.typehints import native_type_compatibility
from apache_beam.typehints.row_type import RowTypeConstraint
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform.beam as tft_beam
from tensorflow_transform import common_types
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

__all__ = [
    'TFTProcessHandlerDict',
]

# tensorflow transform doesn't support the types other than tf.int64,
# tf.float32 and tf.string.
_default_type_to_tensor_type_map = {
    int: tf.int64,
    float: tf.float32,
    str: tf.string,
    bytes: tf.string,
    np.int64: tf.int64,
    np.int32: tf.int64,
    np.float32: tf.float32,
    np.float64: tf.float32,
    np.bytes_: tf.string,
    np.str_: tf.string,
}
_primitive_types = (int, float, str, bytes)

tft_process_handler_dict_input_type = typing.Union[typing.NamedTuple, beam.Row]


class ConvertNamedTupleToDict(
    beam.PTransform[beam.PCollection[tft_process_handler_dict_input_type],
                    beam.PCollection[Dict[str,
                                          common_types.InstanceDictType]]]):
  """
    A PTransform that converts a collection of NamedTuples or Rows into a
    collection of dictionaries.
  """
  def expand(
      self, pcoll: beam.PCollection[tft_process_handler_dict_input_type]
  ) -> beam.PCollection[common_types.InstanceDictType]:
    """
    Args:
      pcoll: A PCollection of NamedTuples or Rows.
    Returns:
      A PCollection of dictionaries.
    """
    if isinstance(pcoll.element_type, RowTypeConstraint):
      # Row instance
      return pcoll | beam.Map(lambda x: x.asdict())
    else:
      # named tuple
      return pcoll | beam.Map(lambda x: x._asdict())


# TODO: Add metrics namespace.
class TFTProcessHandler(ProcessHandler[ProcessInputT, ProcessOutputT]):
  def __init__(
      self,
      *,
      input_types: Optional[Dict[str, type]] = None,
      transforms: List[_TFTOperation] = None,
      namespace: str = 'TFTProcessHandler',
  ):
    """
    A handler class for processing data with TensorFlow Transform (TFT)
    operations. This class is intended to be subclassed, with subclasses
    implementing the `preprocessing_fn` method.

    Args:
      input_types: A dictionary of column names and types.
      transforms: A list of transforms to apply to the data. All the transforms
        are applied in the order they are specified. The input of the
        i-th transform is the output of the (i-1)-th transform. Multi-input
        transforms are not supported yet.
      namespace: A metrics namespace for the TFTProcessHandler.
    """
    super().__init__()
    self._input_types = input_types
    self.transforms = transforms if transforms else []
    self._input_types = input_types
    self._artifact_location = None
    self._namespace = namespace
    self.transformed_schema = None

  def get_raw_data_feature_spec(
      self, input_types: Dict[str, type]) -> dataset_metadata.DatasetMetadata:
    """
    Return a DatasetMetadata object to be used with
    tft_beam.AnalyzeAndTransformDataset.
    Args:
      input_types: A dictionary of column names and types.
    Returns:
      A DatasetMetadata object.
    """
    raw_data_feature_spec = {}
    for key, value in input_types.items():
      raw_data_feature_spec[key] = self._get_raw_data_feature_spec_per_column(
          typ=value, col_name=key)
    raw_data_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(raw_data_feature_spec))
    return raw_data_metadata

  def _get_raw_data_feature_spec_per_column(self, typ: type, col_name: str):
    """
    Return a FeatureSpec object to be used with
    tft_beam.AnalyzeAndTransformDataset
    Args:
      typ: A type of the column.
      col_name: A name of the column.
    Returns:
      A FeatureSpec object.
    """
    # lets conver the builtin types to typing types for consistency.
    typ = native_type_compatibility.convert_builtin_to_typing(typ)
    containers_type = (List._name, Tuple._name)
    is_container = hasattr(typ, '_name') and typ._name in containers_type

    if is_container:
      dtype = typing.get_args(typ)[0]
      if len(typing.get_args(typ)) > 1 or typing.get_origin(dtype) == Union:
        raise RuntimeError(
            f"Incorrect type specifications in {typ} for column {col_name}. "
            f"Please specify a single type.")
      if dtype not in _default_type_to_tensor_type_map:
        raise TypeError(
            f"Unable to identify type: {dtype} specified on column: {col_name}"
            f". Please specify a valid type.")
    else:
      dtype = typ

    is_container = is_container or issubclass(dtype, np.generic)
    if is_container:
      return tf.io.VarLenFeature(_default_type_to_tensor_type_map[dtype])
    else:
      return tf.io.FixedLenFeature([], _default_type_to_tensor_type_map[dtype])

  def get_metadata(self, input_types: Dict[str, type]):
    """
    Return metadata to be used with tft_beam.AnalyzeAndTransformDataset
    Args:
      input_types: A dictionary of column names and types.
    """
    raise NotImplementedError

  def write_transform_artifacts(self, transform_fn, location):
    """
    Write transform artifacts to the given location.
    Args:
      transform_fn: A transform_fn object.
      location: A location to write the artifacts.
    Returns:
      A PCollection of WriteTransformFn writing a TF transform graph.
    """
    return (
        transform_fn
        | 'Write Transform Artifacts' >>
        transform_fn_io.WriteTransformFn(location))

  def infer_output_type(self, input_type):
    if not isinstance(input_type, RowTypeConstraint):
      row_type = RowTypeConstraint.from_user_type(input_type)
    fields = row_type._inner_types()
    return Dict[str, Union[tuple(fields)]]

  def _get_artifact_location(self, pipeline: beam.Pipeline):
    """
    Return the artifact location. If the pipeline options has staging location
    set, then we will use that as the artifact location. Otherwise, we will
    create a temporary directory and use that as the artifact location.
    Args:
      pipeline: A beam pipeline object.
    Returns:
      A location to write the artifacts.
    """
    # let us get the staging location from the pipeline options
    # and initialize it as the artifact location.
    staging_location = pipeline.options.view_as(
        GoogleCloudOptions).staging_location
    if not staging_location:
      return tempfile.mkdtemp()
    else:
      return staging_location

  def preprocessing_fn(
      self, inputs: Dict[str, common_types.ConsistentTensorType]
  ) -> Dict[str, common_types.ConsistentTensorType]:
    """
    A preprocessing_fn which should be implemented by subclasses
    of TFTProcessHandlers. In this method, tft data transforms
    such as scale_0_to_1 functions are called.
    Args:
      inputs: A dictionary of column names and associated data.
    """
    raise NotImplementedError


class TFTProcessHandlerDict(
    TFTProcessHandler[tft_process_handler_dict_input_type, beam.Row]):
  """
    A subclass of TFTProcessHandler specifically for handling
    data in dictionary format. Applies TensorFlow Transform (TFT)
    operations to the input data.

    This only works on the Schema'd PCollection. Please refer to
    https://beam.apache.org/documentation/programming-guide/#schemas
    for more information on Schema'd PCollection.

    Currently, there are two ways to define a schema for a PCollection:

    1) Register a `typing.NamedTuple` type to use RowCoder, and specify it as
      the output type. For example::

        Purchase = typing.NamedTuple('Purchase',
                                    [('item_name', unicode), ('price', float)])
        coders.registry.register_coder(Purchase, coders.RowCoder)
        with Pipeline() as p:
          purchases = (p | beam.Create(...)
                        | beam.Map(..).with_output_types(Purchase))

    2) Produce `beam.Row` instances. Note this option will fail if Beam is
      unable to infer data types for any of the fields. For example::

        with Pipeline() as p:
          purchases = (p | beam.Create(...)
                        | beam.Map(lambda x: beam.Row(item_name=unicode(..),
                                                      price=float(..))))
    In the schema, TFTProcessHandlerDict accepts the following types:
    1. Primitive types: int, float, str, bytes
    2. List of the primitive types.
    3. Numpy arrays.

    For any other types, TFTProcessHandler will raise a TypeError.
  """
  def _validate_input_types(self, input_types: Dict[str, type]):
    """
    Validate the input types.
    Args:
      input_types: A dictionary of column names and types.
    Returns:
      True if the input types are valid, False otherwise.
    """
    # Fail for the cases like
    # Union[List[int], float] and List[List[int]]
    _valid_types_in_container = (int, str, bytes, float)
    for _, typ in input_types.items():
      if hasattr(typ, '__args__'):
        args = typ.__args__
        if len(args) > 1 and args[0] not in _valid_types_in_container:
          return False
    return True

  def get_input_types(self, element_type) -> Dict[str, type]:
    """
    Return a dictionary of column names and types.
    Args:
      element_type: A type of the element. This could be a NamedTuple or a Row.
    Returns:
      A dictionary of column names and types.
    """
    row_type = None
    if not isinstance(element_type, RowTypeConstraint):
      row_type = RowTypeConstraint.from_user_type(element_type)
      if not row_type:
        raise TypeError(
            "Element type must be compatible with Beam Schemas ("
            "https://beam.apache.org/documentation/programming-guide/#schemas)"
            " for to use with MLTransform and TFTProcessHandlerDict.")
    else:
      row_type = element_type
    inferred_types = {name: typ for name, typ in row_type._fields}
    return inferred_types

  def preprocessing_fn(
      self, inputs: Dict[str, common_types.ConsistentTensorType]
  ) -> Dict[str, common_types.ConsistentTensorType]:
    """
    This method is used in the AnalyzeAndTransformDataset step. It applies
    the transforms to the `inputs` in sequential order on the columns
    provided for a given transform.
    Args:
      inputs: A dictionary of column names and data.
    Returns:
      A dictionary of column names and transformed data.
    """
    outputs = inputs.copy()
    for transform in self.transforms:
      columns = transform.columns
      for col in columns:
        if transform.has_artifacts:
          artifacts = transform.get_analyzer_artifacts(
              inputs[col], col_name=col)
          for key, value in artifacts.items():
            outputs[key] = value
        intermediate_result = transform.apply(outputs[col])
        if transform._save_result:
          outputs[transform._output_name] = intermediate_result
        outputs[col] = intermediate_result
    return outputs

  def _get_processing_data_ptransform(
      self,
      raw_data_metadata: dataset_metadata.DatasetMetadata,
      input_types: Dict[str, type]):
    """
    Return a PTransform object that has the preprocessing logic
    using the AnalyzeAndTransformDataset step.
    """
    @beam.ptransform_fn
    def ptransform_fn(
        raw_data: beam.PCollection[tft_process_handler_dict_input_type]
    ) -> beam.PCollection[MLTransformOutput]:
      """
      Args:
        raw_data: A PCollection of NamedTuples or Rows.
      Returns:
        A PCollection of MLTransformOutput.
      """
      # According to
      # https://www.tensorflow.org/tfx/transform/api_docs/python/tft_beam/Context # pylint: disable=line-too-long
      # context location should be on accessible by all workers.
      # So we will use the staging location for tft_beam.Context.
      # Also, we will be using the same location to store the transform_fn
      # articats.

      self._artifact_location = self._get_artifact_location(raw_data.pipeline)
      with tft_beam.Context(temp_dir=self._artifact_location):
        data = (raw_data, raw_data_metadata)
        transformed_metadata: beam_metadata_io.BeamDatasetMetadata
        (transformed_dataset, transformed_metadata), transform_fn = (
        data
        | "AnalyzeAndTransformDataset" >> tft_beam.AnalyzeAndTransformDataset(
        self.preprocessing_fn,
          )
        )
        self.write_transform_artifacts(transform_fn, self._artifact_location)
        self.transformed_schema = self._get_transformed_data_schema(
            metadata=transformed_metadata.dataset_metadata,
            original_types=input_types)

        transformed_dataset |= (
            beam.Map(
                lambda x: self.convert_to_ml_transform_output(
                    x,
                    transformed_metadata.dataset_metadata,
                    transformed_metadata.asset_map)).with_output_types(
                        MLTransformOutput))

        # TODO: Should we output a Row object or a NamedTuple?
        # TODO: If we are outputting beam.Row, remove the above code
        # converting to NamedTuple.

        # with NamedTuple, I am not able set the output type to the
        # self.transformed_schema. With Row object, we can set the output type
        # to the self.transformed_schema.

        row_type = RowTypeConstraint.from_fields(
            list(self.transformed_schema.items()))
        transformed_dataset |= "ConvertToRowType" >> beam.Map(
            lambda x: beam.Row(**x.transformed_data)).with_output_types(
                row_type)

        # transformed_dataset_pcoll.element_type = row_type
        return transformed_dataset

    return ptransform_fn()

  def process_data(
      self, pcoll: beam.PCollection[tft_process_handler_dict_input_type]
  ) -> beam.PCollection[MLTransformOutput]:
    element_type = pcoll.element_type
    input_types = self.get_input_types(element_type=element_type)

    if not self._validate_input_types(input_types):
      raise RuntimeError(
          "Unable to infer schema. Please pass a schema'd PCollection")

    # AnalyzeAndTransformDataset raise type hint since we only accept
    # schema'd PCollection and the current output type would be a
    # custom type(NamedTuple) or a beam.Row type.
    output_type = self.infer_output_type(element_type)
    raw_data = (
        pcoll | ConvertNamedTupleToDict().with_output_types(output_type))
    raw_data_metadata = self.get_metadata(input_types=input_types)
    return raw_data | self._get_processing_data_ptransform(
        raw_data_metadata=raw_data_metadata, input_types=input_types)

  def _get_transformed_data_schema(
      self,
      metadata: dataset_metadata.DatasetMetadata,
      original_types: Dict[str, type]):
    schema = metadata._schema
    transformed_types = {}
    for feature in schema.feature:
      name = feature.name
      feature_type = feature.type
      is_container = not (
          name in original_types and original_types[name] in _primitive_types)
      if feature_type == schema_pb2.FeatureType.FLOAT:
        transformed_types[name] = np.float32 if is_container else float
      elif feature_type == schema_pb2.FeatureType.INT:
        transformed_types[name] = np.int64 if is_container else int
      elif feature_type == schema_pb2.FeatureType.BYTES:
        transformed_types[name] = bytes
      else:
        # TODO: This else condition won't be hit since TFT doesn't output
        # other than float, int and bytes. Refactor the code here.
        raise RuntimeError(
            'Unsupported feature type: %s encountered' % feature_type)
    return transformed_types

  def convert_to_ml_transform_output(self, element, metadata, asset_map):
    return MLTransformOutput(
        transformed_data=element,
        transformed_metadata=metadata,
        asset_map=asset_map)

  def get_metadata(self, input_types: Dict[str, type]):
    return self.get_raw_data_feature_spec(input_types)
