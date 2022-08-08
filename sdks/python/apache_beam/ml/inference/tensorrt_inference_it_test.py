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

import unittest

import pytest

from apache_beam.testing.test_pipeline import TestPipeline

# Protect against environments where TensorRT python library is not available.
# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports

from apache_beam.examples.inference import tensorrt_object_detection


class TensorRTRunInferenceTest(unittest.TestCase):
  @pytest.mark.it_postcommit
  @pytest.mark.uses_tensorrt
  def test_tensorrt_object_detection(self):
    test_pipeline = TestPipeline(is_integration_test=True)
    # TODO: change the input, output to a different bucket
    input = "gs://apache-beam-testing-yeandy/tensorrt_image_file_names.txt"
    output = "gs://apache-beam-ml/tmp/tensorrt_predictions.txt"
    engine_path = "gs://apache-beam-ml/models/ssd_mobilenet_v2_320x320_coco17_tpu-8.trt"  # pylint: disable=line-too-long
    extra_opts = {
        "input": input,
        "output": output,
        "engine_path": engine_path,
        "experiments": "use_runner_v2,worker_accelerator=type:nvidia-tesla-t4;count:2;install-nvidia-driver"  # pylint: disable=line-too-long
    }

    tensorrt_object_detection.run(
        test_pipeline.get_full_options_as_args(**extra_opts),
        save_main_session=False)
