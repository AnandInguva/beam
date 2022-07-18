/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import CommonJobProperties as commonJobProperties
import LoadTestsBuilder as loadTestsBuilder
import PhraseTriggeringPostCommitBuilder
import CronJobBuilder

def now = new Date().format("MMddHHmmss", TimeZone.getTimeZone('UTC'))

def loadTestConfigurations = {
  ->
  [
    // Benchmark test config. Add multiple configs for multiple models.
    // (TODO): Add model name to experiments once decided on which models to use.
    [
      title             : 'Pytorch Imagenet Classification',
      test              : 'apache_beam.testing.benchmarks.inference.pytorch_benchmarks',
      runner            : CommonTestProperties.Runner.DATAFLOW,
      pipelineOptions: [
        job_name              : 'benchmark-tests-pytorch-imagenet-python' + now,
        project               : 'apache-beam-testing',
        region                : 'us-central1',
        staging_location      : 'gs://temp-storage-for-perf-tests/loadtests',
        temp_location         : 'gs://temp-storage-for-perf-tests/loadtests',
        requirements_file     : 'apache_beam/ml/inference/torch_tests_requirements.txt',
        publish_to_big_query  : true,
        metrics_dataset       : 'beam_run_inference',
        metrics_table         : 'torch_inference_imagenet_results_resnet50',
        input_options         : '{}', // this option is not required for RunInference tests.
        influx_measurement    : 'torch_inference_imagenet_resnet50',
        influx_db_name        : InfluxDBCredentialsHelper.InfluxDBDatabaseName,
        influx_hostname       : InfluxDBCredentialsHelper.InfluxDBHostUrl,
        // args defined in the performance test
        pretrained_model_name : 'resnet50',
        // args defined in the example.
        input_file            : 'gs://apache-beam-ml/testing/inputs/imagenet_validation_inputs.txt',
        // TODO: make sure the model_state_dict_path weights are accurate.
        model_state_dict_path : 'gs://apache-beam-ml/models/torchvision.models.resnet50.pth',
        output                : 'gs://temp-storage-for-end-to-end-tests/torch/result_' + now + '.txt'
      ]
    ],
    [
      title             : 'Pytorch Lanugaue Modeling using Hugging face bert-base-uncased model',
      test              : 'apache_beam.testing.benchmarks.inference.pytorch_language_modeling_benchmarks',
      runner            : CommonTestProperties.Runner.DATAFLOW,
      pipelineOptions: [
        job_name              : 'benchmark-tests-pytorch-language-modeling' + now,
        project               : 'apache-beam-testing',
        region                : 'us-central1',
        staging_location      : 'gs://temp-storage-for-perf-tests/loadtests',
        temp_location         : 'gs://temp-storage-for-perf-tests/loadtests',
        requirements_file     : 'apache_beam/ml/inference/torch_tests_requirements.txt',
        publish_to_big_query  : true,
        metrics_dataset       : 'beam_run_inference',
        metrics_table         : 'torch_language_modeling_bert_uncased',
        input_options         : '{}', // this option is not required for RunInference tests.
        influx_measurement    : 'torch_language_modeling_bert_uncased',
        influx_db_name        : InfluxDBCredentialsHelper.InfluxDBDatabaseName,
        influx_hostname       : InfluxDBCredentialsHelper.InfluxDBHostUrl,
        // args defined in the example.
        input_file            : 'gs://apache-beam-ml/testing/inputs/sentences_50k.txt',
        // TODO: make sure the model_state_dict_path weights are accurate.
        model_state_dict_path : 'gs://apache-beam-ml/models/huggingface.BertForMaskedLM.bert-base-uncased.pth',
        output                : 'gs://temp-storage-for-end-to-end-tests/torch/result_' + now + '.txt'
      ]
    ],
  ]
}

def loadTestJob = { scope ->
  List<Map> testScenarios = loadTestConfigurations()
  for (Map testConfig: testScenarios){
    commonJobProperties.setTopLevelMainJobProperties(scope, 'master', 180)
    loadTestsBuilder.loadTest(scope, testConfig.title, testConfig.runner, CommonTestProperties.SDK.PYTHON, testConfig.pipelineOptions, testConfig.test, null, testConfig.pipelineOptions.requirements_file)
  }
}

PhraseTriggeringPostCommitBuilder.postCommitJob(
    'beam_Inference_Python_Benchmarks_Dataflow',
    'Run Inference Benchmarks',
    'Inference benchmarks on Dataflow(\"Run Inference Benchmarks"\"")',
    this
    ) {
      loadTestJob(delegate)
    }

// TODO(anandinguva): Change the cron job to run once a day
CronJobBuilder.cronJob(
    'beam_Inference_Python_Benchmarks_Dataflow', 'H 2 * * *',
    this
    ) {
      loadTestJob(delegate)
    }
