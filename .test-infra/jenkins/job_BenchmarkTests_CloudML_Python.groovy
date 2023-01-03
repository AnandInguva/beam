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
import PhraseTriggeringPostCommitBuilder
import CronJobBuilder

def cloudMLJob = { scope ->
  scope.description('Runs the TFT Criteo Examples on the Dataflow runner.')

  // Set common parameters.
  commonJobProperties.setTopLevelMainJobProperties(scope)

  // Gradle goals for this job.
  scope.steps {
    gradle {
      rootBuildScriptDir(commonJobProperties.checkoutDir)
      commonJobProperties.setGradleSwitches(delegate, 'master', 360)
      tasks(':sdks:python:test-suites:dataflow:tftTests')
    }
  }
}

CronJobBuilder.cronJob(
    'beam_PostCommit_Python_CloudML_tests',
    'H 14 * * *',
    this
    ) {
      cloudMLJob(delegate)
    }
