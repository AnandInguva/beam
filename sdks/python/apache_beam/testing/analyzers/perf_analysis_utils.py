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

from dataclasses import asdict
from dataclasses import dataclass
import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import yaml
from google.api_core import exceptions

from apache_beam.testing.analyzers import constants
from apache_beam.testing.analyzers.github_issues_utils import get_issue_description
from apache_beam.testing.analyzers.github_issues_utils import report_change_point_on_issues
from apache_beam.testing.load_tests import load_test_metrics_utils
from apache_beam.testing.load_tests.load_test_metrics_utils import BigQueryMetricsPublisher
from apache_beam.testing.load_tests.load_test_metrics_utils import big_query_metrics_fetcher
from signal_processing_algorithms.energy_statistics.energy_statistics import e_divisive


@dataclass
class GitHubIssueMetaData:
  """
  This class holds metadata that needs to be published to the
  BigQuery when a GitHub issue is created on a performance
  alert.
  """
  issue_timestamp: pd.Timestamp
  change_point_timestamp: pd.Timestamp
  test_name: str
  metric_name: str
  issue_number: int
  issue_url: str
  test_id: str
  change_point: float


def is_change_point_in_valid_window(
    num_runs_in_change_point_window: int, change_point_index: int) -> bool:
  # If the change point is more than N runs behind the most recent run,
  # Ignore the change point and don't raise an alert for it.
  return num_runs_in_change_point_window >= change_point_index


def get_existing_issues_data(test_name: str, ) -> Optional[pd.DataFrame]:
  """
  Finds the most recent GitHub issue created for the test_name.
  If no table found with name=test_name, return (None, None)
  else return latest created issue_number along with
  """
  query_template = f"""
  SELECT * FROM {constants._BQ_PROJECT_NAME}.{constants._BQ_DATASET}.{test_name}
  ORDER BY {constants._ISSUE_CREATION_TIMESTAMP_LABEL} DESC
  LIMIT 10
  """
  try:
    df = big_query_metrics_fetcher(query_template=query_template)
  except exceptions.NotFound:
    # If no table found, that means this is first performance regression
    # on the current test+metric.
    return None
  return df


def is_perf_alert(
    previous_change_point_timestamps: List[pd.Timestamp],
    change_point_index: int,
    timestamps: List[pd.Timestamp],
    min_runs_between_change_points: int) -> bool:
  """
  Search the previous_change_point_timestamps with current observed
  change point sibling window and determine if it is a duplicate
  change point or not.

  Return False if the current observed change point is a duplicate of
  already reported change points else return True.
  """
  sibling_change_point_min_timestamp = timestamps[min(
      change_point_index + min_runs_between_change_points, len(timestamps) - 1)]
  sibling_change_point_max_timestamp = timestamps[max(
      0, change_point_index - min_runs_between_change_points)]
  # Search a list of previous change point timestamps and compare it with
  # current change point timestamp. We do this in case, if a current change
  # point is already reported in the past.
  for previous_change_point_timestamp in previous_change_point_timestamps:
    if (sibling_change_point_min_timestamp <= previous_change_point_timestamp <=
        sibling_change_point_max_timestamp):
      return False
  return True


def read_test_config(config_file_path: str) -> Dict:
  """
  Reads the config file in which the data required to
  run the change point analysis is specified.
  """
  with open(config_file_path, 'r') as stream:
    config = yaml.safe_load(stream)
  return config


def validate_config(keys):
  return constants._PERF_TEST_KEYS.issubset(keys)


def fetch_metric_data(
    params: Dict[str,
                 Any]) -> Tuple[List[Union[int, float]], List[pd.Timestamp]]:
  metric_data: pd.DataFrame = big_query_metrics_fetcher(
      project_name=params['project'],
      dataset=params['metrics_dataset'],
      table=params['metrics_table'],
      metric_name=params['metric_name'],
      limit=constants._NUM_DATA_POINTS_TO_RUN_CHANGE_POINT_ANALYSIS)
  return (
      metric_data[load_test_metrics_utils.VALUE_LABEL],
      metric_data[load_test_metrics_utils.SUBMIT_TIMESTAMP_LABEL])


def find_latest_change_point_index(metric_values: List[Union[float, int]]):
  change_points_idx = e_divisive(metric_values)
  if not change_points_idx:
    return None
  # Consider the latest change point.
  change_points_idx.sort(reverse=True)
  change_point_index = change_points_idx[0]
  return change_point_index


def publish_issue_metadata_to_big_query(issue_metadata, test_name):
  """
  Published issue_metadata to BigQuery with table name=test_name.
  """
  bq_metrics_publisher = BigQueryMetricsPublisher(
      project_name=constants._BQ_PROJECT_NAME,
      dataset=constants._BQ_DATASET,
      table=test_name,
      bq_schema=constants._SCHEMA)
  bq_metrics_publisher.publish([asdict(issue_metadata)])
  logging.info(
      'GitHub metadata is published to Big Query Dataset %s'
      ', table %s' % (constants._BQ_DATASET, test_name))


def create_performance_alert(
    metric_name: str,
    test_name: str,
    timestamps: List[pd.Timestamp],
    metric_values: List[Union[int, float]],
    change_point_index: int,
    labels: List[str],
    existing_issue_number: Optional[int]) -> Tuple[int, str]:
  """
  Creates performance alert on GitHub issues and returns GitHub issue
  number and issue URL.
  """
  description = get_issue_description(
      metric_name=metric_name,
      timestamps=timestamps,
      metric_values=metric_values,
      change_point_index=change_point_index,
      max_results_to_display=(
          constants._NUM_RESULTS_TO_DISPLAY_ON_ISSUE_DESCRIPTION))

  issue_number, issue_url = report_change_point_on_issues(
        title=constants._TITLE_TEMPLATE.format(test_name, metric_name),
        description=description,
        labels=labels,
        existing_issue_number=existing_issue_number)

  logging.info(
      'Performance regression is alerted on issue #%s. Link to '
      'the issue: %s' % (issue_number, issue_url))
  return issue_number, issue_url
