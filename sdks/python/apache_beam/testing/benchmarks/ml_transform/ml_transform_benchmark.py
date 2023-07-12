
import argparse
import os

import logging
import apache_beam as beam

INPUT_GCS_BUCKET_ROOT = 'gs://apache-beam-ml/datasets/cloudml/criteo'
INPUT_CRITEO_SMALL = 'train10.tsv'
INPUT_CRITEO_SMALL_100MB = '100mb/train.txt'
INPUT_CRITEO_10GB = '10gb/train.txt'


def parse_known_args(argv):
  """Parses args for this workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      # required=True,
      default= os.path.join(
        INPUT_GCS_BUCKET_ROOT, INPUT_CRITEO_SMALL),
      help='Input path for input files.')
  # parser.add_argument(
  #     '--output',
  #     dest='output',
  #     required=True,
  #     help='Output path for output files.')
  # parser.add_argument(
  #     '--classifier',
  #     dest='classifier',
  #     required=True,
  #     help='Name of classifier to use.')
  # parser.add_argument(
  #     '--frequency_threshold',
  #     dest='frequency_threshold',
  #     default=5,  # TODO: Align default with TFT (ie 0).
  #     help='Threshold for minimum number of unique values for a category.')
  # parser.add_argument(
  #     '--shuffle',
  #     action='store_false',
  #     dest='shuffle',
  #     default=True,
  #     help='Skips shuffling the data.')
  # parser.add_argument(
  #     '--benchmark_type',
  #     dest='benchmark_type',
  #     required=True,
  #     help='Type of benchmark to run.')
  return parser.parse_known_args(argv)


def setup_pipeline(p, known_args):
  input_data_pcoll = (
    p
    | beam.io.ReadFromText(known_args.input, coder=beam.coders.BytesCoder())
    | beam.Map(logging.info)
  )

def run(argv=None):
  """Main entry point; defines and runs the pipeline."""
  known_args, pipeline_args = parse_known_args(argv)
  with beam.Pipeline(argv=pipeline_args) as p:
    setup_pipeline(p, known_args)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()