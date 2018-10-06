import sys
import argparse

DEFAULT_TEST_SIZE = 0.3
DEFAULT_ALGORITHM = 'hard_coded'
DEFAULT_NUM_NEIGHBORS = 3

def parse_args():
  # Args: 
  # -f File to load data from 
  # -a Algorithm to use
  # -s Test size (Value 0.0 - 1.0)
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-f',
    '--file',
    help='File to load data from. Valid file types: .csv, .txt',
    default='',
  )
  parser.add_argument(
    '-a',
    '--algorithm',
    help='',
    default=DEFAULT_ALGORITHM
  )
  parser.add_argument(
    '-s',
    '--test_size',
    help='Percentage of data used for testing. i.e. 0.3',
    default=DEFAULT_TEST_SIZE,
  )
  parser.add_argument(
    '-k',
    '--neighbors',
    help='Number of neighbors',
    default=DEFAULT_NUM_NEIGHBORS
  )
  return parser.parse_args()