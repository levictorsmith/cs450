import sys
import numpy as np
from args import parse_args
from data import Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from classifier import HardCodedClassifier, HardCodedModel

def get_algorithm(algorithm):
  return {
    'hard_coded': HardCodedClassifier(),
    'naive_bayes': GaussianNB(),
  }[algorithm]

def get_train_test_data(filename, test_size):
  data = Dataset(filename).get_data()
  return train_test_split(data.data, data.target, test_size=test_size)

def main():
  args = parse_args()
  classifier = get_algorithm(args.algorithm)
  data_train, data_test, target_train, target_test = get_train_test_data(filename=args.file, test_size=args.test_size)
  model = classifier.fit(data_train, target_train)
  targets_predicted = model.predict(data_test)
  correct = 1 - np.mean(target_test != targets_predicted)
  print("Accuracy: {:.2%}".format(correct))

if __name__ == '__main__':
  main()