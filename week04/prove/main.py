import sys
import numpy as np
from args import parse_args
from data import Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as LibraryKNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as LibraryDecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from classifier import HardCodedClassifier, HardCodedModel, KNeighborsClassifier, KNeighborsModel, DecisionTreeClassifier, DecisionTreeModel

def get_algorithm(args):
  algorithm = args.algorithm
  return {
    'hard_coded': HardCodedClassifier(),
    'naive_bayes': GaussianNB(),
    'k_nearest': KNeighborsClassifier(int(args.neighbors)),
    'decision_tree': DecisionTreeClassifier(),
  }[algorithm]

def get_library_version(args):
  algorithm = args.algorithm
  return {
    'k_nearest': LibraryKNeighborsClassifier(int(args.neighbors)),
    'decision_tree': LibraryDecisionTreeClassifier(),
  }[algorithm]

def get_data(preprocessor):
  data = Dataset(preprocessor).get_data()
  return data

def get_train_test_data(preprocessor, test_size):
  data = get_data(preprocessor)
  return train_test_split(data.data, data.target, test_size=test_size)

def main():
  args = parse_args()
  data_train, data_test, target_train, target_test = get_train_test_data(preprocessor=args.preprocessor, test_size=args.test_size)

  classifier = get_algorithm(args)
  temp = get_data(args.preprocessor)
  model = classifier.fit(data_train, target_train, temp.feature_names)
  targets_predicted = model.predict(data_test)
  correct = 1 - np.mean(target_test != targets_predicted)
  print("Accuracy: {:.2%}".format(correct))

  lib_classifier = get_library_version(args)
  lib_model = lib_classifier.fit(data_train, target_train)
  lib_targets_predicted = lib_model.predict(data_test)
  lib_correct = 1 - np.mean(target_test != lib_targets_predicted)
  print("Library Accuracy: {:.2%}".format(lib_correct))

  print("Difference from Library version: {:+.2%}".format(correct - lib_correct))

  # K-Fold Cross Validation
  data = get_data(args.preprocessor)
  k_fold = KFold(n_splits=len(data.data), shuffle=True, random_state=7)
  k_fold_accuracy = cross_val_score(lib_classifier, data.data, data.target, cv=k_fold, n_jobs=1, scoring='accuracy').mean()
  print("K-Fold Accuracy: {:.2%}".format(k_fold_accuracy))

if __name__ == '__main__':
  main()