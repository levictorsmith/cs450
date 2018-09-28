import numpy as np

"""
Hard-coded Classifier
"""
class HardCodedClassifier():
  def fit(self, training_data, training_targets):
    return HardCodedModel()

class HardCodedModel():
  def predict(self, test_data):
    results = [0 for _ in test_data]
    return results

"""
K-Nearest Neighbor Classifier
"""
class KNeighborsClassifier():
  def __init__(self, neighbors):
    self.neighbors = neighbors

  def fit(self, training_data, training_targets):
    self.data = training_data
    self.targets = training_targets
    return KNeighborsModel()

class KNeighborsModel():
  def predict(self, test_data):
    results = [0 for _ in test_data]
    return results
  
  def get_distance(self):
    pass
