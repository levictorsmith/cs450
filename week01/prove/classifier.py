import numpy as np

class HardCodedClassifier():
  def fit(self, training_data, training_targets):
    return HardCodedModel()

class HardCodedModel():
  def predict(self, test_data):
    results = [0 for _ in test_data]
    return results