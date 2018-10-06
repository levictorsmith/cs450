import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

###############################
# Hard-coded Classifier
###############################
class HardCodedClassifier():
  def fit(self, training_data, training_targets):
    return HardCodedModel()

class HardCodedModel():
  def predict(self, test_data):
    results = [0 for _ in test_data]
    return results

###############################
# K-Nearest Neighbor Classifier
###############################
class KNeighborsClassifier():
  def __init__(self, neighbors):
    self.neighbors = neighbors

  def fit(self, training_data, training_targets):
    self.data = training_data
    self.targets = training_targets
    std_scale = StandardScaler().fit(self.data)
    standardized = std_scale.transform(self.data)
    return KNeighborsModel(standardized, self.targets, self.neighbors)

class KNeighborsModel():
  def __init__(self, training_data, training_targets, neighbors):
    self.data = training_data
    self.targets = training_targets
    self.neighbors = neighbors
    pass

  def predict(self, test_data):
    std_scale = StandardScaler().fit(test_data)
    standarized = std_scale.transform(test_data)

    nearest = []
    for x in standarized:
      distances = [self.get_distance(x, n) for n in self.data]
      neighbors = self.get_nearest_neighbors(distances, self.targets)
      nearest.append(np.argmax(np.bincount(neighbors)))
    return np.array(nearest)
  
  def get_distance(self, a, b):
    return np.sum((a - b) ** 2)
  
  def get_nearest_neighbors(self, data, targets):
    indices = np.argsort(data, axis=0)
    return targets[indices[:self.neighbors]]
