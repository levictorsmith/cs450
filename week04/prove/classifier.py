import sys
import numpy as np
import pandas as pd
from math import log2
from node import Node
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

###############################
# Decision Tree Classifier
###############################
class DecisionTreeClassifier():
  def __init__(self):
    self.tree = None
    self.data = None
    self.targets = pd.Series([])
    pass

  # Returns a list of branch-target tuples 
  def get_targets(self, subset, subset_targets, attribute):
    if subset is not pd.DataFrame:
        subset = pd.DataFrame(subset)
    return [(k, np.take(subset_targets, v)) for k, v in subset.groupby(attribute).groups.items()]

  # Call for each category
  def get_entropy(self, feat_targ_tuples, total_len):
    entropies = []
    for feat in feat_targ_tuples:
    # Gather the lists of classes
      targets = pd.Series(feat[1])
      targets_len = len(targets)
      lengths = [len(group) for _, group in targets.groupby(targets)]
      entropies.append((targets_len, -(np.sum([(length/targets_len) * log2(length/targets_len) for length in lengths]))))
    return np.sum([(e[0]/total_len) * e[1] for e in entropies])

  # Pseudocode:
  # If all examples (are the same)/have the same label
  #     return a leaf with that label
  # Else if there are no features left to test
  #     return a leaf with the most common label
  # Else
  #     Consider each available feature
  #     Choose the one that maximizes information gain
  #     Create a new node for that feature

  #     For each possible value of the feature
  #         Create a branch for this value
  #         Create a subset of the examples for each branch
  #         Recursively call the function to create a new node at that branch  
  def build_tree(self, features, subset, subset_t, column_values):
    # print('Build tree')
    examples = subset_t
    if (examples == examples[0]).all():
      return Node(examples[0], None, is_leaf=True)
    elif len(features) == 0:
      return Node(np.argmax(np.bincount(examples)), None, is_leaf=True)
    else:
      best = (-1, sys.maxsize)
      for feature in features:
        subset_size = len(subset[feature])
        examples = self.get_targets(subset, subset_t, feature)
        entropy = self.get_entropy(examples, subset_size)
        best = (feature, entropy) if entropy < best[1] else best
      print(best)
      # for value in column_values[best[0]]:
      #   print(list(filter(lambda x: x != best[0], features)))
      #   print(pd.DataFrame(subset.loc[subset[best[0]] == value]).reset_index(drop=True))
      #   print(np.take(subset_t, subset.index[subset[best[0]] == value]))
        
      return Node(best[0], children={
        value: self.build_tree(list(filter(lambda x: x != best[0], features)), pd.DataFrame(subset.loc[subset[best[0]] == value]).reset_index(drop=True), np.take(subset_t, subset.index[subset[best[0]] == value]), column_values)
        for value in column_values[best[0]]
      })
  
  def fit(self, training_data, training_targets, feature_names=[]):
    self.data = pd.DataFrame(data=training_data, columns=feature_names)
    self.targets = training_targets
    column_values = {}
    for col in feature_names:
      column_values[col] = np.unique(self.data[col])
    self.tree = self.build_tree(feature_names, self.data, self.targets, column_values)
    return DecisionTreeModel(self.tree)

class DecisionTreeModel():
  def __init__(self, training_tree):
    self.tree = training_tree
    pass
  
  def predict(self, test_data):
    pass
