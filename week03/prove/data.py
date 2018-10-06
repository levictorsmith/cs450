from sklearn import datasets
import numpy as np
import pandas as pd
import preprocessors

class Dataset():
  def __init__(self, preprocessor=None):
    self.preprocessor = preprocessor
    if self.preprocessor:
      self.preprocess_data()
    else:
      self.load_sample_data()

  def load_sample_data(self):
    self.data = datasets.load_iris()

  def preprocess_data(self):
    if not self.preprocessor:
      raise ValueError('No Pre-processor specified')
    else:
      preprocessor = self.get_preprocessor()
      self.data = preprocessor.process()

  def get_preprocessor(self):
    return {
      'UCICar': preprocessors.UCICarPreprocessor,
      'AutismSpectrum': preprocessors.AutismSpectrumPreprocessor,
      'AutomobileMPG': preprocessors.AutomobileMPGPreprocessor,
    }[self.preprocessor]()
  
  def get_data(self):
    return self.data