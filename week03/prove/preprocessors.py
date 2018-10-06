import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

class Processed():
  def __init__(self):
    self.data = np.array([])
    self.target = np.array([])
    self.target_names = np.array([])
    self.feature_names = []

class UCICarPreprocessor():
  def __init__(self):
    self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    self.processed = Processed()
  
  def process(self):
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv(self.url, names=headers, na_values=["?"])
    obj_df = df.select_dtypes(include=['object']).copy()

    # Process doors:    2, 3, 4, 5more.
    # Process persons:  2, 4, more.
    obj_df = pd.get_dummies(obj_df, columns=["doors", "persons"], prefix=["doors", "persons"])

    # Process buying:   vhigh, high, med, low.
    # Process maint:    vhigh, high, med, low.
    # Process lug_boot: small, med, big.
    # Process safety:   low, med, high.
    cleanup = {
      "buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
      "maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
      "lug_boot": {"small": 0, "med": 1, "big": 2},
      "safety": {"low": 0, "med": 1, "high": 2},
    }
    # Process class
    obj_df.replace(cleanup, inplace=True)
    encode_class = {
      "class": {
        "unacc": 0,
        "acc": 1,
        "good": 2,
        "vgood": 3,
      }
    }
    obj_df.replace(encode_class, inplace=True)
    self.processed.target = np.array(obj_df["class"])
    # Remove class column once saved in targets array
    obj_df.drop(columns=['class'], inplace=True)

    self.processed.data = obj_df.values
    self.processed.target_names = np.array(df["class"].unique())

    return self.processed

class AutismSpectrumPreprocessor():
  def __init__(self):
    self.url = ""
    self.process()
  
  def process(self):
    print("Process")
    pass

class AutomobileMPGPreprocessor():
  def __init__(self):
    self.filename = ""
    self.process()
  
  def process(self):
    print("Process")
    pass