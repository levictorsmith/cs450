import pandas as pd
import numpy as np


def main():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "wage"]
  data = pd.read_csv(url, names=cols, skipinitialspace=True, na_values=['?'])
  # data.assign(is_female=lambda x: x.sex)
  data['is_female'] = [1 if gender == 'Female' else 0 for gender in data.get('sex')]

  print(data.head(5))
  
  pass

if __name__ == '__main__':
  main()