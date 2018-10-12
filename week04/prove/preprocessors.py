import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder

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
    self.filename = "./autism_spectrum/Autism-Adult-Data.csv"
    self.headers = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", "age", "gender", "ethnicity", "jundice", "austim", "contry_of_res", "used_app_before", "result", "age_desc", "relation", "class"]
    self.processed = Processed()

  def process(self):
    df = pd.read_csv(self.filename, names=self.headers, na_values=["?"], quotechar="'")
    obj_df = df.copy()
    # Process age {numeric}
    obj_df["age"] = obj_df["age"].fillna(0)
    # Process gender {f,m}
    obj_df = pd.get_dummies(obj_df, columns=["gender"], prefix=["is"])

    # Process ethnicity {White-European,Latino,Others,Black,Asian,'Middle Eastern ',Pasifika,'South Asian',Hispanic,Turkish,others}
    obj_df["ethnicity"] = obj_df["ethnicity"].fillna('')
    hee = HashingEncoder(cols=["ethnicity"])
    hee.fit(obj_df)
    obj_df = hee.transform(obj_df)

    # Process jundice {no,yes}
    # Process austim {no,yes}
    # Process used_app_before {no,yes}
    # Class/ASD {NO,YES}
    replace_bool = {
      "jundice": { "no": 0, "yes": 1 },
      "austim": { "no": 0, "yes": 1 },
      "used_app_before": { "no": 0, "yes": 1 },
      "class": { "NO": 0, "YES": 1 },
    }
    obj_df.replace(replace_bool, inplace=True)
    # Process contry_of_res {'United States',Brazil,Spain,Egypt,'New Zealand',Bahamas,Burundi,Austria,Argentina,Jordan,Ireland,'United Arab Emirates',Afghanistan,Lebanon,'United Kingdom','South Africa',Italy,Pakistan,Bangladesh,Chile,France,China,Australia,Canada,'Saudi Arabia',Netherlands,Romania,Sweden,Tonga,Oman,India,Philippines,'Sri Lanka','Sierra Leone',Ethiopia,'Viet Nam',Iran,'Costa Rica',Germany,Mexico,Russia,Armenia,Iceland,Nicaragua,'Hong Kong',Japan,Ukraine,Kazakhstan,AmericanSamoa,Uruguay,Serbia,Portugal,Malaysia,Ecuador,Niger,Belgium,Bolivia,Aruba,Finland,Turkey,Nepal,Indonesia,Angola,Azerbaijan,Iraq,'Czech Republic',Cyprus}
    obj_df["contry_of_res"] = obj_df["contry_of_res"].fillna('')
    hec = HashingEncoder(cols=["contry_of_res"])
    hec.fit(obj_df)
    obj_df = hec.transform(obj_df)

    # Process age_desc {'18 and more'}
    obj_df.drop(columns=["age_desc"], inplace=True)

    # Process relation {Self,Parent,'Health care professional',Relative,Others}
    obj_df["relation"] = obj_df["relation"].fillna('')
    lb_relation = LabelEncoder()
    obj_df["relation"] = lb_relation.fit_transform(obj_df["relation"])

    self.processed.data = obj_df.values
    self.processed.target = np.array(obj_df["class"])
    self.processed.target_names = np.array(df["class"].unique())
    return self.processed

class AutomobileMPGPreprocessor():
  def __init__(self):
    self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    self.headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
    self.processed = Processed()
  
  def process(self):
    df = pd.read_csv(self.url, names=self.headers, na_values=["?"], quotechar="\"", delim_whitespace=True)
    obj_df = df.copy()
    # Process mpg
    obj_df['mpg'] = obj_df['mpg'].astype(int)

    # Process horsepower
    obj_df["horsepower"] = obj_df['horsepower'].fillna(obj_df["horsepower"].mean())

    # Process car name
    obj_df.drop(columns=["car_name"], inplace=True)

    self.processed.data = obj_df.values
    self.processed.target = np.array(obj_df["mpg"])
    self.processed.target_names = np.array(obj_df["mpg"].unique())
    return self.processed
