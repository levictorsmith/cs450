from sklearn import datasets

class Dataset():
  def __init__(self, filename=None):
    self.filename = filename
    if self.filename != None and self.filename != '':
      self.read_file(self.filename)
    else:
      self.load_sample_data()

  def load_sample_data(self):
    self.data = datasets.load_iris()

  def read_file(self, filename):
    # TODO: Read in the file
    self.data = ()
  
  def get_data(self):
    return self.data