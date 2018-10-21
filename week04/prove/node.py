class Node():
  def __init__(self, value, children=None, is_leaf=False):
    self.value = value
    self.children = {}
    self.is_leaf = is_leaf
    if children is not None:
      for key, node in children.items():
        self.add_child(key, node)
  
  def add_child(self, key, node):
    assert isinstance(node, Node)
    self.children[key] = node
