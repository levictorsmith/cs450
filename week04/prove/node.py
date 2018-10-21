class Node():
  def __init__(self, value, children=None, is_leaf=False):
    self.value = value
    self.children = {}
    self.is_leaf = is_leaf
    if children is not None:
      for key, node in children.items():
        self.add_child(key, node)
  
  # def __str__(self, level=0):
  #   ret = "\t"*level+repr(self.value)+"\n"
  #   for key, child in self.children:
  #       ret += child.__str__(level+1)
  #   return ret
  # def __repr__(self):
  #   return '<tree node representation>'
  
  def add_child(self, key, node):
    assert isinstance(node, Node)
    self.children[key] = node
