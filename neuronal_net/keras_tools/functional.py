from functools import reduce


def _compose2(f, g):
   return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
   return reduce(_compose2, fs)



class Input:

   def __init__(self):
      self.functions = []

   def __rshift__(self, function):
      self.functions.append(function)
      return self

   def __call__(self, *a, **kw):
      return compose(*self.functions[::-1])(*a, **kw)
