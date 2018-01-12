from functools import reduce


def _compose2(f, g):
   return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
   return reduce(_compose2, fs)



class Input:

   def __init__(self):
      self.functions = []

   def __rshift__(self, function):
      input = Input()
      input.functions = self.functions.copy()
      input.functions.append(function)
      return input

   def __call__(self, *a, **kw):
      return compose(*self.functions[::-1])(*a, **kw)


INPUTS = Input()
ARGS = Input()

 
# class Pipeable:
# 
#    def __init__(self, super_obj):
#       self.__class__ = type(
#          super_obj.__class__.__name__,
#          (self.__class__, super_obj.__class__),
#          {}
#       )
#       self.__dict__ = super_obj.__dict__
#       # super(Pipeable, self).__init__(**kwargs)
# 
#    def __rshift__(self, function):
#       return ARGS >> self >> function
