from functools import reduce
import inspect

def bind(func, *bound_args, **bound_kws):
   #func_args = inspect.getargspec(func).args   # deprecated
   func_args = [a for a in inspect.signature(func).parameters]
   if not bound_kws.keys().isdisjoint(func_args[:len(bound_args)]):
      raise Exception('Got ambiguous arguments.')
   bound_kws.update(zip(func_args, bound_args))
   for key in bound_kws.keys():
      func_args.remove(key)

   def bound_func(*args, **kws):
      kws.update(zip(func_args, args))
      return func(**bound_kws, **kws)

   return bound_func


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
ARGS = INPUTS
_ = INPUTS

 
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
