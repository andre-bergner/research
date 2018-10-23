import time

class Timer:

   def __enter__(self):
      self.t1 = time.time()
      return self

   def __exit__(self,t,v,tb):
      t2 = time.time()
      print("{0:.4f} seconds".format(t2-self.t1))
