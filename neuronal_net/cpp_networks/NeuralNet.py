import numpy as np
from itertools import accumulate
import cmath




def sigmoid(x):        return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):  return sigmoid(x) * (1-sigmoid(x))

def sigmoid_and_deriv(x):
   s = sigmoid(x)
   return  s, s*(1-s)

# 1/(1+exp(x)) = (1-exp(x))^-1
# ∂x : -(1-exp(x))^-2 * (-exp(x))
#    = exp(x) / (1-exp(x))^2

def nabla_cost(v,y):
   return  v - y




class NeuralNet:

   def __init__(self, sizes, func=sigmoid, dfunc=sigmoid_deriv ):

      np.random.seed(1337)

      self.num_layers = len(sizes)
      self.sizes      = sizes
      #self.biases     = [ np.random.randn(s) for s in sizes[1:] ]
      #self.weights    = [ np.random.randn(m,n) for n,m in zip(sizes[:-1],sizes[1:]) ]
      #self.biases     = [ 0.00001*np.ones(s) for s in sizes[1:] ]
      #self.weights    = [ 0.00001*np.ones((m,n)) for n,m in zip(sizes[:-1],sizes[1:]) ]
      #self.weights    = [ ((-1.)**np.arange(0,n*m)).reshape((m,n)) for n,m in zip(sizes[:-1],sizes[1:]) ]
      self.biases     = [ np.real(cmath.rect(1,.2)**np.arange(1,s+1)) for s in sizes[1:] ]
      self.weights    = [ np.real(cmath.rect(1,.2)**np.arange(1,n*m+1)).reshape((n,m)).T for n,m in zip(sizes[:-1],sizes[1:]) ]
      self.func       = np.vectorize(func)
      self.dfunc      = np.vectorize(dfunc)

   def layer(self,w,b,x):
      u = np.dot(w,x) + b   # node input
      return  u, self.func(u)

   def __call__(self, x):
      for w,b in zip( self.weights, self.biases ):
         u,x = self.layer(w,b,x)
      return x


   # -------------------
   # learning part

   def back_propagate(self, x, y):

      # 1. feed forward an store all node input and outputs

      v = x
      layer_inputs = []
      layer_outputs = [v]
      for w,b in zip( self.weights, self.biases ):
         u,v = self.layer(w,b,v)
         layer_inputs.append(u)
         layer_outputs.append(v)

#      print("==========================")
#      for i,o in zip(layer_inputs,layer_outputs[:-1]):
#         print(o)
#         print(i)
#      print("==========================")

      # 2. back propagate error to compute all partial derivatives

      error = nabla_cost(layer_outputs[-1],y) * self.dfunc(layer_inputs[-1])
      dws = [ np.outer(error, layer_outputs[-2]) ]
      dbs = [ error ]

      for w,u,v in zip( self.weights[::-1], layer_inputs[-2::-1], layer_outputs[-3::-1] ):
         error = np.dot(w.T,error) * self.dfunc(u)
         dws.append( np.outer(error,v) )
         dbs.append( error )

      return  dws[::-1] , dbs[::-1]


   def step_gradient_descent(self, pairs, eta=0.1):

      dws = [ np.zeros(w.shape) for w in self.weights ]
      dbs = [ np.zeros(b.shape) for b in self.biases ]

      for x,y in pairs:
         dws_,dbs_ = self.back_propagate(x,y)
         for dw,dw_ in zip(dws,dws_): dw += dw_
         for db,db_ in zip(dbs,dbs_): db += db_

      for w,dw in zip(self.weights,dws): w -= dw * eta / len(pairs)
      for b,db in zip(self.biases,dbs):  b -= db * eta / len(pairs)


   def stochastic_gradient_descent(self, training_data, n_epochs=100, mini_batch_size=20, eta=0.1):

      n = len(training_data)

      for j in range(n_epochs):

         #np.random.shuffle(training_data)
         mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

         for mini_batch in mini_batches:
            self.step_gradient_descent( mini_batch, eta )

         print("\rEpoch {0}/{1} complete".format(j,n_epochs), end="", flush=True)

      print("")


#    def evaluate(self, test_data):
#        test_results = [(np.argmax(self.feedforward(x)), y)
#                        for (x, y) in test_data]
#        return sum(int(x == y) for (x, y) in test_results)




def spice(n=4): return np.random.random(n)*0.2

# basic training "images"
#
#  1   2   3   4
#  o_  oo  _o  o_
#  o_  __  o_  _o
#

base_training_data = [ ([ 1., 0., 1., 0. ], [1., 0., 0., 0.])
                     , ([ 1., 1., 0., 0. ], [0., 1., 0., 0.])
                     , ([ 0., 1., 1., 0. ], [0., 0., 1., 0.])
                     , ([ 1., 0., 0., 1. ], [0., 0., 0., 1.])
                     ]

training_data = [ (spice()+x,y) for x,y in np.repeat( base_training_data, 100, axis=0 ) ]
#np.random.shuffle(training_data)

net = NeuralNet([4,2,4])
#for k in range(400): net.gradien_descent(training_data,1)



#training_data = [ (spice(1)+x,y) for x,y in np.repeat( [ ([ 0.], [1.]), ([ 1.], [0.]) ], 100, axis=0 ) ]
#np.random.shuffle(training_data)

#n1 = NeuralNet([1,1,1])
#for k in range(400): n1.gradien_descent(training_data,1)
#print(n1([0]))
#print(n1([1]))

#n2 = Network([1,1,1])
#for k in range(400): n2.update_mini_batch(training_data,1)
#print(n2([0]))
#print(n2([1]))





def load_mnist( path = "../../neural-networks-and-deep-learning/data/mnist.pkl.gz" ):

   import gzip
   import pickle

   with gzip.open(path, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      data = u.load()
   
   return data  # training_data, validation_data, test_data

# imshow(1-p[0][0][30].reshape((28,28)))


def reshape_mnist_data( data ):
   def dirac(n): d = np.zeros(10); d[n]=1; return d;
   return [ (x,dirac(n)) for x,n in zip(data[0],data[1]) ]


def dump_mnist_as_json( data, file_name ):

   import json

   def to_int_list(a): return [ int(255*x) for x in a.tolist() ]

   with open(file_name, 'w') as f:
      json.dump({"data":[{"image":to_int_list(img), "number":num.tolist()} for img,num in zip(data[0],data[1]) ]}, f)


def predict_digit( net, img ):

   imshow(img.reshape((28,28)))
   print("predicted: ", net(img))


#net = NeuralNet([ 784, 30, 10 ])


from matplotlib.cbook import flatten

def print_flat(text, range):
   print(text,end='')
   for n in flatten(range): print(n,', ',end='')
   print()




"""
training_data = [ (spice(1)+x,y) for x,y in np.repeat( [ ([ 0.], [1.]), ([ 1.], [0.]) ], 100, axis=0 ) ]
net = NeuralNet([1,1,1])
#net.weights[0][0][0] =  5.36
#net.biases[0][0]     = -3.17
#net.weights[1][0][0] = -7.38
#net.biases[1][0]     =  3.62

print("-------")
print(net([0]))
print(net([1]))
print(np.array(net.back_propagate([1],[0])).T)
net.step_gradient_descent([([1],[0])],1)
print("-------")
print(net([0]))
print(net([1]))
print(np.array(net.back_propagate([1],[0])).T)
"""

np.set_printoptions(precision=4)



net = NeuralNet([784, 30, 10])

training_data, validation_data, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
test_data = reshape_mnist_data(test_data)

print("initial output ----------------------")
print_flat( "", net(training_data[0][0]) )

print("training ----------------------------")
net.stochastic_gradient_descent( training_data, 1, 10, 3 )

for n in range(10):
   #print([ int(x>0.5) for x in net(test_data[n][0])], " vs ", test_data[n][1])
   print_flat( "", net(training_data[n][0]) )







"""
net = NeuralNet([784, 30, 10])
training_data, validation_data, test_data = load_mnist()
training_data = reshape_mnist_data(training_data)
test_data = reshape_mnist_data(test_data)
net.stochastic_gradient_descent( training_data, 1, 10, 3 )

#net.stochastic_gradient_descent( training_data, 30, 10, 3 )
#net.stochastic_gradient_descent( test_data, 30, 10, 3 )

#print("\n--- test data ------------------------------------")
#print_flat("in: ", test_data[0][0])
#print_flat("out: ", test_data[0][1])

print("\n--- feed -----------------------------------------")
print_flat("", net(test_data[0][0]) )

print("\n--- layers ---------------------------------------")
for b in net.biases: print_flat("b:  ",b)
print("\n--- delta layers ---------------------------------")
for b in net.back_propagate(test_data[0][0],test_data[0][1])[1]: print_flat("∂b: ",b)

print("\n*** training network *****************************")
for n in range(10):  net.step_gradient_descent( test_data, 1 )

print("\n--- layers ---------------------------------------")
for b in net.biases: print_flat("b:  ",b)
print("\n--- delta layers ---------------------------------")
for b in net.back_propagate(test_data[0][0],test_data[0][1])[1]: print_flat("∂b: ",b)

#for n in range(10):
#   print([ int(x>0.5) for x in net(test_data[n][0])], " vs ", test_data[n][1])
"""






"""
net = NeuralNet([3,2,2])

net.weights[0][0,0] = 1;
net.weights[0][0,1] = -2;
net.weights[0][0,2] = -3;
net.weights[0][1,0] = -4;
net.weights[0][1,1] = -5;
net.weights[0][1,2] = 6;

net.biases[0][0] = -3;
net.biases[0][1] = -4;

net.weights[1][0,0] = 1;
net.weights[1][0,1] = -20000;
net.weights[1][1,0] = -30000;
net.weights[1][1,1] = 4;

net.biases[1][0] = -5;
net.biases[1][1] = -6;


print( "feeding" )
print( net([1,2,3]) )


for n in range(2):

   dl = net.back_propagate( [1,2,3], [1,0] )

   print("delta layers ----------")
   for w,b in zip(dl[0],dl[1]):
      print("weights:   ", w)
      print("biases:    ", b)

   net.step_gradient_descent([([1,2,3], [1,0])], 3.0)
"""




"""
from matplotlib.cbook import flatten


net = NeuralNet([1000,40,10])
#net = NeuralNet([5,4,4,3])
#net = NeuralNet([1,3,2])

for w,b in zip( net.weights, net.biases ):
   w[:,:] =  0
   w[0,0] =  1
   w[2,2] =  1
   b[:]   = -1
   b[0]   = -2

input = np.zeros(1000)
#input = np.zeros(5)
#input = np.zeros(1)
input[0:2] = 1
#input = np.zeros(1)
#input[0] = 1
output = np.zeros(10)
#output = np.zeros(3)
#output = np.zeros(2)
output[0:2] = 1

for n in range(2):

   print("-------------------------------------------------")
   #print( "feeding:", end='' )
   #print( net(input) )

   dl = net.back_propagate(input, output)

   print("----------- layers -------------------------------")
   for w,b in zip(net.weights,net.biases):
      #print("weights:   ", w)
      #print("biases:    ", b)
      #print("weights:   ",end='')
      #for n in flatten(w.T): print(n,', ',end='')
      #print()
      print("biases:    ",end='')
      for n in flatten(b): print(n,', ',end='')
      print()

   print("----------- delta layers -------------------------")
   for w,b in zip(dl[0],dl[1]):
      #print("weights:   ", w)
      #print("biases:    ", b)
      #print("weights:   ",end='')
      #for n in flatten(w.T): print(n,', ',end='')
      #print()
      print("biases:    ",end='')
      for n in flatten(b): print(n,', ',end='')
      print()

   net.step_gradient_descent([(input, output)], 3.0)

"""