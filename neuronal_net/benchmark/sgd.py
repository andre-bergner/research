# TODO
# gradient update must be normalized by batch size

import numpy

def stochastic_gradient_descent(minimizer, training_data, n_epochs=10, mini_batch_size=20):

   n = len(training_data)

   losses = []

   for n_epoch in range(n_epochs):

      numpy.random.shuffle(training_data)
      mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

      for n_batch, mini_batch in enumerate(mini_batches):

         avg_loss = 0.
         for input, expected in mini_batch:
            loss = minimizer(input, expected)
            avg_loss += loss

         losses.append(avg_loss / mini_batch_size)

         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch+1, n_epochs, 100.*float(n_batch)/len(mini_batches))
              , end="", flush=True)

   print("")

   return losses



def stochastic_gradient_descent2(minimizer, training_data, n_epochs=10, mini_batch_size=20):

   n_data = len(training_data[0])
   n_batches = n_data//mini_batch_size

   losses = []

   for n_epoch in range(n_epochs):

      for n_batch in range(n_batches):

         idx = numpy.random.randint(0, n_data, mini_batch_size)
         input_batch = training_data[0][idx]
         label_batch = training_data[1][idx]

         loss = minimizer(input_batch, label_batch)
         losses.append(loss / mini_batch_size)

         print( "\rEpoch {0}/{1}, {2:.0f}%   "
              . format(n_epoch+1, n_epochs, 100.*float(n_batch)/n_batches)
              , end="", flush=True)
   print("")

   return losses
