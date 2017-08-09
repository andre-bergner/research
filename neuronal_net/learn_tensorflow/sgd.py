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
