
# TODO

## Restructuring Code

* create tools:
  * train with multiple epochs, learning rates...


## Research Ideas

* training AE
  * greedy vs joint parallel vs full vs stacked(?)   --> ask on StackExchange

* learning state space (wavenet-kind):
  * train like denoising-autoencoder
    * learn time-shift operator, i.e. {x[0],x[1],...,x[N]} -> {x[1],x[2],...,x[N+1]}
    * average overlapping part?

* auto-categorizer
  * N parallel auto-encoder
  * second categorzing network with softmax output ( input from x and AEs) P(c)
  * the AE outputs gets multiplied with the respective P(c), i.e. selecting one of the AEs

* deep waveforms
  → (pre-)train denoising autoencoder on basic shapes (triangles, rectangles, circles, etc.)
  → train DAE no simple clear sound (drums?) with noise added in audio domain

* wavelet autoencoder
  * create sliding auto-encoder:
    * perceptual field of length N is encoded by stream of size S
    * use convolutions instead of dense layers
      -> a dense (k->l) layer sitting on top of a conv layer can be implemented by
         using a conv layer with kernel size k and l channels
      -> a dense (k->l) layer sitting on top of a (simulated) dense layer can be implemented by
         using a conv layer with kernel size 1 and l channels (assuming k input channels)
  * learn simpler model: pure freq → learn w or (sinw,cosw)
  * add some proper statistics:
    * avg. error
    * gradients per layers
    * losses
    * test data, cross validation


* document insight: number of features must be small
* next steps:
  * learn postion for diff objects
  * learn diff objects at diff positions
     -->  two layers , 3 layers: stacked or 2 parallel + 1 final

* learn wavelets
  → use linear layers
  → use dyadic grid topology
  → use autoencoders and harmonic mixture signal

