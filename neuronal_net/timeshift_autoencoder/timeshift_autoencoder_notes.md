# Timeshift Autoencoder

## Abstract

Inspired by denoising autoencoder which learn to project onto a manifold a Timeshift Autoencoder (TAE) learns
to move along the manifold. If trained by a time series of a dynamical system it learns an embedding of the
manifold of the dynamical system's flow.

These are preliminary but very exciting results, no in depth discussion. Goal is to enable more research into
this direction.


Instead of learning a regressive process which learns to predict the next value we learn an embedding into
some latent space (encoder), projecting into the next state, and reconstructing the observed state from the latent
space (decoder)

This learned latent space hopfully approximates the system internals state space u[n+k] = f(u[n])

another perspective learning time shif operator (moving window through timee)

Inspired by Denoising Auto-Encoders, which learn to denoise a sample by projecting onto a manifold, the
timeshif AE learns to move a long a manifold in a certain direction (picutre). By learning the manifold,
the hope is that this system will be stable under pertubation as it always stays in the vicinity of the
manifold. The manifold itself would represent the orbit of the system


## Definitions

Timeshift operator E{n} x[k] := x[k+n]


## Points to discuss

* a sound (sample) is not a particular sample from a n-dim space but a manifold --> sovlves the size problem

* laveraging the fact that autoencoder learn manifolds, in particular DAE learn to project onto manifolds
* Dynamical systems
* Taken's embedding theorem
* construction of signal: averaging predictions
* finds perfect embedding in latent space (fills space entirely)
* learning multiple attractors
* linear vs nonlinear AE
* short-time learning gives already good results, due to averaging of projections
* lin. sys. with two frequencies should live in 3d space: torus can be embedded in such
   * works with 3D latent space!
   * interesting observation if only 2D latent space -> approxiamtes second freq in waveshape



## Comparison against existing methods

* AR-NN
* prob-AR-NN (wavenet)
* Approximating Differential Equation from embedded

For the sake of simplcity we compare oinly against a simple auto-regressive neural network that predicts
one sample at a time. Reason: see main differences in the setup and not fency priors, regularizations, etc.


## Application to different signals

* fm-modulated signal
* chaotic system
* spatio-temporal chaos
  * reduce dimensionality dramatically
  * TODO: strong sync --> show that 2 dimensions are enough
* chaotic map-iteration
* ARMA-process
* textures
* reals signals

## generation process

* averaged iteration vs direct iteration


## Regularization Techniques

* training simultanously [y(x), (y∘y)(x)] on [E{n}, E{2n}]x

## Conclusions

* more stable learning then existing techniques, thus faster training times
* better results with noisy and poorly sampled input (to be confirmed)
* better connection to dynamical system theory (state space, embedding)
* connection to insights of auto-encoders
* new regularization techniques
* allows to approximate system with lower dimensionality: AE controls dimension while still having a big window
  example partial differential equation having huge manifold (in theory infinite dim, in practice: num of nodes)
  negative Lyapunov exponents will contract flow



## Open Questions and Investigations

* small time step is worse!
* impact of noise with noisy signals
* impact of sampling rate
* long-term correleations, stiff equations, extreme events

