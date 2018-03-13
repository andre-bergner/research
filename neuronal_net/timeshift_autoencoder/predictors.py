import numpy as np

# transformations needed for the probabilistic model

def p2x(p):
   return np.inner(np.linspace(-1,1,len(p)), p)

def x2p(x, n=64, dev=.02):
   p = np.exp( -(np.linspace(-1,1,n)-x)**2 / dev**2 )
   return p / sum(p)



def predict_ar_model(model, start_frame, n_samples):
   frame_size = start_frame.shape[-1]
   result = start_frame.copy()
   frame = start_frame.copy()
   for _ in range(n_samples):
      result = np.concatenate([result, model.predict(frame.reshape(1,-1))[0]])
      frame = result[-frame_size:]
   return result

def predict_par_model(model, start_frame, n_samples):
   frame_size = start_frame.shape[-1]
   result = start_frame.copy()
   frame = start_frame.copy()
   for _ in range(n_samples):
      result = np.concatenate([result, [p2x(model.predict(frame.reshape(1,-1))[0])] ])
      frame = result[-frame_size:]
   return result


def generate_n_frames_from(model, frame, n_frames=10):
   for n in range(n_frames):
      frame = model.predict(frame)
      yield frame

def xfade_append(xs, ys, n_split):
   # expects xs,ys to have shape (N,...,1)
   # expects time in first dimension: N
   num_left = ys.shape[0] - n_split
   fade = np.linspace(0, 1, num_left)
   fade_tensor = np.tile(fade, list(ys.shape[1:]) + [1]).T
   xs[-num_left:] *= (1-fade_tensor)
   xs[-num_left:] += ys[:num_left] * fade_tensor
   return np.concatenate([xs, ys[-n_split:]], axis=0)

def predict_signal(model, start_frame, shift, n_samples):
   start_frame = start_frame.copy()
   frame_ = start_frame.reshape([1] + list(start_frame.shape))
   frames = np.array([f[0] for f in generate_n_frames_from(model, frame_, int(n_samples/shift))])
   pred_sig = start_frame.copy()
   for f in frames[0:]:
      pred_sig = xfade_append(pred_sig.T, f.T, shift).T
   return pred_sig


def predict_signal2(model, start_frame, shift, n_samples):
   start_frame = start_frame.copy()
   frame_size = start_frame.shape[-1]
   frame = start_frame.reshape([1] + list(start_frame.shape))
   pred_sig = start_frame.copy()
   for n in range(int(n_samples/shift)):
      frame = model.predict(frame)
      pred_sig = xfade_append(pred_sig, frame[0], shift)
      frame = pred_sig[-frame_size:].reshape([1] + list(start_frame.shape))
   return pred_sig


