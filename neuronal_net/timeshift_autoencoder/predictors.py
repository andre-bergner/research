import numpy as np


def predict_ar_model(model, start_frame, n_samples):
   frame_size = start_frame.shape[-1]
   result = start_frame
   frame = start_frame
   for _ in range(n_samples):
      result = np.concatenate([result, model.predict(frame.reshape(1,-1))[0]])
      frame = result[-frame_size:]
   return result

def generate_n_frames_from(model, frame, n_frames=10):
   for n in range(n_frames):
      frame = model.predict(frame)
      yield frame

def xfade_append(xs, ys, n_split):
   num_left = len(ys) - n_split
   fade = np.linspace(0, 1, num_left)
   xs[-num_left:] *= (1-fade)
   xs[-num_left:] += ys[:num_left] * fade
   return np.concatenate([xs, ys[-n_split:]])

def predict_signal(model, start_frame, shift, n_samples):
   frame_ = start_frame.reshape([1] + list(start_frame.shape))
   frames = np.array([f[0] for f in generate_n_frames_from(model, frame_, int(n_samples/shift))])
   pred_sig = start_frame
   for f in frames[0:]:
      pred_sig = xfade_append(pred_sig, f, shift)
   return pred_sig


def predict_signal2(model, start_frame, shift, n_samples):
   frame_size = start_frame.shape[-1]
   frame = start_frame.reshape([1] + list(start_frame.shape))
   pred_sig = start_frame
   for n in range(int(n_samples/shift)):
      frame = model.predict(frame)
      pred_sig = xfade_append(pred_sig, frame[0], shift)
      frame = pred_sig[-frame_size:].reshape([1] + list(start_frame.shape))
   return pred_sig
