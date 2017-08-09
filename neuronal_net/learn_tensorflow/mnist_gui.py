import numpy as np
import matplotlib.pyplot as plt


class DrawableArray:

   def __init__(self, ax):
      self.array = np.zeros((28, 28))

      image = ax.imshow(self.array, cmap='gist_gray_r', vmin=0, vmax=1, aspect='auto')
      self.canvas = image.figure.canvas
      self.image = image
      self.press = None

      self.cidpress = self.canvas.mpl_connect('button_press_event', self.on_press)
      self.cidrelease = self.canvas.mpl_connect('button_release_event', self.on_release)
      self.cidmotion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)


   def _set_pixel(self, x, y):
      self.array[int(y)  , int(x)]   += 1.*(1. - self.array[int(y)  , int(x)]   )
      self.array[int(y)-1, int(x)-1] += .04*(1. - self.array[int(y)-1, int(x)-1] )
      self.array[int(y)-1, int(x)]   += .06*(1. - self.array[int(y)-1, int(x)]   )
      self.array[int(y)-1, int(x)+1] += .04*(1. - self.array[int(y)-1, int(x)+1] )
      self.array[int(y)  , int(x)+1] += .06*(1. - self.array[int(y)  , int(x)+1] )
      self.array[int(y)+1, int(x)+1] += .04*(1. - self.array[int(y)+1, int(x)+1] )
      self.array[int(y)+1, int(x)]   += .06*(1. - self.array[int(y)+1, int(x)]   )
      self.array[int(y)+1, int(x)-1] += .04*(1. - self.array[int(y)+1, int(x)-1] )
      self.array[int(y)  , int(x)-1] += .06*(1. - self.array[int(y)  , int(x)-1] )

   def _update_image(self):
      self.image.set_data(self.array)
      self.canvas.draw()


   def on_press(self, event):
      if event.xdata is None: return
      if event.inaxes != self.image.axes: return

      #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
      #   (event.button, event.x, event.y, event.xdata, event.ydata))
      self.press = event.xdata, event.ydata
      self._set_pixel(event.xdata, event.ydata)
      self._update_image()

   def on_motion(self, event):
      if self.press is None: return
      if event.xdata is None: return

      x_last, y_last = self.press
      dx = event.xdata - x_last
      dy = event.ydata - y_last
      max_dxy = max(abs(dx),abs(dy)) + 5
      self.press = event.xdata, event.ydata

      for x,y in zip( np.linspace(x_last,event.xdata, max_dxy)
                    , np.linspace(y_last,event.ydata, max_dxy)):
         self._set_pixel(x,y)

      self._update_image()

   def on_release(self, event):
      self.press = None
      self.canvas.draw()

   def disconnect(self):
      self.canvas.mpl_disconnect(self.cidpress)
      self.canvas.mpl_disconnect(self.cidrelease)
      self.canvas.mpl_disconnect(self.cidmotion)

   def clear(self,_):
      self.array[:,:] = 0.
      self._update_image()


fig = plt.figure(figsize=(3,5))
img_ax = fig.add_axes([0.1, 0.4, 0.8, 0.55])
img_ax.axis('off')
draw_img = DrawableArray(img_ax)

bar_ax = fig.add_axes([0.1, 0.2, 0.6, 0.15])
bar_ax.xaxis.set_ticks(np.arange(10)+.4)
bar_ax.xaxis.set_ticklabels(np.arange(10))
#bar_ax.xaxis.set_ticks(np.arange(10)+.4, np.arange(10))
bar_ax.yaxis.set_ticks([0,1])

txt_ax = fig.add_axes([0.8, 0.2, 0.1, 0.15])
txt_ax.axis('off')
txt = txt_ax.text(-0.5,0.2,'X', size=50)

btn_clear = plt.Button(plt.axes([0.1, 0.03, 0.3, 0.1]), 'clear')
btn_clear.on_clicked(draw_img.clear)
btn_clear.on_clicked(lambda _: bar_ax.clear())
btn_clear.on_clicked(lambda _: txt.set_text('_'))

btn_predict = plt.Button(plt.axes([0.6, 0.03, 0.3, 0.1]), 'predict')
# btn_predict.on_clicked(draw_img.clear)


