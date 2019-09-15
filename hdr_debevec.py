import numpy as np
import matplotlib.pyplot as mp_plt
from gsolve import gsolve

def plot_crf(crf_channel, C, Zmax):
  mp_plt.figure(figsize=(24,8))
  channel_names = ['red', 'green', 'blue']
  for ch in range(C):
    mp_plt.subplot(1,3,ch+1)
    mp_plt.plot(crf_channel[ch], np.arange(Zmax+1), color=channel_names[ch], linewidth=2)
    mp_plt.xlabel('log(X)')
    mp_plt.ylabel('Pixel intensity')
    mp_plt.title('CRF for {} channel'.format(channel_names[ch]))

  mp_plt.figure(figsize=(8,8))
  for ch in range(C):
    mp_plt.plot(crf_channel[ch], np.arange(Zmax+1), color=channel_names[ch], linewidth=2, label=channel_names[ch]+' channel')
  mp_plt.xlabel('log(X)')
  mp_plt.ylabel('Pixel intensity')
  mp_plt.title('Camera Response Function'.format(channel_names[ch]))
  
  mp_plt.legend()

def hdr_debevec(images, B, lambda_ = 50, num_px = 150):
  num_images = len(images)
  Zmin = 0
  Zmax = 255

  # image parameters
  H,W,C = images[0].shape

  # optmization parameters
  px_idx = np.random.choice(H*W, (num_px,), replace=False)

  # define pixel intensity weighting function w
  w = np.concatenate((np.arange(128) - Zmin, Zmax - np.arange(128,256)))

  # compute Z matrix
  Z = np.empty((num_px, num_images))
  crf_channel = []
  log_irrad_channel = []
  for ch in range(C):
    for j, image in enumerate(images):
      flat_image = image[:,:,ch].flatten()
      Z[:, j] = flat_image[px_idx]

    # get crf and irradiance for each color channel
    [crf, log_irrad] = gsolve(Z.astype('int32'), B, lambda_, w, Zmin, Zmax)
    crf_channel.append(crf)
    log_irrad_channel.append(log_irrad)
    
  plot_crf(crf_channel, C, Zmax)
  return [crf_channel, log_irrad_channel, w]
