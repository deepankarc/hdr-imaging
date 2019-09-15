import numpy as np
import os, glob, sys
import cv2
import matplotlib.pyplot as mp_plt
from load_images import load_images
from compute_irradiance import compute_irradiance
from tonemap import reinhard_tonemap, plot_and_save, local_tonemap


def run_hdr(image_dir, image_ext, root_dir, COMPUTE_CRF, kwargs):
  if(len(kwargs) > 0):
    lambda_ = kwargs['lambda_']
    num_px =  kwargs['num_px']
    gamma = kwargs['gamma']
    alpha =  kwargs['alpha']
    gamma_local =  kwargs['gamma_local']
    saturation_local =  kwargs['saturation_local']
  
  [images, B] = load_images(image_dir, image_ext, root_dir)

  plot_idx = np.random.choice(len(images), (2,), replace=False)
  mp_plt.figure(figsize=(16,16))
  mp_plt.subplot(1,2,1)
  mp_plt.imshow(images[plot_idx[0]])
  mp_plt.title("Exposure time: {} secs".format(np.exp(B[plot_idx[0]])))
  mp_plt.subplot(1,2,2)
  mp_plt.imshow(images[plot_idx[1]])
  mp_plt.title("Exposure time: {} secs".format(np.exp(B[plot_idx[1]])))

  if(COMPUTE_CRF):
    [crf_channel, log_irrad_channel, w] = hdr_debevec(images, B, lambda_=lambda_, num_px=num_px)
    np.save(root_dir+"crf.npy", [crf_channel, log_irrad_channel, w])
  else:
    hdr_loc = kwargs['hdr_loc']
    [crf_channel, log_irrad_channel, w] = np.load(hdr_loc)
  irradiance_map = compute_irradiance(crf_channel, w, images, B)
  tonemapped_img = reinhard_tonemap(irradiance_map, gamma=gamma, alpha=alpha)
  plot_and_save(tonemapped_img, root_dir+image_dir[:-1], "Globally Tonemapped Image")
  local_tonemap(irradiance_map, root_dir+image_dir[:-1]+"_local", saturation=saturation_local, gamma=gamma_local)
  return [tonemapped_img, irradiance_map]
  
if __name__ == "__main__":
  ROOT_DIR = sys.argv[1]
  IMAGE_DIR = sys.argv[2]
  IMAGE_EXT = sys.argv[3]
  COMPUTE_CRF = sys.argv[4]
  kwargs = {'lambda_': 50, 'num_px': 150, 'gamma': 1 / 2.2, 'alpha': 0.35, 'hdr_loc':ROOT_DIR+"crf.npy", 
            'gamma_local':1.5, 'saturation_local':2.0}
  hdr_image, irmap = run_hdr(IMAGE_DIR, IMAGE_EXT, ROOT_DIR, COMPUTE_CRF, kwargs)
