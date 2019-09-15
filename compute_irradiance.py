import numpy as np

def compute_irradiance(crf_channel, w, images, B):
  H,W,C = images[0].shape
  num_images = len(images)
  
  # irradiance map for each color channel
  irradiance_map = np.empty((H*W, C))
  for ch in range(C):
    crf = crf_channel[ch]
    num_ = np.empty((num_images, H*W)) 
    den_ = np.empty((num_images, H*W))
    for j in range(num_images):
      flat_image = (images[j][:,:,ch].flatten()).astype('int32')
      num_[j, :] = np.multiply(w[flat_image], crf[flat_image] - B[j])
      den_[j, :] = w[flat_image]

    irradiance_map[:, ch] = np.sum(num_, axis=0) / (np.sum(den_, axis=0) + 1e-6)

  irradiance_map = np.reshape(np.exp(irradiance_map), (H,W,C))
  
  return irradiance_map
