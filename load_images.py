import glob, os
import numpy as np
import cv2

def load_images(image_dir, image_ext, root_dir):
  iter_items = glob.iglob(root_dir+image_dir+image_ext)

  images = []
  exposure_times = []
  for item in iter_items:
    images.append(cv2.cvtColor(cv2.imread(item), code=cv2.COLOR_BGR2RGB))
    fname = os.path.basename(item)
    num_ = int(fname[:-4].split('_')[0])
    den_ = int(fname[:-4].split('_')[-1])
    exposure_times.append(num_ / den_)

  images = [img for _,img in sorted(zip(exposure_times, images), key=lambda pair: pair[0], reverse=True)]
  B = np.log(sorted(exposure_times, reverse=True))
  
  return [images, B]
