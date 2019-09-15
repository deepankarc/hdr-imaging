### HDR Imaging

This library performs High Dynamic Range post-processing for a given set of images. The technique implemented is discussed in [1]. Broadly the steps involved in the process are:
1. Estimation of the Camera Response Function (CRF)
2. Computation of the irradiance map
3. Tone Mapping

Tone mapping is implemented in two ways - local and global. Global tone mapping has been implemented as discussed in [2].

#### Parameters

ROOT_DIR (String) = root folder which contains the folders for the source images.  
IMAGE_DIR (String) = name of the directory containing images.  
IMAGE_EXT (String) = image extension (eg. .jpg)  
COMPUTE_CRF (bool) = flag to compute CRF using supplied image.  

### Usage

`python run_hdr_image.py ROOT_DIR IMAGE_DIR IMAGE_EXT COMPUTE_CRF`

### Results

<img src="/images/1_4.png" alt="Original Image (Low Exposure)" width="337" height="445">     <img src="/images/32_1.png" alt="Original Image (Low Exposure)" width="337" height="445">
<pre>
                  Fig.1 - Original Images (Left - Low Exposure Image, Right - High Exposure Image) </pre>

<img src="/images/0_Calib_Chapel_CRF0.jpg" alt="Global Tonemapped HDR Image" width="337" height="445">     <img src="/images/0_Calib_Chapel_local_CRF0.jpg" alt="Local Tonemapped HDR Image" width="337" height="445">

<pre>
                  Fig.2 - HDR Images (Left - Global Tonemapping, Right - Local Tonemapping) </pre>

References

[1] Paul E. Debevec Jitendra Malik - Recovering High Dynamic Range Radiance Maps from Photographs (http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf)

[2] Reinhard, Erik and Stark, Michael and Shirley, Peter and Ferwerda, James - Photographic Tone Reproduction for Digital Images (http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf)
